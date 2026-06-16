# $ venv/bin/python hello-gpu/04_memory_bandwidth_vs_compute.py
#
# Goal: not every kernel is limited by the same resource. A kernel that
# moves a lot of data but does little math per element is MEMORY-BOUND — its
# speed is capped by how fast bytes move between memory and the GPU cores,
# no matter how many cores you have. A kernel that does heavy math on data
# already in registers is COMPUTE-BOUND — its speed is capped by arithmetic
# throughput instead, and moving MORE data barely matters. "Arithmetic
# intensity" (FLOPs per byte moved) is what determines which regime a kernel
# is in — this file puts the same n through both kinds of kernel and times
# each, to make that distinction concrete instead of just asserted.
# Step 4: Memory-bound vs. compute-bound kernels, benchmarked side by side

import time

import Metal
import numpy as np

device = Metal.MTLCreateSystemDefaultDevice()

shader_source = """
#include <metal_stdlib>
using namespace metal;

// One read, one write per thread, ~zero math — as memory-bound as a kernel gets.
kernel void copy_kernel(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         uint index [[thread_position_in_grid]])
{
    output[index] = input[index];
}

// Same ONE read, ONE write per thread — identical data movement to
// copy_kernel — but hundreds of multiply-adds on a value already sitting
// in a register before that single write.
kernel void compute_heavy_kernel(device const float* input [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                                  uint index [[thread_position_in_grid]])
{
    float value = input[index];
    for (int i = 0; i < 500; i++) {
        value = value * 1.0000001f + 0.0000001f;  // arbitrary, just FLOPs that can't be optimized away
    }
    output[index] = value;
}
"""

library, error = device.newLibraryWithSource_options_error_(shader_source, None, None)
copy_pipeline, _ = device.newComputePipelineStateWithFunction_error_(
    library.newFunctionWithName_("copy_kernel"), None
)
compute_pipeline, _ = device.newComputePipelineStateWithFunction_error_(
    library.newFunctionWithName_("compute_heavy_kernel"), None
)
queue = device.newCommandQueue()


def run(pipeline, n: int) -> float:
    data = np.random.rand(n).astype(np.float32)
    buf_in = device.newBufferWithBytes_length_options_(data.tobytes(), data.nbytes, 0)
    buf_out = device.newBufferWithLength_options_(data.nbytes, 0)

    command_buffer = queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_in, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_out, 0, 1)
    max_threads = pipeline.maxTotalThreadsPerThreadgroup()
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n, 1, 1), Metal.MTLSizeMake(min(n, max_threads), 1, 1)
    )
    encoder.endEncoding()

    start = time.perf_counter()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    return time.perf_counter() - start


n = 20_000_000
run(copy_pipeline, 1_000)          # warm-up dispatch — first-ever kernel run pays a one-time driver/JIT cost
run(compute_pipeline, 1_000)

copy_time = run(copy_pipeline, n)
compute_time = run(compute_pipeline, n)

bytes_moved = n * 4 * 2  # one read + one write, 4 bytes each, per element — SAME for both kernels
flops = n * 500 * 2      # 500 iterations * (1 multiply + 1 add) per element — compute_heavy_kernel only

print(f"n = {n:,} elements\n")
print(f"copy_kernel:          {copy_time*1000:.2f} ms   ({bytes_moved / copy_time / 1e9:.1f} GB/s effective bandwidth)")
print(f"compute_heavy_kernel: {compute_time*1000:.2f} ms   ({flops / compute_time / 1e9:.1f} GFLOP/s effective throughput)")
print(f"\ncompute_heavy_kernel took {compute_time / copy_time:.1f}x longer, moving the EXACT SAME amount of data")
print("— the extra time is all arithmetic throughput, not memory bandwidth.")
