# $ venv/bin/python hello-gpu/08_cpu_vs_gpu_scaling.py
#
# Goal: the GPU isn't unconditionally faster — every dispatch pays fixed
# overhead (driver call, command buffer setup, synchronization) that a CPU
# call doesn't. At small n, that fixed cost dominates and the CPU wins; at
# large n, the GPU's massive parallelism (thousands of threads at once vs.
# numpy's CPU-vectorized handful of cores) wins by a growing margin. This
# file runs the SAME per-element workload from step 4's compute_heavy_kernel
# at increasing sizes on both, to find that crossover directly instead of
# assuming "GPU = faster" unconditionally.
# Step 8: CPU vs. GPU, same workload, across sizes -- finding where the GPU actually wins

import time

import Metal
import numpy as np

device = Metal.MTLCreateSystemDefaultDevice()

shader_source = """
#include <metal_stdlib>
using namespace metal;

kernel void compute_heavy(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           uint index [[thread_position_in_grid]])
{
    float value = input[index];
    for (int i = 0; i < 500; i++) {
        value = value * 1.0000001f + 0.0000001f;
    }
    output[index] = value;
}
"""

library, error = device.newLibraryWithSource_options_error_(shader_source, None, None)
pipeline, _ = device.newComputePipelineStateWithFunction_error_(
    library.newFunctionWithName_("compute_heavy"), None
)
queue = device.newCommandQueue()


def run_gpu(data: np.ndarray) -> float:
    """Times the WHOLE round trip -- buffer creation, dispatch, and
    waitUntilCompleted -- not just the kernel itself. A real caller pays for
    all of it, so excluding setup/readback would understate the GPU's actual
    fixed cost and make the crossover point below look artificially small."""
    start = time.perf_counter()
    buf_in = device.newBufferWithBytes_length_options_(data.tobytes(), data.nbytes, 0)
    buf_out = device.newBufferWithLength_options_(data.nbytes, 0)

    command_buffer = queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_in, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_out, 0, 1)
    max_threads = pipeline.maxTotalThreadsPerThreadgroup()
    n = len(data)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n, 1, 1), Metal.MTLSizeMake(min(n, max_threads), 1, 1)
    )
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    _ = np.frombuffer(buf_out.contents().as_buffer(data.nbytes), dtype=np.float32)
    return time.perf_counter() - start


def run_cpu(data: np.ndarray) -> float:
    """The identical 500-iteration formula, vectorized across the whole
    array with numpy (which itself uses CPU SIMD instructions under the
    hood) -- the fair CPU baseline, not a naive per-element Python loop."""
    start = time.perf_counter()
    value = data.copy()
    for _ in range(500):
        value = value * 1.0000001 + 0.0000001
    return time.perf_counter() - start


# One warm-up of each, matching earlier steps' note on first-dispatch cost
# (and, on the CPU side, letting numpy settle into any lazy setup it does).
run_gpu(np.random.rand(1_000).astype(np.float32))
run_cpu(np.random.rand(1_000).astype(np.float32))

print(f"{'n':>12} {'CPU (numpy)':>14} {'GPU (Metal)':>14} {'faster':>10}")
for n in [1_000, 10_000, 100_000, 1_000_000, 5_000_000, 20_000_000, 50_000_000]:
    data = np.random.rand(n).astype(np.float32)
    cpu_time = run_cpu(data)
    gpu_time = run_gpu(data)
    winner = "GPU" if gpu_time < cpu_time else "CPU"
    print(f"{n:>12,} {cpu_time*1000:>11.2f} ms {gpu_time*1000:>11.2f} ms {winner:>10}")

# Two things to notice, not just one "crossover point":
#  - At small n, BOTH sides are dominated by fixed per-call overhead rather
#    than actual work -- the CPU side pays it 500 times (once per numpy
#    call in the loop above), the GPU side pays it once per dispatch. Which
#    one wins down here can be noisy/inconsistent run to run.
#  - At large n, the GPU's advantage isn't subtle or close -- thousands of
#    threads computing at once beats numpy's CPU-vectorized handful of
#    cores by a widening margin, not a fixed ratio, as n grows.
print("\nAt small n the two sides' FIXED overheads (500 Python/numpy calls vs. one GPU dispatch) can go either")
print("way and aren't very stable run to run -- the real signal is the widening GPU lead at large n above.")
