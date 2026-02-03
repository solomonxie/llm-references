# $ venv/bin/python 05_threadgroup_shared_memory_reduction.py
#
# Goal: threadgroup memory — a small, fast pool of on-chip memory shared by
# every thread in one threadgroup (step 3), read/write in a few cycles
# instead of the hundreds it costs to round-trip to device (global) memory.
# Summing an array is the canonical use: instead of every thread fighting
# over one shared total, each threadgroup first reduces ITS OWN chunk in
# fast shared memory (a "tree reduction" — pairwise sums, halving the active
# thread count each round), leaving only one partial sum per threadgroup to
# combine afterward — device memory only gets touched once per threadgroup,
# not once per PAIR of elements.
# Step 5: Parallel reduction using threadgroup (shared) memory and a tree-sum pattern

import Metal
import numpy as np

device = Metal.MTLCreateSystemDefaultDevice()

THREADGROUP_SIZE = 256  # baked into the shader source below — must match the Python-side dispatch exactly

shader_source = f"""
#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum(device const float* input [[buffer(0)]],
                        device float* partial_sums [[buffer(1)]],
                        uint global_id [[thread_position_in_grid]],
                        uint local_id [[thread_position_in_threadgroup]],
                        uint group_id [[threadgroup_position_in_grid]])
{{
    // Fast on-chip memory, local to this ONE threadgroup — invisible to
    // every other threadgroup, gone once this dispatch finishes.
    threadgroup float shared_data[{THREADGROUP_SIZE}];
    shared_data[local_id] = input[global_id];

    // Every thread must finish its write above before ANY thread reads a
    // neighbor's value below — without this barrier, a fast thread could
    // read shared_data[local_id + stride] before that slot's write happens,
    // a race identical in spirit to step 7's, just inside one threadgroup.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction: round 1 sums pairs 256 apart (128 active threads),
    // round 2 sums pairs 64 apart using round 1's results (64 active
    // threads), ... down to 1 active thread holding the whole threadgroup's
    // sum — log2(256) = 8 rounds instead of 256 sequential additions.
    for (uint stride = {THREADGROUP_SIZE} / 2; stride > 0; stride >>= 1) {{
        if (local_id < stride) {{
            shared_data[local_id] += shared_data[local_id + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Only thread 0 needs to write out — shared_data[0] now holds this
    // WHOLE threadgroup's sum, one write to device memory per threadgroup
    // instead of one per element.
    if (local_id == 0) {{
        partial_sums[group_id] = shared_data[0];
    }}
}}
"""

library, error = device.newLibraryWithSource_options_error_(shader_source, None, None)
if error is not None:
    raise RuntimeError(f"shader compile failed: {error}")
pipeline, _ = device.newComputePipelineStateWithFunction_error_(
    library.newFunctionWithName_("reduce_sum"), None
)

n = THREADGROUP_SIZE * 1000  # exactly 1000 threadgroups' worth — no partial/ragged threadgroup to handle
data = np.random.rand(n).astype(np.float32)

buf_input = device.newBufferWithBytes_length_options_(data.tobytes(), data.nbytes, 0)
num_threadgroups = n // THREADGROUP_SIZE
buf_partial_sums = device.newBufferWithLength_options_(num_threadgroups * 4, 0)

queue = device.newCommandQueue()
command_buffer = queue.commandBuffer()
encoder = command_buffer.computeCommandEncoder()
encoder.setComputePipelineState_(pipeline)
encoder.setBuffer_offset_atIndex_(buf_input, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_partial_sums, 0, 1)
encoder.dispatchThreads_threadsPerThreadgroup_(
    Metal.MTLSizeMake(n, 1, 1), Metal.MTLSizeMake(THREADGROUP_SIZE, 1, 1)
)
encoder.endEncoding()
command_buffer.commit()
command_buffer.waitUntilCompleted()

partial_sums = np.frombuffer(buf_partial_sums.contents().as_buffer(num_threadgroups * 4), dtype=np.float32)

# The final combine — summing 1,000 partial sums — is cheap enough to just
# finish on the CPU; the GPU already did the expensive part (reducing 256,000
# elements down to 1,000). A fully GPU-side finish would run this same
# kernel AGAIN on partial_sums, or use an atomic add (step 7) instead.
gpu_total = partial_sums.sum()
cpu_total = data.sum()

print(f"n = {n:,} elements, {num_threadgroups} threadgroups of {THREADGROUP_SIZE}")
print(f"GPU (tree reduction -> {num_threadgroups} partial sums, finished on CPU): {gpu_total:.4f}")
print(f"CPU (numpy .sum() directly): {cpu_total:.4f}")
print(f"difference: {abs(gpu_total - cpu_total):.6f}  (may not be exactly 0 -- floating-point addition isn't")
print("associative, so summing in a different ORDER can give a slightly different rounding result)")
