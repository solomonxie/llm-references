# $ venv/bin/python 02_first_kernel_vector_add.py
#
# Goal: the full round trip — write a GPU kernel, compile it, run it, read
# the result. A kernel is a function written from ONE thread's point of
# view; the same function body runs on many threads at once, each with a
# different `index`, computing one output element each. This is the whole
# idea behind GPU parallelism: instead of a CPU loop doing N additions one
# after another, N GPU threads each do exactly ONE addition, simultaneously.
# Step 2: A first compute kernel -- vector addition, compiled and dispatched from Python

import Metal
import numpy as np

device = Metal.MTLCreateSystemDefaultDevice()

# Metal Shading Language (MSL) — C++-like, compiled at runtime by the
# driver. `kernel` marks a GPU entry point; `[[buffer(N)]]` binds an
# argument to whatever buffer the host code attaches at index N (below);
# `[[thread_position_in_grid]]` is filled in automatically PER THREAD — this
# one line is what makes every thread do different work despite running the
# exact same compiled function.
shader_source = """
#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* result [[buffer(2)]],
                        uint index [[thread_position_in_grid]])
{
    result[index] = a[index] + b[index];
}
"""

library, error = device.newLibraryWithSource_options_error_(shader_source, None, None)
if error is not None:
    raise RuntimeError(f"shader compile failed: {error}")

function = library.newFunctionWithName_("add_arrays")
# A "pipeline state" is the compiled, GPU-ready form of one kernel function
# — created once, reused across many dispatches (steps 4+ do exactly that).
pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)

n = 1_000_000
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)

buffer_a = device.newBufferWithBytes_length_options_(a.tobytes(), a.nbytes, 0)
buffer_b = device.newBufferWithBytes_length_options_(b.tobytes(), b.nbytes, 0)
buffer_result = device.newBufferWithLength_options_(a.nbytes, 0)

# A command queue accepts work; a command buffer is one batch of work; a
# compute command encoder is where kernel dispatches actually get recorded
# into that batch — this three-level structure is fixed Metal boilerplate,
# the same every time (steps 3+ stop re-explaining it).
queue = device.newCommandQueue()
command_buffer = queue.commandBuffer()
encoder = command_buffer.computeCommandEncoder()
encoder.setComputePipelineState_(pipeline)
encoder.setBuffer_offset_atIndex_(buffer_a, 0, 0)
encoder.setBuffer_offset_atIndex_(buffer_b, 0, 1)
encoder.setBuffer_offset_atIndex_(buffer_result, 0, 2)

# "Grid" = total threads to launch (one per output element, n of them here);
# "threadgroup" = threads that run together in one physically co-located
# batch (step 3 digs into why that grouping exists and matters).
grid_size = Metal.MTLSizeMake(n, 1, 1)
max_threads = pipeline.maxTotalThreadsPerThreadgroup()
threadgroup_size = Metal.MTLSizeMake(min(n, max_threads), 1, 1)
encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
encoder.endEncoding()

command_buffer.commit()                # submit the work to the GPU — asynchronous, returns immediately
command_buffer.waitUntilCompleted()    # block until it's actually done, so the readback below is safe

result = np.frombuffer(buffer_result.contents().as_buffer(a.nbytes), dtype=np.float32)

print(f"n = {n:,}")
print(f"a[:5]      = {a[:5]}")
print(f"b[:5]      = {b[:5]}")
print(f"result[:5] = {result[:5]}")
print(f"matches numpy (a + b): {np.allclose(result, a + b)}")
