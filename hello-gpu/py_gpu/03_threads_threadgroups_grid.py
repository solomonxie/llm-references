# $ venv/bin/python hello-gpu/py_gpu/03_threads_threadgroups_grid.py
#
# Goal: unpack the "grid" / "threadgroup" / "thread" hierarchy step 2 used
# without explaining. Every dispatch launches a GRID of threads, split into
# equal-sized THREADGROUPS — threads in the same threadgroup run physically
# together (same GPU core, sharing fast on-chip memory — step 5 uses this),
# threads in different threadgroups may run at totally different times, in
# any order. Three built-in attributes tell a thread where it sits in that
# hierarchy; this kernel just writes all three out per thread so it's
# visible from Python instead of taken on faith.
# Step 3: The grid/threadgroup/thread hierarchy, made visible by having each thread report its own indices

import Metal
import numpy as np

device = Metal.MTLCreateSystemDefaultDevice()

shader_source = """
#include <metal_stdlib>
using namespace metal;

kernel void report_indices(device uint* grid_index [[buffer(0)]],
                            device uint* threadgroup_index [[buffer(1)]],
                            device uint* local_index [[buffer(2)]],
                            uint tig [[thread_position_in_grid]],
                            uint tgpig [[threadgroup_position_in_grid]],
                            uint tpitg [[thread_position_in_threadgroup]])
{
    grid_index[tig] = tig;                  // this thread's position among ALL threads in the dispatch
    threadgroup_index[tig] = tgpig;         // which threadgroup this thread belongs to
    local_index[tig] = tpitg;               // this thread's position WITHIN its own threadgroup
}
"""

library, error = device.newLibraryWithSource_options_error_(shader_source, None, None)
function = library.newFunctionWithName_("report_indices")
pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)

n = 16
threadgroup_width = 4  # deliberately small and hand-picked so 16 threads split into exactly 4 threadgroups

buf_grid = device.newBufferWithLength_options_(n * 4, 0)          # uint32 = 4 bytes each
buf_threadgroup = device.newBufferWithLength_options_(n * 4, 0)
buf_local = device.newBufferWithLength_options_(n * 4, 0)

queue = device.newCommandQueue()
command_buffer = queue.commandBuffer()
encoder = command_buffer.computeCommandEncoder()
encoder.setComputePipelineState_(pipeline)
encoder.setBuffer_offset_atIndex_(buf_grid, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_threadgroup, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_local, 0, 2)

encoder.dispatchThreads_threadsPerThreadgroup_(
    Metal.MTLSizeMake(n, 1, 1), Metal.MTLSizeMake(threadgroup_width, 1, 1)
)
encoder.endEncoding()
command_buffer.commit()
command_buffer.waitUntilCompleted()

grid_index = np.frombuffer(buf_grid.contents().as_buffer(n * 4), dtype=np.uint32)
threadgroup_index = np.frombuffer(buf_threadgroup.contents().as_buffer(n * 4), dtype=np.uint32)
local_index = np.frombuffer(buf_local.contents().as_buffer(n * 4), dtype=np.uint32)

print(f"{n} threads total, {threadgroup_width} threads per threadgroup -> {n // threadgroup_width} threadgroups\n")
print(f"{'grid index':>12} {'threadgroup':>12} {'local index':>12}")
for i in range(n):
    print(f"{grid_index[i]:>12} {threadgroup_index[i]:>12} {local_index[i]:>12}")

# The relationship this table always holds, at any grid/threadgroup size:
assert np.array_equal(grid_index, threadgroup_index * threadgroup_width + local_index)
print("\ngrid_index == threadgroup_index * threadgroup_width + local_index  (always true — verified above)")
