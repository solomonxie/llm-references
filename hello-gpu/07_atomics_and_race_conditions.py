# $ venv/bin/python 07_atomics_and_race_conditions.py
#
# Goal: what happens without synchronization, made concrete instead of
# theoretical. Every thread here tries to increment the SAME counter — a
# plain `counter[0] = counter[0] + 1` is really three steps (read, add,
# write), and with a million threads genuinely running concurrently, any two
# threads that both READ the same old value before either WRITES its
# incremented one back cause one of those increments to be silently lost.
# At this thread count essentially EVERY thread collides this way (below:
# the final count comes out at 1, not some number closer to a million) —
# this isn't a rare edge case, it's what unsynchronized concurrent writes to
# one address do by default. Atomic operations fix it by making
# read-modify-write one indivisible hardware operation instead of three
# separate steps another thread can land between.
# Step 7: A visible race condition (lost updates) vs. an atomic fix

import Metal
import numpy as np

device = Metal.MTLCreateSystemDefaultDevice()

shader_source = """
#include <metal_stdlib>
using namespace metal;

kernel void racy_increment(device uint* counter [[buffer(0)]],
                            uint index [[thread_position_in_grid]])
{
    counter[0] = counter[0] + 1;  // read, add, write -- three separate, un-synchronized steps
}

kernel void atomic_increment(device atomic_uint* counter [[buffer(0)]],
                              uint index [[thread_position_in_grid]])
{
    // One indivisible hardware operation -- no other thread's
    // read/modify/write can interleave partway through this one.
    atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
}
"""

library, error = device.newLibraryWithSource_options_error_(shader_source, None, None)
if error is not None:
    raise RuntimeError(f"shader compile failed: {error}")
racy_pipeline, _ = device.newComputePipelineStateWithFunction_error_(
    library.newFunctionWithName_("racy_increment"), None
)
atomic_pipeline, _ = device.newComputePipelineStateWithFunction_error_(
    library.newFunctionWithName_("atomic_increment"), None
)
queue = device.newCommandQueue()


def run(pipeline, n: int) -> int:
    buf_counter = device.newBufferWithBytes_length_options_(np.uint32(0).tobytes(), 4, 0)

    command_buffer = queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_counter, 0, 0)
    max_threads = pipeline.maxTotalThreadsPerThreadgroup()
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n, 1, 1), Metal.MTLSizeMake(min(n, max_threads), 1, 1)
    )
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    return int(np.frombuffer(buf_counter.contents().as_buffer(4), dtype=np.uint32)[0])


n = 1_000_000
print(f"{n:,} threads, each trying to increment one shared counter\n")

for trial in range(3):
    result = run(racy_pipeline, n)
    lost = n - result
    print(f"racy_increment,   trial {trial}: counter = {result:>9,}   ({lost:,} updates lost)")

atomic_result = run(atomic_pipeline, n)
print(f"\natomic_increment: counter = {atomic_result:,}   (exactly {n:,}, every single trial, every run)")
