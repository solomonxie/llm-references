# $ venv/bin/python 06_matrix_transpose_tiling.py
#
# Goal: memory ACCESS PATTERN matters as much as memory access COUNT. When
# neighboring threads read/write neighboring addresses, the GPU coalesces
# those into one wide memory transaction; when neighboring threads touch
# addresses far apart, each thread costs its own separate (slow) transaction.
# A naive matrix transpose reads contiguously but writes with a big stride
# (uncoalesced) — this file times that against a tiled version that does the
# transpose INSIDE threadgroup shared memory (step 5) instead, so both the
# read from and the write to device memory stay contiguous.
# Step 6: Naive vs. tiled matrix transpose -- coalesced vs. strided memory access, benchmarked

import time

import Metal
import numpy as np

device = Metal.MTLCreateSystemDefaultDevice()

TILE_SIZE = 16  # threadgroup is TILE_SIZE x TILE_SIZE = 256 threads -- comfortably under any device's per-threadgroup limit

shader_source = f"""
#include <metal_stdlib>
using namespace metal;

// Reads input[row][col] -- consecutive threads (varying col) read
// CONSECUTIVE addresses, coalesced. Writes output[col][row] -- consecutive
// threads now write addresses `height` elements apart, one wide
// transaction becomes `width` separate narrow ones.
kernel void transpose_naive(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& width [[buffer(2)]],
                             constant uint& height [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]])
{{
    if (gid.x >= width || gid.y >= height) return;
    output[gid.x * height + gid.y] = input[gid.y * width + gid.x];
}}

kernel void transpose_tiled(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& width [[buffer(2)]],
                             constant uint& height [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]],
                             uint2 tid [[thread_position_in_threadgroup]],
                             uint2 group_id [[threadgroup_position_in_grid]])
{{
    threadgroup float tile[{TILE_SIZE}][{TILE_SIZE}];

    // Load: read input[row][col], one contiguous row per threadgroup-row --
    // coalesced, identical access pattern to transpose_naive's read.
    if (gid.x < width && gid.y < height) {{
        tile[tid.y][tid.x] = input[gid.y * width + gid.x];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);  // whole tile must be loaded before anyone reads a transposed slot

    // Store: swap which OUTPUT tile this threadgroup targets (group_id.x/y
    // flipped), then read the tile back TRANSPOSED (tid.x/y flipped) from
    // fast shared memory -- the actual transpose happens here, on-chip,
    // not via a strided device-memory write.
    uint out_x = group_id.y * {TILE_SIZE} + tid.x;
    uint out_y = group_id.x * {TILE_SIZE} + tid.y;
    if (out_x < height && out_y < width) {{
        output[out_y * height + out_x] = tile[tid.x][tid.y];
    }}
}}
"""

library, error = device.newLibraryWithSource_options_error_(shader_source, None, None)
if error is not None:
    raise RuntimeError(f"shader compile failed: {error}")
naive_pipeline, _ = device.newComputePipelineStateWithFunction_error_(
    library.newFunctionWithName_("transpose_naive"), None
)
tiled_pipeline, _ = device.newComputePipelineStateWithFunction_error_(
    library.newFunctionWithName_("transpose_tiled"), None
)
queue = device.newCommandQueue()


def run(pipeline, matrix: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = matrix.shape
    buf_in = device.newBufferWithBytes_length_options_(matrix.tobytes(), matrix.nbytes, 0)
    buf_out = device.newBufferWithLength_options_(matrix.nbytes, 0)
    buf_width = device.newBufferWithBytes_length_options_(np.uint32(width).tobytes(), 4, 0)
    buf_height = device.newBufferWithBytes_length_options_(np.uint32(height).tobytes(), 4, 0)

    command_buffer = queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_in, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_out, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_width, 0, 2)
    encoder.setBuffer_offset_atIndex_(buf_height, 0, 3)

    grid = Metal.MTLSizeMake(width, height, 1)
    threadgroup = Metal.MTLSizeMake(TILE_SIZE, TILE_SIZE, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid, threadgroup)
    encoder.endEncoding()

    start = time.perf_counter()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    elapsed = time.perf_counter() - start

    result = np.frombuffer(buf_out.contents().as_buffer(matrix.nbytes), dtype=np.float32).reshape(width, height)
    return result, elapsed


n = 2048  # divisible by TILE_SIZE with no remainder -- keeps this file focused on the coalescing point, not edge-handling
matrix = np.random.rand(n, n).astype(np.float32)

run(naive_pipeline, matrix[:TILE_SIZE, :TILE_SIZE])  # warm-up (see 04's note on first-dispatch cost)
run(tiled_pipeline, matrix[:TILE_SIZE, :TILE_SIZE])

naive_result, naive_time = run(naive_pipeline, matrix)
tiled_result, tiled_time = run(tiled_pipeline, matrix)

print(f"{n}x{n} matrix transpose")
print(f"naive (strided writes):        {naive_time*1000:.2f} ms")
print(f"tiled (shared-memory swap):    {tiled_time*1000:.2f} ms")
print(f"speedup: {naive_time / tiled_time:.2f}x")

expected = matrix.T
print(f"\nnaive correct:  {np.array_equal(naive_result, expected)}")
print(f"tiled correct:  {np.array_equal(tiled_result, expected)}")
