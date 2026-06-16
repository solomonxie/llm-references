// $ clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation hello-gpu/06_matrix_transpose_tiling.mm -o /tmp/06_matrix_transpose_tiling && /tmp/06_matrix_transpose_tiling
//
// Goal: memory ACCESS PATTERN matters as much as memory access COUNT. When
// neighboring threads read/write neighboring addresses, the GPU coalesces
// those into one wide memory transaction; when neighboring threads touch
// addresses far apart, each thread costs its own separate (slow) transaction.
// A naive matrix transpose reads contiguously but writes with a big stride
// (uncoalesced) — this file times that against a tiled version that does the
// transpose INSIDE threadgroup shared memory (step 5) instead, so both the
// read from and the write to device memory stay contiguous.
// Step 6: Naive vs. tiled matrix transpose -- coalesced vs. strided memory access, benchmarked

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        const int TILE_SIZE = 16;  // threadgroup is TILE_SIZE x TILE_SIZE = 256 threads -- comfortably under any device's per-threadgroup limit

        std::string src = R"MSL(
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
{
    if (gid.x >= width || gid.y >= height) return;
    output[gid.x * height + gid.y] = input[gid.y * width + gid.x];
}

kernel void transpose_tiled(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& width [[buffer(2)]],
                             constant uint& height [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]],
                             uint2 tid [[thread_position_in_threadgroup]],
                             uint2 group_id [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TILE_SIZE_PLACEHOLDER][TILE_SIZE_PLACEHOLDER];

    // Load: read input[row][col], one contiguous row per threadgroup-row --
    // coalesced, identical access pattern to transpose_naive's read.
    if (gid.x < width && gid.y < height) {
        tile[tid.y][tid.x] = input[gid.y * width + gid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // whole tile must be loaded before anyone reads a transposed slot

    // Store: swap which OUTPUT tile this threadgroup targets (group_id.x/y
    // flipped), then read the tile back TRANSPOSED (tid.x/y flipped) from
    // fast shared memory -- the actual transpose happens here, on-chip,
    // not via a strided device-memory write.
    uint out_x = group_id.y * TILE_SIZE_PLACEHOLDER + tid.x;
    uint out_y = group_id.x * TILE_SIZE_PLACEHOLDER + tid.y;
    if (out_x < height && out_y < width) {
        output[out_y * height + out_x] = tile[tid.x][tid.y];
    }
}
)MSL";
        size_t pos;
        while ((pos = src.find("TILE_SIZE_PLACEHOLDER")) != std::string::npos) {
            src.replace(pos, strlen("TILE_SIZE_PLACEHOLDER"), std::to_string(TILE_SIZE));
        }
        NSString *source = [NSString stringWithUTF8String:src.c_str()];

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            fprintf(stderr, "shader compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> naivePipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"transpose_naive"] error:&error];
        id<MTLComputePipelineState> tiledPipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"transpose_tiled"] error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];

        auto run = [&](id<MTLComputePipelineState> pipeline, const std::vector<float> &matrix, uint32_t width, uint32_t height, std::vector<float> &outResult) -> double {
            size_t bytes = matrix.size() * sizeof(float);
            id<MTLBuffer> bufIn = [device newBufferWithBytes:matrix.data() length:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufOut = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufWidth = [device newBufferWithBytes:&width length:sizeof(width) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufHeight = [device newBufferWithBytes:&height length:sizeof(height) options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:bufIn offset:0 atIndex:0];
            [encoder setBuffer:bufOut offset:0 atIndex:1];
            [encoder setBuffer:bufWidth offset:0 atIndex:2];
            [encoder setBuffer:bufHeight offset:0 atIndex:3];

            MTLSize grid = MTLSizeMake(width, height, 1);
            MTLSize threadgroup = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
            [encoder endEncoding];

            auto start = std::chrono::high_resolution_clock::now();
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            auto end = std::chrono::high_resolution_clock::now();

            outResult.resize(matrix.size());
            memcpy(outResult.data(), bufOut.contents, bytes);
            return std::chrono::duration<double>(end - start).count();
        };

        const int n = 2048;  // divisible by TILE_SIZE with no remainder -- keeps this file focused on the coalescing point, not edge-handling
        std::vector<float> matrix(n * n);
        for (auto &v : matrix) v = (float)rand() / RAND_MAX;

        std::vector<float> warmup;
        std::vector<float> subMatrix(matrix.begin(), matrix.begin() + TILE_SIZE * TILE_SIZE);
        run(naivePipeline, subMatrix, TILE_SIZE, TILE_SIZE, warmup);  // warm-up (see 04's note on first-dispatch cost)
        run(tiledPipeline, subMatrix, TILE_SIZE, TILE_SIZE, warmup);

        std::vector<float> naiveResult, tiledResult;
        double naiveTime = run(naivePipeline, matrix, n, n, naiveResult);
        double tiledTime = run(tiledPipeline, matrix, n, n, tiledResult);

        printf("%dx%d matrix transpose\n", n, n);
        printf("naive (strided writes):        %.2f ms\n", naiveTime * 1000);
        printf("tiled (shared-memory swap):    %.2f ms\n", tiledTime * 1000);
        printf("speedup: %.2fx\n", naiveTime / tiledTime);

        bool naiveCorrect = true, tiledCorrect = true;
        for (int row = 0; row < n && (naiveCorrect || tiledCorrect); row++) {
            for (int col = 0; col < n; col++) {
                float expected = matrix[col * n + row];
                if (naiveResult[row * n + col] != expected) naiveCorrect = false;
                if (tiledResult[row * n + col] != expected) tiledCorrect = false;
            }
        }
        printf("\nnaive correct:  %s\n", naiveCorrect ? "true" : "false");
        printf("tiled correct:  %s\n", tiledCorrect ? "true" : "false");
    }
    return 0;
}
