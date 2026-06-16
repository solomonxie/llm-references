// $ clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation hello-gpu/04_memory_bandwidth_vs_compute.mm -o /tmp/04_memory_bandwidth_vs_compute && /tmp/04_memory_bandwidth_vs_compute
//
// Goal: not every kernel is limited by the same resource. A kernel that
// moves a lot of data but does little math per element is MEMORY-BOUND — its
// speed is capped by how fast bytes move between memory and the GPU cores,
// no matter how many cores you have. A kernel that does heavy math on data
// already in registers is COMPUTE-BOUND — its speed is capped by arithmetic
// throughput instead, and moving MORE data barely matters. "Arithmetic
// intensity" (FLOPs per byte moved) is what determines which regime a kernel
// is in — this file puts the same n through both kinds of kernel and times
// each, to make that distinction concrete instead of just asserted.
// Step 4: Memory-bound vs. compute-bound kernels, benchmarked side by side

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

static std::string commas(long long n) {
    std::string s = std::to_string(n), out;
    int count = 0;
    for (int i = (int)s.size() - 1; i >= 0; i--) {
        out.push_back(s[i]);
        if (++count % 3 == 0 && i != 0) out.push_back(',');
    }
    std::reverse(out.begin(), out.end());
    return out;
}

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;

static double run(id<MTLComputePipelineState> pipeline, int n) {
    std::vector<float> data(n);
    for (int i = 0; i < n; i++) data[i] = (float)rand() / RAND_MAX;

    id<MTLBuffer> bufIn = [device newBufferWithBytes:data.data() length:n * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufOut = [device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:bufIn offset:0 atIndex:0];
    [encoder setBuffer:bufOut offset:0 atIndex:1];
    NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
    [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)n, maxThreads), 1, 1)];
    [encoder endEncoding];

    auto start = std::chrono::high_resolution_clock::now();
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();

        NSString *source = @R"MSL(
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
)MSL";

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            fprintf(stderr, "shader compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> copyPipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"copy_kernel"] error:&error];
        id<MTLComputePipelineState> computePipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"compute_heavy_kernel"] error:&error];
        queue = [device newCommandQueue];

        const int n = 20'000'000;
        run(copyPipeline, 1'000);          // warm-up dispatch — first-ever kernel run pays a one-time driver/JIT cost
        run(computePipeline, 1'000);

        double copyTime = run(copyPipeline, n);
        double computeTime = run(computePipeline, n);

        long long bytesMoved = (long long)n * 4 * 2;  // one read + one write, 4 bytes each, per element — SAME for both kernels
        long long flops = (long long)n * 500 * 2;      // 500 iterations * (1 multiply + 1 add) per element — compute_heavy_kernel only

        printf("n = %s elements\n\n", commas(n).c_str());
        printf("copy_kernel:          %.2f ms   (%.1f GB/s effective bandwidth)\n", copyTime * 1000, bytesMoved / copyTime / 1e9);
        printf("compute_heavy_kernel: %.2f ms   (%.1f GFLOP/s effective throughput)\n", computeTime * 1000, flops / computeTime / 1e9);
        printf("\ncompute_heavy_kernel took %.1fx longer, moving the EXACT SAME amount of data\n", computeTime / copyTime);
        printf("— the extra time is all arithmetic throughput, not memory bandwidth.\n");
    }
    return 0;
}
