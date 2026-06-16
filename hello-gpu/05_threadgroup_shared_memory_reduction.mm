// $ clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation hello-gpu/05_threadgroup_shared_memory_reduction.mm -o /tmp/05_threadgroup_shared_memory_reduction && /tmp/05_threadgroup_shared_memory_reduction
//
// Goal: threadgroup memory — a small, fast pool of on-chip memory shared by
// every thread in one threadgroup (step 3), read/write in a few cycles
// instead of the hundreds it costs to round-trip to device (global) memory.
// Summing an array is the canonical use: instead of every thread fighting
// over one shared total, each threadgroup first reduces ITS OWN chunk in
// fast shared memory (a "tree reduction" — pairwise sums, halving the active
// thread count each round), leaving only one partial sum per threadgroup to
// combine afterward — device memory only gets touched once per threadgroup,
// not once per PAIR of elements.
// Step 5: Parallel reduction using threadgroup (shared) memory and a tree-sum pattern

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        const int THREADGROUP_SIZE = 256;  // baked into the shader source below — must match the C++-side dispatch exactly

        std::string src = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum(device const float* input [[buffer(0)]],
                        device float* partial_sums [[buffer(1)]],
                        uint global_id [[thread_position_in_grid]],
                        uint local_id [[thread_position_in_threadgroup]],
                        uint group_id [[threadgroup_position_in_grid]])
{
    // Fast on-chip memory, local to this ONE threadgroup — invisible to
    // every other threadgroup, gone once this dispatch finishes.
    threadgroup float shared_data[THREADGROUP_SIZE_PLACEHOLDER];
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
    for (uint stride = THREADGROUP_SIZE_PLACEHOLDER / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_data[local_id] += shared_data[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Only thread 0 needs to write out — shared_data[0] now holds this
    // WHOLE threadgroup's sum, one write to device memory per threadgroup
    // instead of one per element.
    if (local_id == 0) {
        partial_sums[group_id] = shared_data[0];
    }
}
)MSL";
        size_t pos;
        while ((pos = src.find("THREADGROUP_SIZE_PLACEHOLDER")) != std::string::npos) {
            src.replace(pos, strlen("THREADGROUP_SIZE_PLACEHOLDER"), std::to_string(THREADGROUP_SIZE));
        }
        NSString *source = [NSString stringWithUTF8String:src.c_str()];

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            fprintf(stderr, "shader compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"reduce_sum"] error:&error];

        const int n = THREADGROUP_SIZE * 1000;  // exactly 1000 threadgroups' worth — no partial/ragged threadgroup to handle
        std::vector<float> data(n);
        for (int i = 0; i < n; i++) data[i] = (float)rand() / RAND_MAX;

        id<MTLBuffer> bufInput = [device newBufferWithBytes:data.data() length:n * sizeof(float) options:MTLResourceStorageModeShared];
        int numThreadgroups = n / THREADGROUP_SIZE;
        id<MTLBuffer> bufPartialSums = [device newBufferWithLength:numThreadgroups * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufInput offset:0 atIndex:0];
        [encoder setBuffer:bufPartialSums offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(THREADGROUP_SIZE, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float *partialSums = (float *)bufPartialSums.contents;

        // The final combine — summing 1,000 partial sums — is cheap enough to
        // just finish on the CPU; the GPU already did the expensive part
        // (reducing 256,000 elements down to 1,000). A fully GPU-side finish
        // would run this same kernel AGAIN on partial_sums, or use an atomic
        // add (step 7) instead.
        double gpuTotal = 0;
        for (int i = 0; i < numThreadgroups; i++) gpuTotal += partialSums[i];
        double cpuTotal = 0;
        for (int i = 0; i < n; i++) cpuTotal += data[i];

        printf("n = %s elements, %d threadgroups of %d\n", commas(n).c_str(), numThreadgroups, THREADGROUP_SIZE);
        printf("GPU (tree reduction -> %d partial sums, finished on CPU): %.4f\n", numThreadgroups, gpuTotal);
        printf("CPU (direct sum): %.4f\n", cpuTotal);
        printf("difference: %.6f  (may not be exactly 0 -- floating-point addition isn't\n", std::fabs(gpuTotal - cpuTotal));
        printf("associative, so summing in a different ORDER can give a slightly different rounding result)\n");
    }
    return 0;
}
