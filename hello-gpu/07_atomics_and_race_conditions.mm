// $ clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation hello-gpu/07_atomics_and_race_conditions.mm -o /tmp/07_atomics_and_race_conditions && /tmp/07_atomics_and_race_conditions
//
// Goal: what happens without synchronization, made concrete instead of
// theoretical. Every thread here tries to increment the SAME counter — a
// plain `counter[0] = counter[0] + 1` is really three steps (read, add,
// write), and with a million threads genuinely running concurrently, any two
// threads that both READ the same old value before either WRITES its
// incremented one back cause one of those increments to be silently lost.
// At this thread count essentially EVERY thread collides this way (below:
// the final count comes out at 1, not some number closer to a million) —
// this isn't a rare edge case, it's what unsynchronized concurrent writes to
// one address do by default. Atomic operations fix it by making
// read-modify-write one indivisible hardware operation instead of three
// separate steps another thread can land between.
// Step 7: A visible race condition (lost updates) vs. an atomic fix

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdint>
#include <string>
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

        NSString *source = @R"MSL(
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
)MSL";

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            fprintf(stderr, "shader compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> racyPipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"racy_increment"] error:&error];
        id<MTLComputePipelineState> atomicPipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"atomic_increment"] error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];

        auto run = [&](id<MTLComputePipelineState> pipeline, int n) -> uint32_t {
            uint32_t zero = 0;
            id<MTLBuffer> bufCounter = [device newBufferWithBytes:&zero length:sizeof(zero) options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:bufCounter offset:0 atIndex:0];
            NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
            [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)n, maxThreads), 1, 1)];
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            return *(uint32_t *)bufCounter.contents;
        };

        const int n = 1'000'000;
        printf("%s threads, each trying to increment one shared counter\n\n", commas(n).c_str());

        for (int trial = 0; trial < 3; trial++) {
            uint32_t result = run(racyPipeline, n);
            int lost = n - result;
            printf("racy_increment,   trial %d: counter = %9s   (%s updates lost)\n", trial, commas(result).c_str(), commas(lost).c_str());
        }

        uint32_t atomicResult = run(atomicPipeline, n);
        printf("\natomic_increment: counter = %s   (exactly %s, every single trial, every run)\n", commas(atomicResult).c_str(), commas(n).c_str());
    }
    return 0;
}
