// $ clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation hello-gpu/02_first_kernel_vector_add.mm -o /tmp/02_first_kernel_vector_add && /tmp/02_first_kernel_vector_add
//
// Goal: the full round trip — write a GPU kernel, compile it, run it, read
// the result. A kernel is a function written from ONE thread's point of
// view; the same function body runs on many threads at once, each with a
// different `index`, computing one output element each. This is the whole
// idea behind GPU parallelism: instead of a CPU loop doing N additions one
// after another, N GPU threads each do exactly ONE addition, simultaneously.
// Step 2: A first compute kernel -- vector addition, compiled and dispatched from C++

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
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

        // Metal Shading Language (MSL) — C++-like, compiled at runtime by
        // the driver. `kernel` marks a GPU entry point; `[[buffer(N)]]`
        // binds an argument to whatever buffer the host code attaches at
        // index N (below); `[[thread_position_in_grid]]` is filled in
        // automatically PER THREAD — this one line is what makes every
        // thread do different work despite running the exact same compiled
        // function.
        NSString *source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* result [[buffer(2)]],
                        uint index [[thread_position_in_grid]])
{
    result[index] = a[index] + b[index];
}
)MSL";

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            fprintf(stderr, "shader compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"add_arrays"];
        // A "pipeline state" is the compiled, GPU-ready form of one kernel
        // function — created once, reused across many dispatches (steps 4+
        // do exactly that).
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

        const int n = 1'000'000;
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; i++) {
            a[i] = (float)rand() / RAND_MAX;
            b[i] = (float)rand() / RAND_MAX;
        }

        id<MTLBuffer> bufA = [device newBufferWithBytes:a.data() length:n * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:b.data() length:n * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufResult = [device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];

        // A command queue accepts work; a command buffer is one batch of
        // work; a compute command encoder is where kernel dispatches
        // actually get recorded into that batch — this three-level
        // structure is fixed Metal boilerplate, the same every time (steps
        // 3+ stop re-explaining it).
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufResult offset:0 atIndex:2];

        // "Grid" = total threads to launch (one per output element, n of
        // them here); "threadgroup" = threads that run together in one
        // physically co-located batch (step 3 digs into why that grouping
        // exists and matters).
        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
        MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)n, maxThreads), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];                  // submit the work to the GPU — asynchronous, returns immediately
        [commandBuffer waitUntilCompleted];       // block until it's actually done, so the readback below is safe

        float *result = (float *)bufResult.contents;

        bool matches = true;
        for (int i = 0; i < n; i++) {
            if (std::fabs(result[i] - (a[i] + b[i])) > 1e-5f) { matches = false; break; }
        }

        printf("n = %s\n", commas(n).c_str());
        printf("a[:5]      = [%.4f, %.4f, %.4f, %.4f, %.4f]\n", a[0], a[1], a[2], a[3], a[4]);
        printf("b[:5]      = [%.4f, %.4f, %.4f, %.4f, %.4f]\n", b[0], b[1], b[2], b[3], b[4]);
        printf("result[:5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n", result[0], result[1], result[2], result[3], result[4]);
        printf("matches CPU (a + b): %s\n", matches ? "true" : "false");
    }
    return 0;
}
