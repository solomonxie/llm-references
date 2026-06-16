// $ clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation hello-gpu/08_cpu_vs_gpu_scaling.mm -o /tmp/08_cpu_vs_gpu_scaling && /tmp/08_cpu_vs_gpu_scaling
//
// Goal: the GPU isn't unconditionally faster — every dispatch pays fixed
// overhead (driver call, command buffer setup, synchronization) that a CPU
// call doesn't. At small n, that fixed cost dominates and the CPU wins; at
// large n, the GPU's massive parallelism (thousands of threads at once vs.
// the CPU's handful of cores) wins by a growing margin. This file runs the
// SAME per-element workload from step 4's compute_heavy_kernel at
// increasing sizes on both, to find that crossover directly instead of
// assuming "GPU = faster" unconditionally.
// Step 8: CPU vs. GPU, same workload, across sizes -- finding where the GPU actually wins

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
static id<MTLComputePipelineState> pipeline;

// Times the WHOLE round trip -- buffer creation, dispatch, and
// waitUntilCompleted -- not just the kernel itself. A real caller pays for
// all of it, so excluding setup/readback would understate the GPU's actual
// fixed cost and make the crossover point below look artificially small.
static double runGpu(const std::vector<float> &data) {
    auto start = std::chrono::high_resolution_clock::now();
    int n = (int)data.size();
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
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    volatile float *readback = (float *)bufOut.contents;
    (void)readback[0];
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// The identical 500-iteration formula, run over the whole array on the CPU
// -- the fair CPU baseline.
static double runCpu(const std::vector<float> &data) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> value = data;
    for (int iter = 0; iter < 500; iter++) {
        for (auto &v : value) v = v * 1.0000001f + 0.0000001f;
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();

        NSString *source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void compute_heavy(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           uint index [[thread_position_in_grid]])
{
    float value = input[index];
    for (int i = 0; i < 500; i++) {
        value = value * 1.0000001f + 0.0000001f;
    }
    output[index] = value;
}
)MSL";

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"compute_heavy"] error:&error];
        queue = [device newCommandQueue];

        auto randomData = [](int n) {
            std::vector<float> data(n);
            for (auto &v : data) v = (float)rand() / RAND_MAX;
            return data;
        };

        // One warm-up of each, matching earlier steps' note on first-dispatch cost.
        runGpu(randomData(1'000));
        runCpu(randomData(1'000));

        printf("%12s %14s %14s %10s\n", "n", "CPU", "GPU (Metal)", "faster");
        for (int n : {1'000, 10'000, 100'000, 1'000'000, 5'000'000, 20'000'000, 50'000'000}) {
            std::vector<float> data = randomData(n);
            double cpuTime = runCpu(data);
            double gpuTime = runGpu(data);
            const char *winner = gpuTime < cpuTime ? "GPU" : "CPU";
            printf("%12s %11.2f ms %11.2f ms %10s\n", commas(n).c_str(), cpuTime * 1000, gpuTime * 1000, winner);
        }

        // Two things to notice, not just one "crossover point":
        //  - At small n, BOTH sides are dominated by fixed per-call overhead
        //    rather than actual work -- the CPU side pays it 500 times (once
        //    per outer loop iteration above), the GPU side pays it once per
        //    dispatch. Which one wins down here can be noisy/inconsistent
        //    run to run.
        //  - At large n, the GPU's advantage isn't subtle or close --
        //    thousands of threads computing at once beats the CPU's handful
        //    of cores by a widening margin, not a fixed ratio, as n grows.
        printf("\nAt small n the two sides' FIXED overheads (500 loop iterations vs. one GPU dispatch) can go either\n");
        printf("way and aren't very stable run to run -- the real signal is the widening GPU lead at large n above.\n");
    }
    return 0;
}
