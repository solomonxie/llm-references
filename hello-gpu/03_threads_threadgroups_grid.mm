// $ clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation hello-gpu/03_threads_threadgroups_grid.mm -o /tmp/03_threads_threadgroups_grid && /tmp/03_threads_threadgroups_grid
//
// Goal: unpack the "grid" / "threadgroup" / "thread" hierarchy step 2 used
// without explaining. Every dispatch launches a GRID of threads, split into
// equal-sized THREADGROUPS — threads in the same threadgroup run physically
// together (same GPU core, sharing fast on-chip memory — step 5 uses this),
// threads in different threadgroups may run at totally different times, in
// any order. Three built-in attributes tell a thread where it sits in that
// hierarchy; this kernel just writes all three out per thread so it's
// visible from C++ instead of taken on faith.
// Step 3: The grid/threadgroup/thread hierarchy, made visible by having each thread report its own indices

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cassert>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        NSString *source = @R"MSL(
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
)MSL";

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"report_indices"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

        const int n = 16;
        const int threadgroupWidth = 4;  // deliberately small and hand-picked so 16 threads split into exactly 4 threadgroups

        id<MTLBuffer> bufGrid = [device newBufferWithLength:n * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufThreadgroup = [device newBufferWithLength:n * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufLocal = [device newBufferWithLength:n * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufGrid offset:0 atIndex:0];
        [encoder setBuffer:bufThreadgroup offset:0 atIndex:1];
        [encoder setBuffer:bufLocal offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadgroupWidth, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        uint32_t *gridIndex = (uint32_t *)bufGrid.contents;
        uint32_t *threadgroupIndex = (uint32_t *)bufThreadgroup.contents;
        uint32_t *localIndex = (uint32_t *)bufLocal.contents;

        printf("%d threads total, %d threads per threadgroup -> %d threadgroups\n\n", n, threadgroupWidth, n / threadgroupWidth);
        printf("  grid index  threadgroup  local index\n");
        for (int i = 0; i < n; i++) {
            printf("%12u %12u %12u\n", gridIndex[i], threadgroupIndex[i], localIndex[i]);
        }

        // The relationship this table always holds, at any grid/threadgroup size:
        for (int i = 0; i < n; i++) {
            assert(gridIndex[i] == threadgroupIndex[i] * threadgroupWidth + localIndex[i]);
        }
        printf("\ngrid_index == threadgroup_index * threadgroup_width + local_index  (always true — verified above)\n");
    }
    return 0;
}
