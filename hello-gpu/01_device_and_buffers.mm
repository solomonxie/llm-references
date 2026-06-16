// $ clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation hello-gpu/01_device_and_buffers.mm -o /tmp/01_device_and_buffers && /tmp/01_device_and_buffers
//
// Goal: the GPU memory model, before any GPU code runs at all. A discrete
// GPU (most NVIDIA cards) has its own separate VRAM — every array has to be
// explicitly copied CPU -> GPU before use and GPU -> CPU to read results
// back, and that copy has real cost. Apple Silicon's GPU instead shares the
// same physical RAM as the CPU ("unified memory") — an MTLBuffer's bytes are
// directly readable/writable from C++ via a pointer, no transfer step. This
// file only touches that memory model; no kernel runs yet (step 2).
// Step 1: MTLDevice + MTLBuffer -- the memory model, no kernel yet

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <vector>

int main() {
    @autoreleasepool {
        // MTLCreateSystemDefaultDevice() is the entry point to everything
        // else in this series — one object representing "the GPU," used to
        // create buffers, compile shaders, and issue work.
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", device.name.UTF8String);
        printf("unified memory: %s\n", device.hasUnifiedMemory ? "true" : "false");
        printf("max buffer length: %.1f GB\n", device.maxBufferLength / 1e9);

        // A buffer is just a block of memory the GPU can address —
        // MTLResourceStorageModeShared is the mode that gives BOTH CPU and
        // GPU a view of the exact same bytes (only meaningful because of
        // unified memory; a discrete GPU would need MTLResourceStorageModePrivate
        // + an explicit copy instead).
        std::vector<float> data = {1, 2, 3, 4, 5};
        id<MTLBuffer> buffer = [device newBufferWithBytes:data.data()
                                                    length:data.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        printf("\ncreated a buffer holding %lu bytes\n", (unsigned long)buffer.length);

        // .contents hands back a raw pointer to those bytes — cast straight
        // to float*, no copy.
        float *view = (float *)buffer.contents;
        printf("read back through the buffer: [%.1f, %.1f, %.1f, %.1f, %.1f]\n",
               view[0], view[1], view[2], view[3], view[4]);

        // The GPU hasn't touched this buffer at all yet — this is purely a
        // CPU-side write/read through the SAME memory a kernel will later
        // read/write into. Mutating the pointer mutates the buffer directly,
        // in place:
        for (int i = 0; i < 5; i++) view[i] *= 10;
        printf("after in-place *10: [%.1f, %.1f, %.1f, %.1f, %.1f]\n",
               view[0], view[1], view[2], view[3], view[4]);

        // An empty buffer, sized but not initialized — the shape a kernel's
        // OUTPUT buffer usually starts as (steps 2+ write into one of these).
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:data.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
        float *outView = (float *)outputBuffer.contents;
        printf("\nempty output buffer: %lu bytes, uninitialized contents: [%.1f, %.1f, %.1f, %.1f, %.1f]\n",
               (unsigned long)outputBuffer.length, outView[0], outView[1], outView[2], outView[3], outView[4]);
    }
    return 0;
}
