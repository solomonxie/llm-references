# $ venv/bin/python hello-gpu/py_gpu/01_device_and_buffers.py
#
# Goal: the GPU memory model, before any GPU code runs at all. A discrete
# GPU (most NVIDIA cards) has its own separate VRAM — every array has to be
# explicitly copied CPU -> GPU before use and GPU -> CPU to read results
# back, and that copy has real cost. Apple Silicon's GPU instead shares the
# same physical RAM as the CPU ("unified memory") — an MTLBuffer's bytes are
# directly readable/writable from Python via a pointer, no transfer step.
# This file only touches that memory model; no kernel runs yet (step 2).
# Step 1: MTLDevice + MTLBuffer -- the memory model, no kernel yet

import Metal
import numpy as np

# MTLCreateSystemDefaultDevice() is the entry point to everything else in
# this series — one object representing "the GPU," used to create buffers,
# compile shaders, and issue work.
device = Metal.MTLCreateSystemDefaultDevice()
print(f"device: {device.name()}")
print(f"unified memory: {device.hasUnifiedMemory()}")
print(f"max buffer length: {device.maxBufferLength() / 1e9:.1f} GB")

# A buffer is just a block of memory the GPU can address — options=0 below
# means MTLResourceStorageModeShared, the mode that gives BOTH CPU and GPU a
# view of the exact same bytes (only meaningful because of unified memory;
# a discrete GPU would need MTLResourceStorageModePrivate + an explicit copy
# instead).
data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
buffer = device.newBufferWithBytes_length_options_(data.tobytes(), data.nbytes, 0)
print(f"\ncreated a buffer holding {buffer.length()} bytes")

# .contents() hands back a raw pointer to those bytes — as_buffer() wraps it
# as something numpy can read directly, no copy.
view = np.frombuffer(buffer.contents().as_buffer(buffer.length()), dtype=np.float32)
print(f"read back through the buffer: {view}")

# The GPU hasn't touched this buffer at all yet — this is purely a CPU-side
# write/read through the SAME memory a kernel will later read/write into.
# Mutating the numpy view mutates the buffer directly, in place:
view[:] = view * 10
readback = np.frombuffer(buffer.contents().as_buffer(buffer.length()), dtype=np.float32)
print(f"after in-place *10: {readback}")

# An empty buffer, sized but not initialized — the shape a kernel's OUTPUT
# buffer usually starts as (steps 2+ write into one of these).
output_buffer = device.newBufferWithLength_options_(data.nbytes, 0)
print(f"\nempty output buffer: {output_buffer.length()} bytes, uninitialized contents: "
      f"{np.frombuffer(output_buffer.contents().as_buffer(output_buffer.length()), dtype=np.float32)}")
