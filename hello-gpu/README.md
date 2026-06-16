# hello-gpu

Goal: GPU programming fundamentals — the parallel-computing techniques underneath any GPU
workload, deliberately separate from ML inference/training (see `hello-inference`/`hello-neuralnet`
for that side). Uses real Metal compute kernels (Apple Silicon's native GPU API) dispatched from
Objective-C++ rather than a framework like PyTorch that hides the kernel/thread/memory model this
series is specifically about. A Python/PyObjC variant of the same eight lessons lives in `py_gpu/`.

Each file is a complete, standalone, runnable `.mm` (Objective-C++) program.

## Setup

```sh
# from the repo root
clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation \
    hello-gpu/01_device_and_buffers.mm -o /tmp/01_device_and_buffers && /tmp/01_device_and_buffers
```

Each file's own compile-and-run command is its first line. Objective-C++ (not pure C++) because
that's literally how Metal ships — `Metal.framework`/`Foundation.framework` are Objective-C APIs;
no external dependency to vendor, just Xcode's Command Line Tools.

Requires a Mac with Apple Silicon (or any Metal 2+ GPU) — no CUDA/NVIDIA hardware needed, and
nothing here needs a model download or a running Ollama server.

## Notes

- Every concept here (grid/threadgroup/thread, shared memory, coalescing, atomics) has a direct
  CUDA equivalent (grid/block/thread, shared memory, coalescing, atomics) — same ideas, different
  API vocabulary, since Metal is what's actually available on this hardware.
- `04`'s and `08`'s exact timings are hardware- and system-load-dependent — the RELATIVE
  comparisons (compute-heavy vs. copy; CPU vs. GPU at growing n) are the point, not the absolute
  millisecond values.
