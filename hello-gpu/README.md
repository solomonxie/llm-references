# hello-gpu

Goal: GPU programming fundamentals — the parallel-computing techniques underneath any GPU
workload, deliberately separate from ML inference/training (see `hello-inference`/`hello-neuralnet`
for that side). Uses real Metal compute kernels (Apple Silicon's native GPU API, dispatched from
Python via PyObjC) rather than a framework like PyTorch that hides the kernel/thread/memory model
this series is specifically about.

Each file is a complete, standalone, runnable script.

| File | Demonstrates |
|---|---|
| `01_device_and_buffers.py` | `MTLDevice`/`MTLBuffer` and Apple Silicon's unified-memory model — no kernel yet |
| `02_first_kernel_vector_add.py` | Writing, compiling, and dispatching a first real kernel |
| `03_threads_threadgroups_grid.py` | The grid/threadgroup/thread hierarchy, made visible by having each thread report its own indices |
| `04_memory_bandwidth_vs_compute.py` | Memory-bound vs. compute-bound kernels, benchmarked side by side |
| `05_threadgroup_shared_memory_reduction.py` | Parallel reduction using threadgroup (shared) memory and a tree-sum pattern |
| `06_matrix_transpose_tiling.py` | Naive (strided) vs. tiled (coalesced) matrix transpose, benchmarked |
| `07_atomics_and_race_conditions.py` | A visible race condition (lost updates) vs. an atomic fix |
| `08_cpu_vs_gpu_scaling.py` | The same workload on CPU vs. GPU across sizes — where the GPU actually wins, and why it isn't unconditional |

## Setup

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_device_and_buffers.py
```

Requires a Mac with Apple Silicon (or any Metal 2+ GPU) — no CUDA/NVIDIA hardware needed, and
nothing here needs a model download or a running Ollama server.

## Notes

- Every concept here (grid/threadgroup/thread, shared memory, coalescing, atomics) has a direct
  CUDA equivalent (grid/block/thread, shared memory, coalescing, atomics) — same ideas, different
  API vocabulary, since Metal is what's actually available on this hardware.
- `04`'s and `08`'s exact timings are hardware- and system-load-dependent — the RELATIVE
  comparisons (compute-heavy vs. copy; CPU vs. GPU at growing n) are the point, not the absolute
  millisecond values.
