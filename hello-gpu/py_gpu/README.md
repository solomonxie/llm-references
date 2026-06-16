# py_gpu

Python/PyObjC variant of `hello-gpu` — same eight lessons, same Metal API, dispatched from
Python instead of Objective-C++. See `../README.md` for the primary (C++) version and the
concept notes; this exists as a reference for readers who'd rather stay in Python.

Each file is a complete, standalone, runnable script.

## Setup

```sh
# from the repo root
python3 -m venv venv && venv/bin/pip install -r hello-gpu/py_gpu/requirements.txt
venv/bin/python hello-gpu/py_gpu/01_device_and_buffers.py
```

Requires a Mac with Apple Silicon (or any Metal 2+ GPU) — no CUDA/NVIDIA hardware needed, and
nothing here needs a model download or a running Ollama server.
