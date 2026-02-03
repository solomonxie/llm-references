# hello-quantization

Goal: post-training quantization on a small open model (`distilgpt2`) -- fp16/bf16 casting,
int8 and int4 quantization implemented from scratch (with real bit-packing for int4),
activation calibration, and a real `bitsandbytes` comparison -- measuring the actual
memory/latency/quality tradeoff at each step rather than assuming one.

Each file is a complete, standalone, runnable script -- later files re-declare code from
earlier ones rather than importing across numbered files.

## Setup

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_fp32_baseline.py   # first run downloads distilgpt2 (~350MB), cached after
```

## Notes

- Steps 3-6 use "fake quantization" (round-trip through the narrower format, but stored back
  as float32) to isolate rounding error without needing a real int8/int4 compute backend --
  step 8 explains why its memory numbers look the same as fp32 as a result.
- Step 7 (`bitsandbytes`) requires a CUDA GPU -- its int8/4-bit kernels have no CPU or Apple
  Silicon backend. It raises a clear error rather than running on unsupported hardware.
