# hello-inference-server

Goal: what a real LLM inference server does beyond "call `model.generate()`" -- streaming,
KV caching across requests, batching concurrent requests together, and a proper continuous-
batching scheduler, ending with a load test measuring throughput and latency percentiles.
Tokens are small integers over a toy 20-symbol vocabulary (untrained weights) -- the point is
serving mechanics, not language quality; see `hello-tokenizer`/`hello-inference` for that side.

Each file starts its own FastAPI server in a background thread on its own port, then acts as
its own client against it, so `python3 NN_file.py` is a complete, standalone demo -- no
separate server process or terminal needed.

## Setup

```sh
# from the repo root
python3 -m venv venv && venv/bin/pip install -r hello-inference-server/requirements.txt
venv/bin/python hello-inference-server/01_minimal_http_generate.py
```

## Notes

- Steps 4-6's batching assumes equal-length prompts and does no padding attention-masking --
  `hello-inference/06_batching.py` covers real padding + `attention_mask` handling in isolation.
- Step 3 reuses the same explicit KV-cache pattern as `hello-inference/05_kv_cache_speed.py`,
  now kept alive across separate HTTP requests instead of separate function calls.
