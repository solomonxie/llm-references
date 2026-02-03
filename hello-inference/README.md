# hello-inference

Goal: what happens between "here's a trained model" and "here's generated text" — decoding
strategies and inference-time mechanics, on a real pretrained model (GPT-2, 124M params, runs on
CPU) instead of a toy. Follows naturally from `hello-transformer` (the architecture) into how that
architecture is actually *used* to generate.

Nearly everything here is implemented by hand (a manual `model(...)` forward-pass loop) rather than
via `model.generate()` — partly pedagogy (steps 01-transformer's spirit: don't let the mechanism
hide behind a library flag), partly practical: `transformers`' KV-cache path
(`past_key_values`/`DynamicCache`) crashes with a SIGBUS on at least one Apple Silicon
torch/Accelerate combination tested during development, independent of whether it's reached via
`.generate()` or a direct `model()` call. Every file below sidesteps it — either by never touching
that code path (01-04, 06-07 all recompute the full sequence each step) or, where the whole point
IS caching (05), by using a small hand-built model instead of GPT-2's internals.

| File | Demonstrates |
|---|---|
| `01_load_and_greedy_decode.py` | Loading GPT-2; a by-hand greedy decode loop (`argmax`, repeat) |
| `02_temperature_sampling.py` | Scaling logits by temperature before sampling |
| `03_top_k_top_p.py` | Top-k and nucleus (top-p) filtering before sampling |
| `04_repetition_penalty.py` | Fixing greedy's repetition-loop failure mode |
| `05_kv_cache_speed.py` | Why/how KV-caching speeds up decoding — hand-built model + cache, benchmarked |
| `06_batching.py` | Left-padding + `attention_mask` + `position_ids` to decode several prompts in one batch |
| `07_beam_search.py` | Keeping top-k candidate sequences by cumulative log-probability, not just the single best token |

## Setup

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_load_and_greedy_decode.py   # first run downloads gpt2 (~500MB), cached after
```

## Notes

- `05` uses a small random-weight hand-built decoder, not GPT-2 — see the crash note above.
  Un-trained weights are fine there; the point is the cache mechanism's *speed*, not output quality.
- If a future `transformers`/`torch`/macOS combination fixes the underlying crash, `05` could be
  redone against real GPT-2 `past_key_values` for a more end-to-end demo — worth revisiting.
