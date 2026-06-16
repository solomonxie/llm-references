# karpathy-microgpt

Goal: Andrej Karpathy's `microgpt.py` -- a complete GPT (autograd, attention, Adam, sampling) in
~200 lines of dependency-free Python -- vendored unmodified in `original/`, then walked through
one section at a time. Where `hello-llm-training` independently rebuilds a tiny GPT with
`torch`, this is the same idea taken to the extreme: no tensors, no library, every gradient a
plain Python scalar.

Each numbered file is a complete, standalone, runnable script -- later files re-declare code
from earlier ones rather than importing across numbered files. `06_inference_sampling.py` is,
line for line, the same algorithm as `original/microgpt.py`.

## Setup

No dependencies -- standard library only.

```sh
python3 01_dataset_and_tokenizer.py
# ...
python3 06_inference_sampling.py   # ~90s on one CPU core, trains + samples 20 new names
```

## Steps

| File | Adds |
| --- | --- |
| `01_dataset_and_tokenizer.py` | Load the names dataset, char-level tokenizer + BOS token |
| `02_autograd_engine.py` | `Value`: scalar autograd (forward ops build a graph, `.backward()` fills gradients) |
| `03_model_parameters.py` | `state_dict` -- token/position embeddings + one transformer layer's weights, randomly initialized |
| `04_forward_pass.py` | `gpt()`: the architecture (RMSNorm, multi-head attention w/ KV cache, MLP), run once, untrained |
| `05_training_loop.py` | Adam optimizer + training loop: next-char prediction over the dataset, 1000 steps |
| `06_inference_sampling.py` | Temperature-sampled generation -- the full file, end to end |

## Notes

- `original/microgpt.py` is reproduced for commentary/education; see `original/README.md` for
  source, vendored commit, and license status (none declared upstream as of vendoring).
- The model is intentionally tiny (1 layer, 16-dim embeddings, ~4,200 params) trained on ~32k
  names -- the point is watching the *entire* algorithm, not building something capable.
- `keys`/`values` in `gpt()` are a running cache across positions within one document -- the
  same KV-cache idea as `hello-inference/05_kv_cache_speed.py`, at scalar-by-scalar scale.
