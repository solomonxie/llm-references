# hello-speculative-decoding

Goal: speculative decoding -- using a small, cheap "draft" model to propose several tokens
at once, then checking all of them in a single forward pass of the big "target" model,
accepting whatever's consistent with the target's own distribution and resampling from
there. Draft is `distilgpt2`, target is `gpt2` -- they share GPT-2's tokenizer/vocab, a hard
requirement for this to work at all.

Each file is a complete, standalone, runnable script -- later files re-declare code from
earlier ones rather than importing across numbered files.

## Setup

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_baseline_autoregressive_speed.py   # downloads gpt2 (~500MB) + distilgpt2 (~350MB)
```

## Notes

- Every forward call recomputes the full sequence rather than using `past_key_values` --
  same reasoning as `hello-inference`'s README (a known SIGBUS on some Apple Silicon
  torch/Accelerate combinations), and it keeps every call's shape directly comparable
  across steps.
- Speculative decoding's target-call reduction is exact and easy to measure; the wall-clock
  speedup is smaller and hardware-dependent -- step 5 measures both and explains the gap.
