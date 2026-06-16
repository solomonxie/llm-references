# hello-finetune

Goal: full fine-tuning vs. LoRA/QLoRA, on a small open model (`distilgpt2`, 82M params, CPU-
friendly). A made-up set of "facts" the base model can't know runs through every step, so
each step's success is just: did the model learn them?

Each file is a complete, standalone, runnable script -- later files re-declare code from
earlier ones rather than importing across numbered files.

## Setup

```sh
# from the repo root
python3 -m venv venv && venv/bin/pip install -r hello-finetune/requirements.txt
venv/bin/python hello-finetune/01_load_small_model.py   # first run downloads distilgpt2 (~350MB), cached after
```

## Notes

- Steps 3-4 implement LoRA from scratch (no `peft`) so the mechanism is visible; steps 6-7
  switch to the real `peft`/`bitsandbytes` libraries once that mechanism is established.
- Step 6 (QLoRA) requires a CUDA GPU -- `bitsandbytes`' 4-bit kernels have no CPU or Apple
  Silicon backend. It raises a clear error rather than running on unsupported hardware; the
  LoRA math it demonstrates is identical to steps 3-5, which run anywhere.
- Steps 6-7 write adapter/model files (`qlora_adapter/`, `lora_adapter/`, `merged_model/`) into
  this directory when run -- gitignored, not committed.
