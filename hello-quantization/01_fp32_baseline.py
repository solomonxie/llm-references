# $ venv/bin/python 01_fp32_baseline.py
#
# Goal: the baseline every later step compares against -- a small model
# (distilgpt2) in its native fp32 precision. Every weight is a 32-bit
# float; every later step trades some of that precision away for less
# memory and (on suitable hardware) more speed, and this series keeps
# asking: how much quality is actually lost for how much saved?
# Step 1: fp32 baseline -- memory footprint, latency, generation quality

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()

param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"dtype: {next(model.parameters()).dtype}")
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"weight memory: {param_bytes / 1e6:.1f} MB")

prompt = "The history of computing began with"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - start

text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\ngenerated ({elapsed:.2f}s): {text!r}")

# Perplexity on a fixed held-out sentence -- the quality metric every later
# step's quantized model gets compared against.
eval_text = "The quick brown fox jumps over the lazy dog and runs into the forest."
eval_inputs = tokenizer(eval_text, return_tensors="pt")
with torch.no_grad():
    loss = model(**eval_inputs, labels=eval_inputs["input_ids"]).loss
perplexity = torch.exp(loss).item()
print(f"perplexity on eval sentence: {perplexity:.2f}")
