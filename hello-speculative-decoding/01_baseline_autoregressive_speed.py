# $ venv/bin/python hello-speculative-decoding/01_baseline_autoregressive_speed.py
#
# Goal: the baseline every later step compares against -- one target-model
# forward pass per generated token, strictly sequential. This is the
# fundamental latency bottleneck speculative decoding attacks: a big,
# accurate model is slow per call, and normal decoding calls it once per
# token no matter what.
#
# Like hello-inference (see its README), this recomputes the full sequence
# each step rather than using `past_key_values` -- avoids a known SIGBUS on
# some Apple Silicon torch/Accelerate combinations, and keeps every forward
# call in this whole series directly comparable (same call shape throughout).
# Step 1: Sequential greedy decoding, target model only, timed

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_MODEL = "gpt2"  # 124M params -- the "slow, accurate" model throughout this series

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL)
target.eval()


@torch.no_grad()
def generate_baseline(prompt: str, max_new_tokens: int) -> tuple[str, int, float]:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    forward_calls = 0
    start = time.perf_counter()

    for _ in range(max_new_tokens):
        logits = target(input_ids).logits       # one full-sequence forward pass...
        forward_calls += 1
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)  # ...produces exactly ONE new token
        input_ids = torch.cat([input_ids, next_token], dim=1)

    elapsed = time.perf_counter() - start
    return tokenizer.decode(input_ids[0], skip_special_tokens=True), forward_calls, elapsed


prompt = "The history of computing began with"
text, calls, elapsed = generate_baseline(prompt, max_new_tokens=20)

print(f"generated: {text!r}")
print(f"\ntarget model forward calls: {calls}  (exactly 1 per generated token)")
print(f"wall time: {elapsed:.2f}s  ({elapsed / calls * 1000:.1f}ms/token)")
print("\nevery later step in this series tries to generate the SAME number of")
print("tokens with FEWER target forward calls than this 1-call-per-token baseline.")
