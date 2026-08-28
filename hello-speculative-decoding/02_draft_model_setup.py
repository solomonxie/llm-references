# $ venv/bin/python 02_draft_model_setup.py
#
# Goal: speculative decoding needs two models sharing the same tokenizer/
# vocabulary -- a small, cheap "draft" model that proposes tokens fast but
# less accurately, and the big "target" model from step 1 that's slow but
# is the one whose distribution the final output actually has to match.
# distilgpt2 (half of GPT-2's 12 layers) and gpt2 share GPT-2's exact
# byte-level BPE tokenizer, making them a valid draft/target pair.
# Step 2: Loading draft + target models, confirming they share a vocabulary

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT_MODEL = "distilgpt2"   # 82M params, ~half the layers of gpt2
TARGET_MODEL = "gpt2"        # 124M params

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)  # same tokenizer serves both models
draft = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL).eval()
target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL).eval()

# Vocab compatibility is not optional -- if it didn't match, a token id the
# draft proposes could mean something completely different to the target.
draft_vocab = draft.config.vocab_size
target_vocab = target.config.vocab_size
print(f"draft vocab size:  {draft_vocab}")
print(f"target vocab size: {target_vocab}")
print(f"shared vocabulary: {draft_vocab == target_vocab}")

draft_params = sum(p.numel() for p in draft.parameters())
target_params = sum(p.numel() for p in target.parameters())
print(f"\ndraft params:  {draft_params:,}")
print(f"target params: {target_params:,}  ({target_params / draft_params:.1f}x the draft)")


@torch.no_grad()
def time_forward(model, input_ids: torch.Tensor, n_calls: int = 20) -> float:
    start = time.perf_counter()
    for _ in range(n_calls):
        model(input_ids)
    return (time.perf_counter() - start) / n_calls


prompt = "The history of computing began with"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

draft_latency = time_forward(draft, input_ids)
target_latency = time_forward(target, input_ids)
print(f"\nper-call latency -- draft: {draft_latency * 1000:.1f}ms, target: {target_latency * 1000:.1f}ms "
      f"({target_latency / draft_latency:.1f}x slower)")

# Both models greedy-decode independently here, just to see how their
# continuations of the same prompt differ -- step 3 starts combining them.
with torch.no_grad():
    draft_out = draft.generate(input_ids, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    target_out = target.generate(input_ids, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(f"\ndraft alone:  {tokenizer.decode(draft_out[0], skip_special_tokens=True)!r}")
print(f"target alone: {tokenizer.decode(target_out[0], skip_special_tokens=True)!r}")
