# $ venv/bin/python 06_batching.py
#
# Goal: generate for several prompts in one forward pass instead of one
# prompt at a time. A GPU (and, to a lesser extent, a CPU's matmul kernels)
# is underused running one small sequence at a time — stacking N prompts
# into a (N, seq_len) batch does N sequences' worth of work per forward call
# at close to the cost of one, instead of N separate calls.
#
# The wrinkle: prompts are different lengths, but a batch needs one fixed
# tensor shape. Padding solves the shape mismatch; LEFT-padding (pad tokens
# BEFORE the real tokens, not after) is what makes `logits[:, -1, :]` still
# mean "this sequence's true next-token prediction" for every row after
# padding — with right-padding, the last column would be a pad position for
# any shorter sequence instead of its actual last real token.
# Step 6: Left-padding + attention_mask + position_ids to decode several prompts in one batch

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 ships with no pad token — reuse eos, a common convention
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

prompts = [
    "The capital of France is",
    "In machine learning, a neural network",
    "def fibonacci(n):",
    "The best way to learn a new programming language is",
]


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    # Left-padding shifts every real token rightward by however many pad
    # tokens precede it, but position 0 should still mean "this sequence's
    # first real token" — cumsum(attention_mask) - 1 recovers that per row,
    # independent of how much padding precedes it. Padded positions get
    # clamped to 0 (masked_fill below) since their position value is never
    # actually used (attention_mask excludes them from attention anyway).
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


@torch.no_grad()
def batched_greedy_decode(prompts: list[str], max_new_tokens: int) -> list[str]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    generated, mask = encoded.input_ids, encoded.attention_mask

    for _ in range(max_new_tokens):
        position_ids = position_ids_from_mask(mask)
        logits = model(generated, attention_mask=mask, position_ids=position_ids).logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)  # (batch, 1) — one next-token choice per row
        generated = torch.cat([generated, next_token], dim=1)
        mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)  # the new token is always real, never padding

    return [tokenizer.decode(row, skip_special_tokens=True) for row in generated]


@torch.no_grad()
def sequential_greedy_decode(prompt: str, max_new_tokens: int) -> str:
    generated = tokenizer(prompt, return_tensors="pt").input_ids
    for _ in range(max_new_tokens):
        next_token = model(generated).logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return tokenizer.decode(generated[0])


MAX_NEW_TOKENS = 15

start = time.perf_counter()
batched_results = batched_greedy_decode(prompts, MAX_NEW_TOKENS)
batched_time = time.perf_counter() - start

start = time.perf_counter()
sequential_results = [sequential_greedy_decode(p, MAX_NEW_TOKENS) for p in prompts]
sequential_time = time.perf_counter() - start

print("batched results:")
for r in batched_results:
    print(f"  {r!r}")

print(f"\nbatched:    {batched_time:.3f}s for {len(prompts)} prompts")
print(f"sequential: {sequential_time:.3f}s for {len(prompts)} prompts")
print(f"speedup: {sequential_time / batched_time:.2f}x")

# Left-padding did its job if these match: the tail end of each batched
# result (after the padding-shifted prefix) should read the same as running
# that same prompt alone.
print("\nsanity check — batched vs. sequential should produce the SAME tokens per prompt:")
for prompt, batched, sequential in zip(prompts, batched_results, sequential_results):
    match = "✓" if batched.strip() == sequential.strip() else "✗"
    print(f"  {match} {prompt!r}")
