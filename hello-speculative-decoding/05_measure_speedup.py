# $ venv/bin/python 05_measure_speedup.py
#
# Goal: put step 1's baseline and step 4's speculative loop head to head,
# generating the same number of tokens for the same prompt, and measure
# the actual wall-clock speedup -- not just the "fewer target calls"
# theory, since the draft model's own calls aren't free either.
# Step 5: Baseline vs. speculative decoding, wall-clock speedup measured

import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

DRAFT_MODEL, TARGET_MODEL, K = "distilgpt2", "gpt2", 4

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
draft = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL).eval()
target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL).eval()


@torch.no_grad()
def generate_baseline(prompt: str, max_new_tokens: int) -> tuple[str, float]:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        next_token = target(input_ids).logits[:, -1, :].argmax(-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    elapsed = time.perf_counter() - start
    return tokenizer.decode(input_ids[0], skip_special_tokens=True), elapsed


@torch.no_grad()
def speculative_generate(prompt: str, max_new_tokens: int) -> tuple[str, float, dict]:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    n_generated, target_calls, total_proposed, total_accepted = 0, 0, 0, 0
    start = time.perf_counter()

    while n_generated < max_new_tokens:
        k = min(K, max_new_tokens - n_generated)
        draft_tokens, draft_dists, current = [], [], input_ids
        for _ in range(k):
            probs = F.softmax(draft(current).logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(next_token.item())
            draft_dists.append(probs[0])
            current = torch.cat([current, next_token], dim=1)
        total_proposed += k

        verify_input = torch.cat([input_ids, torch.tensor([draft_tokens])], dim=1)
        target_probs = F.softmax(target(verify_input).logits, dim=-1)
        target_calls += 1
        prompt_len = input_ids.shape[1]

        accepted_tokens = []
        for i, token in enumerate(draft_tokens):
            target_dist = target_probs[0, prompt_len - 1 + i, :]
            p_prob, q_prob = target_dist[token].item(), draft_dists[i][token].item()
            if torch.rand(1).item() <= min(1.0, p_prob / q_prob):
                accepted_tokens.append(token)
                total_accepted += 1
            else:
                residual = torch.clamp(target_dist - draft_dists[i], min=0)
                residual /= residual.sum()
                accepted_tokens.append(torch.multinomial(residual, num_samples=1).item())
                break
        else:
            bonus_dist = target_probs[0, prompt_len - 1 + k, :]
            accepted_tokens.append(torch.multinomial(bonus_dist, num_samples=1).item())

        input_ids = torch.cat([input_ids, torch.tensor([accepted_tokens])], dim=1)
        n_generated += len(accepted_tokens)

    elapsed = time.perf_counter() - start
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return text, elapsed, {"target_calls": target_calls, "acceptance_rate": total_accepted / total_proposed}


prompt = "The history of computing began with"
MAX_NEW_TOKENS = 30

baseline_text, baseline_time = generate_baseline(prompt, MAX_NEW_TOKENS)
spec_text, spec_time, spec_stats = speculative_generate(prompt, MAX_NEW_TOKENS)

print(f"baseline:     {baseline_time:.2f}s  ({MAX_NEW_TOKENS} target calls)")
print(f"speculative:  {spec_time:.2f}s  ({spec_stats['target_calls']} target calls, "
      f"{spec_stats['acceptance_rate']:.0%} draft acceptance rate)")
print(f"\nspeedup: {baseline_time / spec_time:.2f}x")
print(f"target-call reduction: {MAX_NEW_TOKENS / spec_stats['target_calls']:.2f}x")
print("\nwall-clock speedup is smaller than the target-call reduction alone would")
print("suggest -- the draft model's own forward calls, and this CPU-bound demo's")
print("small batch sizes, both eat into the theoretical gain. Speculative decoding's")
print("real-world payoff grows with a slower target / faster draft / higher")
print("acceptance rate, and is most pronounced on GPU where each forward call's")
print("fixed overhead (not raw FLOPs) dominates at these small batch sizes.")

print(f"\nbaseline text:    {baseline_text!r}")
print(f"speculative text: {spec_text!r}")
