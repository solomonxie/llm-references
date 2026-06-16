# $ venv/bin/python hello-speculative-decoding/04_speculative_loop_full.py
#
# Goal: wrap step 3's single round into a full generation loop -- draft K,
# verify with one target call, accept a prefix, resample on the first
# rejection (or take a free bonus token if all K were accepted, since the
# verifying forward pass already computed that next position's
# distribution too), then repeat from wherever generation actually landed.
# Step 4: The complete speculative decoding loop

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

DRAFT_MODEL, TARGET_MODEL, K = "distilgpt2", "gpt2", 4

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
draft = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL).eval()
target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL).eval()


@torch.no_grad()
def speculative_generate(prompt: str, max_new_tokens: int) -> tuple[str, dict]:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    n_generated = 0
    target_calls = 0
    total_proposed = 0
    total_accepted = 0

    while n_generated < max_new_tokens:
        k = min(K, max_new_tokens - n_generated)

        # 1. Draft k tokens, recording its distribution at each position.
        draft_tokens, draft_dists = [], []
        current = input_ids
        for _ in range(k):
            probs = F.softmax(draft(current).logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(next_token.item())
            draft_dists.append(probs[0])
            current = torch.cat([current, next_token], dim=1)
        total_proposed += k

        # 2. Verify all k in one target forward pass.
        verify_input = torch.cat([input_ids, torch.tensor([draft_tokens])], dim=1)
        target_logits = target(verify_input).logits
        target_calls += 1
        target_probs = F.softmax(target_logits, dim=-1)
        prompt_len = input_ids.shape[1]

        # 3. Accept/reject in order; stop at the first rejection.
        accepted_tokens = []
        for i, token in enumerate(draft_tokens):
            target_dist = target_probs[0, prompt_len - 1 + i, :]
            p_prob, q_prob = target_dist[token].item(), draft_dists[i][token].item()
            if torch.rand(1).item() <= min(1.0, p_prob / q_prob):
                accepted_tokens.append(token)
            else:
                residual = torch.clamp(target_dist - draft_dists[i], min=0)
                residual /= residual.sum()
                resampled = torch.multinomial(residual, num_samples=1).item()
                accepted_tokens.append(resampled)
                break
        else:
            # All k accepted -- take a free bonus token from the target's
            # own next-position distribution, already sitting in target_probs.
            bonus_dist = target_probs[0, prompt_len - 1 + k, :]
            accepted_tokens.append(torch.multinomial(bonus_dist, num_samples=1).item())

        total_accepted += sum(1 for t in accepted_tokens if t in draft_tokens)
        input_ids = torch.cat([input_ids, torch.tensor([accepted_tokens])], dim=1)
        n_generated += len(accepted_tokens)

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    stats = {"target_calls": target_calls, "total_proposed": total_proposed,
             "tokens_generated": n_generated, "acceptance_rate": total_accepted / total_proposed}
    return text, stats


prompt = "The history of computing began with"
text, stats = speculative_generate(prompt, max_new_tokens=20)

print(f"generated: {text!r}\n")
print(f"tokens generated:   {stats['tokens_generated']}")
print(f"target forward calls: {stats['target_calls']}  (vs {stats['tokens_generated']} for step 1's baseline)")
print(f"draft tokens proposed: {stats['total_proposed']}, acceptance rate: {stats['acceptance_rate']:.0%}")
print(f"\ntokens per target call: {stats['tokens_generated'] / stats['target_calls']:.2f} "
      "(step 1's baseline is always exactly 1.0)")
