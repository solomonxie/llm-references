# $ venv/bin/python 03_draft_then_verify_single_step.py
#
# Goal: one full round of speculative decoding's core trick. The draft
# model proposes K tokens autoregressively (K slow-ish small forward
# calls). Then -- the actual speedup -- the TARGET model checks all K of
# them in a single forward pass over the whole draft continuation at once
# (a transformer computes every position's logits in parallel; it doesn't
# need K sequential calls to score K positions, only to GENERATE them).
# Each proposed token is then accepted or rejected by comparing the two
# models' probabilities for it -- accepting with probability
# min(1, p_target/p_draft) keeps the final output an exact sample from the
# target's own distribution, never worse than just running the target alone.
# Step 3: Draft K tokens, verify with one target forward pass, accept/reject

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

DRAFT_MODEL, TARGET_MODEL, K = "distilgpt2", "gpt2", 4

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
draft = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL).eval()
target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL).eval()

prompt = "The history of computing began with"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

# --- 1. Draft proposes K tokens, recording its FULL distribution at each
#        position (needed for the residual resample below, not just the
#        probability of the one token it happened to sample) -----------
draft_tokens, draft_dists = [], []
current = input_ids
with torch.no_grad():
    for _ in range(K):
        logits = draft(current).logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        draft_tokens.append(next_token.item())
        draft_dists.append(probs[0])
        current = torch.cat([current, next_token], dim=1)

print(f"draft proposed: {[tokenizer.decode([t]) for t in draft_tokens]}")

# --- 2. Target verifies all K at once -- ONE forward pass, not K --------
verify_input = torch.cat([input_ids, torch.tensor([draft_tokens])], dim=1)
with torch.no_grad():
    target_logits = target(verify_input).logits
# Position (len(prompt) - 1 + i) is where the target predicts draft_tokens[i],
# exactly like a normal next-token prediction one step earlier in the sequence.
prompt_len = input_ids.shape[1]
target_probs_all = F.softmax(target_logits, dim=-1)

# --- 3. Walk the K proposed tokens in order, accept/reject each ---------
accepted = []
resampled_token = None
for i, token in enumerate(draft_tokens):
    target_dist = target_probs_all[0, prompt_len - 1 + i, :]
    q_prob = draft_dists[i][token].item()
    p_prob = target_dist[token].item()
    accept_ratio = min(1.0, p_prob / q_prob)
    u = torch.rand(1).item()
    decision = "ACCEPT" if u <= accept_ratio else "REJECT"
    print(f"  token {i}: {tokenizer.decode([token])!r}  "
          f"p_target={p_prob:.4f}  q_draft={q_prob:.4f}  ratio={accept_ratio:.3f}  u={u:.3f}  -> {decision}")
    if decision == "ACCEPT":
        accepted.append(token)
    else:
        # Resample from the residual distribution: what the target believes
        # MINUS what the draft already accounted for, renormalized -- this
        # is what keeps the overall output distributed exactly as the
        # target alone would have produced it, despite drafting from a
        # weaker model.
        residual = torch.clamp(target_dist - draft_dists[i], min=0)
        residual /= residual.sum()
        resampled_token = torch.multinomial(residual, num_samples=1).item()
        print(f"    resampled from residual distribution: {tokenizer.decode([resampled_token])!r}")
        break

print(f"\naccepted {len(accepted)}/{K} draft tokens this round: "
      f"{[tokenizer.decode([t]) for t in accepted]}")
if resampled_token is not None:
    print(f"plus one resampled token: {tokenizer.decode([resampled_token])!r}")
else:
    print("all K accepted -- a bonus token can be sampled for free from the target's")
    print("own next-position distribution (already computed in this same forward pass)")
