# $ venv/bin/python hello-rlhf/05_dpo_from_scratch.py
#
# Goal: DPO (Direct Preference Optimization) -- reaches the same place as
# PPO (step 4) without a separate reward model or an RL rollout loop at
# all. The trick: the Bradley-Terry reward-model objective (step 2) and an
# RL policy's optimal solution under a KL constraint to a reference policy
# turn out to have a closed form relating them -- substituting it back in
# gives a loss computable directly from (chosen, rejected, policy,
# reference policy) log-probabilities. One loss function, one optimizer,
# no rollouts, no reward model, no clipping.
# Step 5: Direct Preference Optimization -- one loss, straight from pairs

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

torch.manual_seed(0)
random.seed(0)

VOCAB = ["A", "B", "C", "D"]
SEQ_LEN = 5
V = len(VOCAB)
BETA = 0.5  # DPO's temperature -- how sharply it trusts the preference signal


def random_sequence():
    return [random.randrange(V) for _ in range(SEQ_LEN)]


def true_score(seq):
    return sum(1 for tok in seq if tok == 0)


def build_preference_pairs(n_pairs):
    pairs = []
    while len(pairs) < n_pairs:
        a, b = random_sequence(), random_sequence()
        sa, sb = true_score(a), true_score(b)
        if sa == sb:
            continue
        chosen, rejected = (a, b) if sa > sb else (b, a)
        pairs.append((chosen, rejected))
    return pairs


dataset = build_preference_pairs(200)

policy_logits = nn.Parameter(torch.zeros(SEQ_LEN, V))
ref_logits = policy_logits.detach().clone()   # frozen reference, same role as PPO's KL anchor
optimizer = torch.optim.Adam([policy_logits], lr=0.05)


def seq_logprob(logits: torch.Tensor, seq: list[int]) -> torch.Tensor:
    dist = Categorical(logits=logits)
    return dist.log_prob(torch.tensor(seq)).sum()


for epoch in range(150):
    random.shuffle(dataset)
    total_loss = 0.0
    for chosen, rejected in dataset:
        pi_chosen = seq_logprob(policy_logits, chosen)
        pi_rejected = seq_logprob(policy_logits, rejected)
        with torch.no_grad():
            ref_chosen = seq_logprob(ref_logits, chosen)
            ref_rejected = seq_logprob(ref_logits, rejected)

        # The DPO loss: push the policy's chosen-vs-rejected log-ratio
        # (relative to the reference) up, exactly like Bradley-Terry (step 2)
        # pushed reward(chosen) - reward(rejected) up -- the policy's own
        # log-probabilities are standing in for the reward model here.
        logits_diff = BETA * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
        loss = -F.logsigmoid(logits_diff)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 30 == 0 or epoch == 149:
        print(f"epoch {epoch:3d}  avg loss {total_loss / len(dataset):.4f}")

with torch.no_grad():
    final_dist = Categorical(logits=policy_logits)
    samples = final_dist.sample((10,))
print("\nfinal policy samples:")
for s in samples:
    print(f"  {''.join(VOCAB[t] for t in s.tolist())}  (true_score={true_score(s.tolist())})")
print(f"\nlearned per-position P(A): {F.softmax(policy_logits, dim=-1)[:, 0].detach().numpy().round(2)}")
print("(same end result as step 4's PPO run -- reached with a single supervised")
print("loss instead of a reward model plus a sampling/clipping/KL RL loop)")
