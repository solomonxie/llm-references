# $ venv/bin/python hello-rlhf/04_ppo_from_scratch.py
#
# Goal: PPO (Proximal Policy Optimization) applied to the reward model from
# steps 2-3. The policy here is a table of per-position logits (not a full
# autoregressive network -- the point is PPO's loss mechanics, not
# architecture; see hello-transformer for the architecture side) so its
# exact KL divergence to a frozen reference policy is computable in closed
# form, no estimator needed. Core PPO idea: sample sequences, score them
# with the reward model, then update the policy toward higher-reward
# sequences -- but only by a *clipped* amount per step (so one batch of
# rollouts can't swing the policy too far), plus a KL penalty keeping the
# policy from drifting too far from where it started (which is what keeps
# it from "reward hacking" into degenerate output the reward model
# happens to score highly).
# Step 4: A minimal PPO loop, reward model + clipped objective + KL penalty

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence

torch.manual_seed(0)
random.seed(0)

VOCAB = ["A", "B", "C", "D"]
SEQ_LEN = 5
V = len(VOCAB)


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


def one_hot_flat(seq):
    return F.one_hot(torch.tensor(seq), num_classes=V).float().flatten()


class RewardModel(nn.Module):
    def __init__(self, seq_len, vocab_size, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(seq_len * vocab_size, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, seq):
        return self.net(one_hot_flat(seq)).squeeze(-1)


# --- reward model, trained exactly as in step 2 -------------------------
dataset = build_preference_pairs(200)
reward_model = RewardModel(SEQ_LEN, V)
reward_opt = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
for _ in range(200):
    random.shuffle(dataset)
    for chosen, rejected in dataset:
        loss = -F.logsigmoid(reward_model(chosen) - reward_model(rejected))
        reward_opt.zero_grad()
        loss.backward()
        reward_opt.step()
for p in reward_model.parameters():
    p.requires_grad = False

# --- PPO ------------------------------------------------------------------
policy_logits = nn.Parameter(torch.zeros(SEQ_LEN, V))       # starts uniform over all 4 symbols
ref_logits = policy_logits.detach().clone()                  # frozen snapshot -- the KL anchor
policy_optimizer = torch.optim.Adam([policy_logits], lr=0.05)

BATCH_SIZE = 32
CLIP_EPS = 0.2
KL_COEF = 0.1
ROLLOUTS = 60
INNER_EPOCHS = 4


def batch_reward(actions: torch.Tensor) -> torch.Tensor:
    one_hot = F.one_hot(actions, num_classes=V).float().view(actions.shape[0], -1)
    with torch.no_grad():
        return reward_model.net(one_hot).squeeze(-1)


for rollout in range(ROLLOUTS):
    # 1. Roll out a batch under the CURRENT policy, snapshotting log-probs
    #    ("old" for this update) and rewards -- these stay fixed through
    #    all inner_epochs below even as policy_logits keeps moving.
    with torch.no_grad():
        old_dist = Categorical(logits=policy_logits)
        actions = old_dist.sample((BATCH_SIZE,))            # (B, L)
        old_log_probs = old_dist.log_prob(actions).sum(-1)   # (B,)
        rewards = batch_reward(actions)
        advantage = rewards - rewards.mean()                 # simplest possible baseline

    # 2. Several gradient steps on the SAME rollout batch -- this is what
    #    "proximal" protects against: without clipping, repeated updates on
    #    stale data could push the policy arbitrarily far from what was
    #    actually sampled.
    for _ in range(INNER_EPOCHS):
        new_dist = Categorical(logits=policy_logits)
        new_log_probs = new_dist.log_prob(actions).sum(-1)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate1 = ratio * advantage
        surrogate2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantage
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        ref_dist = Categorical(logits=ref_logits)
        kl = kl_divergence(new_dist, ref_dist).sum()          # exact, since both are simple tables

        loss = policy_loss + KL_COEF * kl

        policy_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()

    if rollout % 10 == 0 or rollout == ROLLOUTS - 1:
        avg_true_score = sum(true_score(a.tolist()) for a in actions) / BATCH_SIZE
        print(f"rollout {rollout:2d}  avg reward {rewards.mean():.3f}  "
              f"avg true_score {avg_true_score:.2f}/{SEQ_LEN}  KL {kl.item():.3f}")

with torch.no_grad():
    final_dist = Categorical(logits=policy_logits)
    samples = final_dist.sample((10,))
print("\nfinal policy samples:")
for s in samples:
    print(f"  {''.join(VOCAB[t] for t in s.tolist())}  (true_score={true_score(s.tolist())})")
print(f"\nlearned per-position P(A): {F.softmax(policy_logits, dim=-1)[:, 0].detach().numpy().round(2)}")
print("(started uniform at 0.25 each -- PPO pushed it toward 'A', the symbol the")
print("reward model learned to prefer, without ever seeing true_score directly)")
