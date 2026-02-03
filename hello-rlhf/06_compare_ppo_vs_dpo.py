# $ venv/bin/python 06_compare_ppo_vs_dpo.py
#
# Goal: train the same starting policy with step 4's PPO and step 5's DPO,
# on the same preference data, and compare where each one ends up -- both
# in true_score (the hidden ground truth neither ever sees) and in how much
# machinery each approach needed to get there.
# Step 6: PPO vs. DPO, same policy init, same preference data, compared

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


dataset = build_preference_pairs(200)
INIT_LOGITS = torch.zeros(SEQ_LEN, V)  # identical starting point for both runs


def train_ppo() -> torch.Tensor:
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

    policy_logits = nn.Parameter(INIT_LOGITS.clone())
    ref_logits = INIT_LOGITS.clone()
    optimizer = torch.optim.Adam([policy_logits], lr=0.05)

    def batch_reward(actions):
        one_hot = F.one_hot(actions, num_classes=V).float().view(actions.shape[0], -1)
        with torch.no_grad():
            return reward_model.net(one_hot).squeeze(-1)

    for _ in range(60):
        with torch.no_grad():
            old_dist = Categorical(logits=policy_logits)
            actions = old_dist.sample((32,))
            old_log_probs = old_dist.log_prob(actions).sum(-1)
            rewards = batch_reward(actions)
            advantage = rewards - rewards.mean()

        for _ in range(4):
            new_dist = Categorical(logits=policy_logits)
            new_log_probs = new_dist.log_prob(actions).sum(-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate = torch.min(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage)
            kl = kl_divergence(new_dist, Categorical(logits=ref_logits)).sum()
            loss = -surrogate.mean() + 0.1 * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return policy_logits.detach()


def train_dpo(beta: float = 0.5) -> torch.Tensor:
    policy_logits = nn.Parameter(INIT_LOGITS.clone())
    ref_logits = INIT_LOGITS.clone()
    optimizer = torch.optim.Adam([policy_logits], lr=0.05)

    def seq_logprob(logits, seq):
        return Categorical(logits=logits).log_prob(torch.tensor(seq)).sum()

    for _ in range(150):
        random.shuffle(dataset)
        for chosen, rejected in dataset:
            pi_c, pi_r = seq_logprob(policy_logits, chosen), seq_logprob(policy_logits, rejected)
            with torch.no_grad():
                ref_c, ref_r = seq_logprob(ref_logits, chosen), seq_logprob(ref_logits, rejected)
            logits_diff = beta * ((pi_c - ref_c) - (pi_r - ref_r))
            loss = -F.logsigmoid(logits_diff)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return policy_logits.detach()


def evaluate(logits: torch.Tensor, n_samples: int = 200) -> float:
    with torch.no_grad():
        samples = Categorical(logits=logits).sample((n_samples,))
    return sum(true_score(s.tolist()) for s in samples) / n_samples


ppo_logits = train_ppo()
dpo_logits = train_dpo()

print(f"{'method':10s} {'avg true_score':16s} {'P(A) per position':s}")
for name, logits in [("PPO", ppo_logits), ("DPO", dpo_logits)]:
    avg_score = evaluate(logits)
    p_a = F.softmax(logits, dim=-1)[:, 0].numpy().round(2)
    print(f"{name:10s} {avg_score:<16.2f} {p_a}")

print(f"\nrandom-policy baseline avg true_score: {SEQ_LEN / V:.2f} (1 in 4 chance of 'A' per position)")
print("\nPPO needed: a trained reward model, sampled rollouts, importance-sampling")
print("ratios, clipping, and a KL penalty. DPO needed: one loss function over the")
print("preference pairs directly. Both should land near the same policy here.")
