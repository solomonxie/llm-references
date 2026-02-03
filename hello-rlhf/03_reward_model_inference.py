# $ venv/bin/python 03_reward_model_inference.py
#
# Goal: what a trained reward model is actually used for -- scoring a batch
# of candidate completions (e.g. several samples from a policy) and picking
# the best one, "best-of-N" sampling, one of the simplest ways a reward
# model improves generation without any RL training at all.
# Step 3: Using a trained reward model to rank/select among candidates

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

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
reward_model = RewardModel(SEQ_LEN, V)
optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
for _ in range(200):
    random.shuffle(dataset)
    for chosen, rejected in dataset:
        loss = -F.logsigmoid(reward_model(chosen) - reward_model(rejected))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- inference: best-of-N selection ------------------------------------
# Simulate N samples from some upstream policy (here, just random sequences
# standing in for "whatever a language model generated") and use the reward
# model to pick the one it scores highest.
N_CANDIDATES = 8
candidates = [random_sequence() for _ in range(N_CANDIDATES)]

with torch.no_grad():
    scored = [(seq, reward_model(seq).item()) for seq in candidates]

scored.sort(key=lambda x: -x[1])
print(f"{'candidate':10s} {'reward':8s} {'true score':10s}")
for seq, reward in scored:
    marker = " <- picked" if seq == scored[0][0] else ""
    print(f"{''.join(VOCAB[t] for t in seq):10s} {reward:8.3f} {true_score(seq):10d}{marker}")

best_by_reward = scored[0][0]
best_by_truth = max(candidates, key=true_score)
print(f"\nreward model's top pick has true_score={true_score(best_by_reward)}; "
      f"the actual best candidate has true_score={true_score(best_by_truth)}")
