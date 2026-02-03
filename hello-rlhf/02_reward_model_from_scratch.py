# $ venv/bin/python 02_reward_model_from_scratch.py
#
# Goal: turn preference pairs into a scalar reward function. The
# Bradley-Terry model says: P(chosen beats rejected) = sigmoid(r(chosen) -
# r(rejected)). Training a reward model is exactly maximizing that
# likelihood over the dataset -- no direct reward labels ever exist, only
# which-of-two-is-better comparisons, and the model has to back out a
# consistent scalar scale from those comparisons alone.
# Step 2: Training a reward model from pairwise preferences (Bradley-Terry)

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


dataset = build_preference_pairs(200)


def one_hot_flat(seq: list[int]) -> torch.Tensor:
    return F.one_hot(torch.tensor(seq), num_classes=V).float().flatten()


class RewardModel(nn.Module):
    def __init__(self, seq_len: int, vocab_size: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * vocab_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq: list[int]) -> torch.Tensor:
        return self.net(one_hot_flat(seq)).squeeze(-1)


reward_model = RewardModel(SEQ_LEN, V)
optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)

for epoch in range(200):
    random.shuffle(dataset)
    total_loss = 0.0
    for chosen, rejected in dataset:
        r_chosen = reward_model(chosen)
        r_rejected = reward_model(rejected)
        # Bradley-Terry / pairwise logistic loss: push r(chosen) - r(rejected) up.
        loss = -F.logsigmoid(r_chosen - r_rejected)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  avg loss {total_loss / len(dataset):.4f}")

# Sanity check: does the learned reward correlate with the hidden true_score,
# despite never seeing true_score directly -- only pairwise comparisons?
with torch.no_grad():
    test_seqs = [random_sequence() for _ in range(10)]
    print(f"\n{'sequence':10s} {'true score':11s} {'learned reward':15s}")
    for seq in sorted(test_seqs, key=true_score, reverse=True):
        r = reward_model(seq).item()
        print(f"{''.join(VOCAB[t] for t in seq):10s} {true_score(seq):11d} {r:15.3f}")
