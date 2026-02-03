# $ venv/bin/python 03_skipgram_negative_sampling.py
#
# Goal: word2vec's Skip-gram model -- learn a dense vector per word by
# training it to predict *context* words from a *center* word. Negative
# sampling turns the impossibly expensive "predict the right word out of
# the whole vocab" softmax into a cheap binary classification: does this
# (center, context) pair actually co-occur, or is it a random pairing?
# Step 3: Skip-gram with negative sampling, trained from scratch

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

corpus = [
    "the king is a strong man",
    "the queen is a strong woman",
    "the king rules the kingdom",
    "the queen rules the kingdom",
    "the prince is a young man",
    "the princess is a young woman",
    "man and woman are people",
    "king and queen are royalty",
    "the dog is a loyal animal",
    "the cat is an independent animal",
    "dog and cat are pets",
    "the man walks the dog",
    "the woman walks the cat",
]

vocab = sorted({w for line in corpus for w in line.split()})
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}
V = len(vocab)

# Build (center, context) pairs within a sliding window.
WINDOW = 2
pairs = []
for line in corpus:
    ids = [stoi[w] for w in line.split()]
    for i, center in enumerate(ids):
        for j in range(max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)):
            if i != j:
                pairs.append((center, ids[j]))

# Unigram^0.75 sampling distribution for negatives -- word2vec's trick to
# sample rare words a bit more often than raw frequency would.
freq = torch.zeros(V)
for line in corpus:
    for w in line.split():
        freq[stoi[w]] += 1
neg_dist = (freq ** 0.75)
neg_dist /= neg_dist.sum()

EMBED_DIM = 16
K_NEGATIVES = 5

center_embed = nn.Embedding(V, EMBED_DIM)
context_embed = nn.Embedding(V, EMBED_DIM)
optimizer = torch.optim.Adam(list(center_embed.parameters()) + list(context_embed.parameters()), lr=0.02)

for epoch in range(300):
    random.shuffle(pairs)
    total_loss = 0.0
    for center, context in pairs:
        center_id = torch.tensor([center])
        context_id = torch.tensor([context])
        neg_ids = torch.multinomial(neg_dist, K_NEGATIVES, replacement=True)

        v_c = center_embed(center_id)          # (1, D)
        v_o = context_embed(context_id)        # (1, D)
        v_neg = context_embed(neg_ids)          # (K, D)

        pos_score = (v_c * v_o).sum()
        neg_score = (v_c @ v_neg.T).squeeze(0)  # (K,)

        # Maximize log-sigmoid(pos) + sum log-sigmoid(-neg): the positive
        # pair should score high, sampled-random pairs should score low.
        loss = -F.logsigmoid(pos_score) - F.logsigmoid(-neg_score).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 50 == 0 or epoch == 299:
        print(f"epoch {epoch:3d}  avg loss {total_loss / len(pairs):.4f}")

embeddings = center_embed.weight.detach()


def most_similar(word: str, top_k: int = 3) -> list[tuple[str, float]]:
    target = embeddings[stoi[word]]
    sims = F.cosine_similarity(target.unsqueeze(0), embeddings)
    ranked = sorted(
        ((itos[i], sims[i].item()) for i in range(V) if i != stoi[word]),
        key=lambda x: -x[1],
    )
    return ranked[:top_k]


print()
for word in ["king", "queen", "dog"]:
    print(f"most similar to {word!r}: {most_similar(word)}")
