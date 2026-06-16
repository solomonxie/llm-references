# $ venv/bin/python hello-embeddings/06_analogy_arithmetic.py
#
# Goal: word2vec's famous party trick -- "king - man + woman ~= queen".
# If the embedding space captures relationships as consistent directions
# (here, roughly "royalty" and "gender"), then vector arithmetic on word
# vectors should land near the word that completes the analogy.
# Step 6: Vector arithmetic on trained embeddings (a - b + c ~= d)

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

WINDOW = 2
pairs = []
for line in corpus:
    ids = [stoi[w] for w in line.split()]
    for i, center in enumerate(ids):
        for j in range(max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)):
            if i != j:
                pairs.append((center, ids[j]))

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
    for center, context in pairs:
        center_id, context_id = torch.tensor([center]), torch.tensor([context])
        neg_ids = torch.multinomial(neg_dist, K_NEGATIVES, replacement=True)
        v_c, v_o, v_neg = center_embed(center_id), context_embed(context_id), context_embed(neg_ids)
        pos_score = (v_c * v_o).sum()
        neg_score = (v_c @ v_neg.T).squeeze(0)
        loss = -F.logsigmoid(pos_score) - F.logsigmoid(-neg_score).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

embeddings = center_embed.weight.detach()


def nearest(vec: torch.Tensor, exclude: set[str], top_k: int = 3) -> list[tuple[str, float]]:
    sims = F.cosine_similarity(vec.unsqueeze(0), embeddings)
    ranked = sorted(
        ((itos[i], sims[i].item()) for i in range(V) if itos[i] not in exclude),
        key=lambda x: -x[1],
    )
    return ranked[:top_k]


def analogy(a: str, b: str, c: str, top_k: int = 3) -> list[tuple[str, float]]:
    # a is to b as c is to ? -- solve for the vector a - b + c.
    vec = embeddings[stoi[a]] - embeddings[stoi[b]] + embeddings[stoi[c]]
    return nearest(vec, exclude={a, b, c}, top_k=top_k)


print("king - man + woman ~= ?")
for word, score in analogy("king", "man", "woman"):
    print(f"  {word:10s} {score:.3f}")

print("\nqueen - woman + man ~= ?")
for word, score in analogy("queen", "woman", "man"):
    print(f"  {word:10s} {score:.3f}")

print("\ndog - cat + queen ~= ? (a nonsense analogy, for contrast)")
for word, score in analogy("dog", "cat", "queen"):
    print(f"  {word:10s} {score:.3f}")

print("\nNote: with a dozen training sentences the space is small and noisy --")
print("this works far more reliably on real word2vec vectors trained on billions")
print("of words. The mechanism (vector offsets encode relationships) is identical.")
