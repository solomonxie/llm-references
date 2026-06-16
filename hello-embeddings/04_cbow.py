# $ venv/bin/python hello-embeddings/04_cbow.py
#
# Goal: word2vec's other half. CBOW (Continuous Bag of Words) flips
# Skip-gram's prediction direction -- instead of one center word predicting
# each context word separately, all the context words together (averaged)
# predict the single center word. Fewer training examples per sentence,
# each one denser; word2vec's paper found CBOW trains faster and does
# slightly better on frequent words, Skip-gram better on rare ones.
# Step 4: CBOW -- averaged context predicts the center word

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

# Build (context_ids, center) examples -- the whole window predicts one word.
WINDOW = 2
examples = []
for line in corpus:
    ids = [stoi[w] for w in line.split()]
    for i, center in enumerate(ids):
        context = [ids[j] for j in range(max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)) if j != i]
        if context:
            examples.append((context, center))

EMBED_DIM = 16

# One shared embedding table (unlike Skip-gram's separate center/context
# tables) -- CBOW's context vectors get averaged, so there's no asymmetry
# to preserve between "predicting" and "predicted" roles here.
embed = nn.Embedding(V, EMBED_DIM)
output_layer = nn.Linear(EMBED_DIM, V)  # projects averaged context vector -> vocab logits
optimizer = torch.optim.Adam(list(embed.parameters()) + list(output_layer.parameters()), lr=0.02)

for epoch in range(300):
    random.shuffle(examples)
    total_loss = 0.0
    for context, center in examples:
        context_ids = torch.tensor(context)
        center_id = torch.tensor([center])

        context_vecs = embed(context_ids)          # (context_len, D)
        avg_vec = context_vecs.mean(dim=0, keepdim=True)  # (1, D)
        logits = output_layer(avg_vec)              # (1, V)

        loss = F.cross_entropy(logits, center_id)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 50 == 0 or epoch == 299:
        print(f"epoch {epoch:3d}  avg loss {total_loss / len(examples):.4f}")

embeddings = embed.weight.detach()


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
