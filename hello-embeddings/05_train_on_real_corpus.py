# $ venv/bin/python hello-embeddings/05_train_on_real_corpus.py
#
# Goal: steps 3-4 trained on a dozen hand-written sentences -- too small for
# the statistics that make word2vec work (many words only ever appear next
# to one or two others). This scales the same Skip-gram code up to a larger,
# programmatically generated corpus with real distributional structure:
# several semantic clusters, skewed word frequencies, and words that only
# ever appear rarely -- much closer to what a real corpus looks like.
# Step 5: The same Skip-gram training, scaled to a larger, messier corpus

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

# Build a larger corpus from templates over semantic clusters, so word
# frequency is naturally skewed (common template words appear constantly,
# cluster-specific nouns appear only within their own sentences) the same
# way real text is -- unlike steps 3-4's hand-tuned dozen sentences.
clusters = {
    "royalty": ["king", "queen", "prince", "princess", "duke", "duchess"],
    "animals": ["dog", "cat", "wolf", "fox", "horse", "sparrow"],
    "weather": ["rain", "snow", "wind", "storm", "sunshine", "fog"],
    "colors": ["red", "blue", "green", "golden", "silver", "dark"],
}
templates = [
    "the {a} and the {b} met at the old hall",
    "a {a} is not the same as a {b}",
    "everyone talked about the {a} near the {b}",
    "the {a} reminded her of the {b} from before",
    "under the {a} sky stood a lone {b}",
]

random.seed(1)
lines = []
for cluster_words in clusters.values():
    for _ in range(25):
        a, b = random.sample(cluster_words, 2)
        lines.append(random.choice(templates).format(a=a, b=b))
# A few cross-cluster sentences too, like real text mixing topics.
for _ in range(10):
    a = random.choice(random.choice(list(clusters.values())))
    b = random.choice(random.choice(list(clusters.values())))
    lines.append(random.choice(templates).format(a=a, b=b))
corpus = lines

vocab = sorted({w for line in corpus for w in line.split()})
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}
V = len(vocab)
print(f"corpus: {len(corpus)} sentences, vocab: {V} words")

WINDOW = 2
pairs = []
for line in corpus:
    ids = [stoi[w] for w in line.split()]
    for i, center in enumerate(ids):
        for j in range(max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)):
            if i != j:
                pairs.append((center, ids[j]))
print(f"{len(pairs)} training pairs")

freq = torch.zeros(V)
for line in corpus:
    for w in line.split():
        freq[stoi[w]] += 1
neg_dist = (freq ** 0.75)
neg_dist /= neg_dist.sum()

EMBED_DIM = 32
K_NEGATIVES = 8
center_embed = nn.Embedding(V, EMBED_DIM)
context_embed = nn.Embedding(V, EMBED_DIM)
optimizer = torch.optim.Adam(list(center_embed.parameters()) + list(context_embed.parameters()), lr=0.01)

for epoch in range(60):
    random.shuffle(pairs)
    total_loss = 0.0
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
        total_loss += loss.item()

    if epoch % 10 == 0 or epoch == 59:
        print(f"epoch {epoch:2d}  avg loss {total_loss / len(pairs):.4f}")

embeddings = center_embed.weight.detach()


def most_similar(word: str, top_k: int = 5) -> list[tuple[str, float]]:
    target = embeddings[stoi[word]]
    sims = F.cosine_similarity(target.unsqueeze(0), embeddings)
    ranked = sorted(((itos[i], sims[i].item()) for i in range(V) if i != stoi[word]), key=lambda x: -x[1])
    return ranked[:top_k]


print()
for word in ["king", "wolf", "storm", "golden"]:
    print(f"most similar to {word!r}: {most_similar(word)}")
