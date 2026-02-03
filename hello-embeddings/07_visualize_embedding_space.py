# $ venv/bin/python 07_visualize_embedding_space.py
#
# Goal: embeddings live in a 32-dimensional space (step 5) -- impossible to
# look at directly. PCA projects that space down to 2D while preserving as
# much variance as possible, so semantic clusters (if the training actually
# learned them) become visible as literal clusters on a scatter plot.
# Step 7: PCA projection of trained embeddings to 2D, plotted and labeled

import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

torch.manual_seed(0)
random.seed(1)

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

corpus = []
for cluster_words in clusters.values():
    for _ in range(25):
        a, b = random.sample(cluster_words, 2)
        corpus.append(random.choice(templates).format(a=a, b=b))
for _ in range(10):
    a = random.choice(random.choice(list(clusters.values())))
    b = random.choice(random.choice(list(clusters.values())))
    corpus.append(random.choice(templates).format(a=a, b=b))

vocab = sorted({w for line in corpus for w in line.split()})
stoi = {w: i for i, w in enumerate(vocab)}
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

EMBED_DIM = 32
K_NEGATIVES = 8
center_embed = nn.Embedding(V, EMBED_DIM)
context_embed = nn.Embedding(V, EMBED_DIM)
optimizer = torch.optim.Adam(list(center_embed.parameters()) + list(context_embed.parameters()), lr=0.01)

for epoch in range(60):
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

embeddings = center_embed.weight.detach().numpy()

# Project 32D -> 2D. PCA finds the 2 directions of greatest variance in the
# embedding space -- not necessarily "meaning", but often correlated with it
# when the space has learned real structure.
coords = PCA(n_components=2).fit_transform(embeddings)

word_to_cluster = {w: cname for cname, words in clusters.items() for w in words}
colors = {"royalty": "tab:purple", "animals": "tab:green", "weather": "tab:blue", "colors": "tab:orange"}

plt.figure(figsize=(9, 7))
for i, word in enumerate(vocab):
    cluster = word_to_cluster.get(word, "other")
    color = colors.get(cluster, "gray")
    plt.scatter(*coords[i], color=color, s=40)
    plt.annotate(word, coords[i], fontsize=8, alpha=0.8, xytext=(3, 3), textcoords="offset points")

handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=name, markersize=8)
           for name, c in colors.items()]
plt.legend(handles=handles, title="cluster (ground truth)")
plt.title("Skip-gram embeddings, PCA-projected to 2D")
plt.tight_layout()
plt.savefig("embedding_space.png", dpi=150)
print("saved embedding_space.png")
print("\nwords from the same generating cluster should sit closer together than")
print("words from different clusters -- how tight that grouping looks is a rough,")
print("visual proxy for how well the co-occurrence signal was actually learned.")
