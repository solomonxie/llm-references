# $ venv/bin/python hello-embeddings/02_cosine_similarity.py
#
# Goal: a proper similarity metric for vectors of any origin (co-occurrence
# rows here, learned embeddings from step 3 on). Cosine similarity measures
# the angle between two vectors, not their magnitude -- so word frequency
# (which inflates raw dot products) doesn't dominate the comparison.
# Step 2: Cosine similarity, applied to co-occurrence rows

import numpy as np

corpus = [
    "the king ruled the kingdom",
    "the queen ruled the kingdom",
    "the king wore a crown",
    "the queen wore a crown",
    "the dog chased the cat",
    "the cat chased the mouse",
]

vocab = sorted({w for line in corpus for w in line.split()})
stoi = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

WINDOW = 2
cooc = np.zeros((V, V))
for line in corpus:
    words = line.split()
    for i, w in enumerate(words):
        for j in range(max(0, i - WINDOW), min(len(words), i + WINDOW + 1)):
            if i != j:
                cooc[stoi[w], stoi[words[j]]] += 1


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(a @ b / denom)


def most_similar(word: str, top_k: int = 3) -> list[tuple[str, float]]:
    target = cooc[stoi[word]]
    sims = [(w, cosine_sim(target, cooc[stoi[w]])) for w in vocab if w != word]
    return sorted(sims, key=lambda x: -x[1])[:top_k]


for word in ["king", "queen", "dog", "the"]:
    print(f"most similar to {word!r}: {most_similar(word)}")

# Cosine similarity is invariant to scale -- doubling a vector doesn't change
# its similarity to anything, unlike a raw dot product.
king_row = cooc[stoi["king"]]
print(f"\ncosine(king, king)      = {cosine_sim(king_row, king_row):.3f}")
print(f"cosine(king, king * 5)  = {cosine_sim(king_row, king_row * 5):.3f}  (still 1.0)")
print(f"dot(king, king)         = {king_row @ king_row:.1f}")
print(f"dot(king, king * 5)     = {king_row @ (king_row * 5):.1f}  (5x larger, misleadingly)")
