# $ venv/bin/python hello-embeddings/01_one_hot_and_cooccurrence.py
#
# Goal: the starting point every embedding method improves on. A one-hot
# vector represents a word but carries zero information about meaning --
# every pair of distinct words is equally "different". A co-occurrence
# matrix (how often word A appears near word B) is the first step toward
# vectors that actually encode relationships.
# Step 1: One-hot vectors and a word co-occurrence matrix

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

# One-hot: each word is a length-V vector with a single 1.
one_hot = np.eye(V)
print(f"vocab ({V}): {vocab}")
print(f"one-hot('king')  = {one_hot[stoi['king']]}")
print(f"one-hot('queen') = {one_hot[stoi['queen']]}")

# Every pair of distinct one-hot vectors has the same dot product (0) --
# "king" and "queen" look exactly as unrelated as "king" and "mouse".
dot_king_queen = one_hot[stoi["king"]] @ one_hot[stoi["queen"]]
dot_king_mouse = one_hot[stoi["king"]] @ one_hot[stoi["mouse"]]
print(f"\ndot(king, queen) = {dot_king_queen}, dot(king, mouse) = {dot_king_mouse}")
print("identical -- one-hot vectors encode identity, not meaning.")

# Co-occurrence matrix: count how often each word appears within WINDOW
# positions of each other word, across the whole corpus.
WINDOW = 2
cooc = np.zeros((V, V))
for line in corpus:
    words = line.split()
    for i, w in enumerate(words):
        for j in range(max(0, i - WINDOW), min(len(words), i + WINDOW + 1)):
            if i != j:
                cooc[stoi[w], stoi[words[j]]] += 1

print(f"\nco-occurrence row for 'king':  {dict(zip(vocab, cooc[stoi['king']].astype(int)))}")
print(f"co-occurrence row for 'queen': {dict(zip(vocab, cooc[stoi['queen']].astype(int)))}")
print("\n'king' and 'queen' now have similar *rows* (both co-occur with 'ruled',")
print("'kingdom', 'wore', 'crown') even though their one-hot vectors don't overlap")
print("at all -- this is the raw material step 2's similarity metric uses.")
