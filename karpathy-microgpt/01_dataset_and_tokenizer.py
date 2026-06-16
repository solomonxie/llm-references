# $ python3 01_dataset_and_tokenizer.py
#
# Goal: walk Andrej Karpathy's microgpt.py (verbatim copy in ../original/,
# github.com/karpathy/8627fe009c40f57531cb18360106ce95) one section at a
# time. This step is his first two blocks, unmodified: how the training
# data loads, and how text becomes token ids. The tokenizer is the
# simplest possible one -- no BPE, no library, one integer per unique
# character in the dataset, plus one reserved id (BOS) that marks both the
# start and the end of a generated sequence.
# Step 1: dataset (32k names) + char-level tokenizer

import os                          # os.path.exists
import random                      # random.seed, random.shuffle
random.seed(42)                    # Let there be order among chaos

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs)))  # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars)                    # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1         # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# --- everything below is this step's own demo, not part of the original file ---
sample = docs[0]
tokens = [BOS] + [uchars.index(ch) for ch in sample] + [BOS]
print(f"\nsample doc: {sample!r}")
print(f"tokens:     {tokens}  (BOS={BOS} at both ends)")
print(f"decoded:    {''.join(uchars[t] for t in tokens if t != BOS)!r}")
