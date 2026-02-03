# $ venv/bin/python 03_byte_pair_encoding.py
#
# Goal: Byte-Pair Encoding (BPE) — start from characters (like step 2), then
# repeatedly merge the most frequent adjacent pair of symbols into a new
# single symbol. This is how real subword vocabularies (GPT-2, GPT-4, etc.)
# are built: common words end up as one token, rare words fall back to
# smaller pieces instead of <unk>.
# Step 3: Training a BPE merge list from scratch on a toy corpus

from collections import Counter

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barks at the fox",
    "the fox and the dog are friends",
]

END = "</w>"  # marks a word boundary so merges never cross words


def word_to_symbols(word: str) -> list[str]:
    return list(word) + [END]


# Start: every word is a list of its characters (+ end marker).
word_freqs = Counter(w for line in corpus for w in line.split())
splits = {word: word_to_symbols(word) for word in word_freqs}


def get_pair_counts(splits: dict[str, list[str]]) -> Counter:
    counts = Counter()
    for word, symbols in splits.items():
        freq = word_freqs[word]
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += freq
    return counts


def merge_pair(pair: tuple[str, str], splits: dict[str, list[str]]) -> dict[str, list[str]]:
    a, b = pair
    merged = a + b
    new_splits = {}
    for word, symbols in splits.items():
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        new_splits[word] = out
    return new_splits


NUM_MERGES = 15
merges: list[tuple[str, str]] = []

for step in range(NUM_MERGES):
    pair_counts = get_pair_counts(splits)
    if not pair_counts:
        break
    best_pair = pair_counts.most_common(1)[0][0]
    merges.append(best_pair)
    splits = merge_pair(best_pair, splits)
    print(f"merge {step + 1:2d}: {best_pair} -> {''.join(best_pair)!r} "
          f"(seen {pair_counts[best_pair]}x)")

print(f"\nlearned {len(merges)} merges")
print("resulting word segmentations:")
for word in sorted(word_freqs):
    print(f"  {word!r:12s} -> {splits[word]}")
