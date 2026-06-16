# $ venv/bin/python hello-tokenizer/05_byte_level_bpe.py
#
# Goal: steps 3-4's BPE still fails on a character never seen in training
# (an emoji, accented letter, CJK text, ...) -- there's no <unk> escape
# hatch for those since they're not even in the starting symbol alphabet.
# GPT-2's fix: run BPE over raw UTF-8 *bytes* (always exactly 256 possible
# starting symbols) instead of characters, so every possible string is
# representable no matter what characters it contains.
# Step 5: Byte-level BPE — an alphabet of 256 always covers any input

from collections import Counter

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barks at the fox",
    "café naïve \U0001f98a",  # accented chars + an emoji -- never seen above
]

END = "</w>"


# A byte is just an int 0-255; represent each as its own symbol string so it
# can be merged with the same code as steps 3-4.
def word_to_byte_symbols(word: str) -> list[str]:
    return [str(b) for b in word.encode("utf-8")] + [END]


word_freqs = Counter(w for line in corpus for w in line.split())
splits = {word: word_to_byte_symbols(word) for word in word_freqs}


def get_pair_counts(splits):
    counts = Counter()
    for word, symbols in splits.items():
        freq = word_freqs[word]
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += freq
    return counts


def merge_pair(pair, splits):
    a, b = pair
    merged = a + "," + b  # comma-join so merged byte-groups stay unambiguous
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


NUM_MERGES = 20
merges = []
for step in range(NUM_MERGES):
    pair_counts = get_pair_counts(splits)
    if not pair_counts:
        break
    best_pair = pair_counts.most_common(1)[0][0]
    merges.append(best_pair)
    splits = merge_pair(best_pair, splits)

print(f"learned {len(merges)} merges over a 256-symbol byte alphabet\n")
for word in sorted(word_freqs):
    print(f"  {word!r:10s} bytes={list(word.encode('utf-8'))} -> {splits[word]}")


def bytes_to_text(symbols: list[str]) -> str:
    ids = []
    for sym in symbols:
        if sym == END:
            continue
        ids.extend(int(x) for x in sym.split(","))
    return bytes(ids).decode("utf-8")


# Round-trip every word, including the emoji/accented one -- no <unk> exists
# anywhere in this scheme because every string decomposes into bytes.
print("\nround-trip check (decoding the byte ids back to text):")
for word in sorted(word_freqs):
    print(f"  {word!r} -> {bytes_to_text(splits[word])!r}")
