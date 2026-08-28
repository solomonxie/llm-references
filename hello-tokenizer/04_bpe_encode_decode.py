# $ venv/bin/python 04_bpe_encode_decode.py
#
# Goal: turn the learned merge list from step 3 into an actual encode/decode
# pair that works on new text — including words never seen during training.
# Step 4: Applying learned BPE merges to encode/decode arbitrary text

from collections import Counter

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barks at the fox",
    "the fox and the dog are friends",
]

END = "</w>"


def word_to_symbols(word: str) -> list[str]:
    return list(word) + [END]


word_freqs = Counter(w for line in corpus for w in line.split())
splits = {word: word_to_symbols(word) for word in word_freqs}


def get_pair_counts(splits):
    counts = Counter()
    for word, symbols in splits.items():
        freq = word_freqs[word]
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += freq
    return counts


def merge_pair(pair, splits):
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
for _ in range(NUM_MERGES):
    pair_counts = get_pair_counts(splits)
    if not pair_counts:
        break
    best_pair = pair_counts.most_common(1)[0][0]
    merges.append(best_pair)
    splits = merge_pair(best_pair, splits)

UNK = "<unk>"
vocab = [UNK] + sorted({sym for symbols in splits.values() for sym in symbols})
stoi = {sym: i for i, sym in enumerate(vocab)}
itos = {i: sym for sym, i in stoi.items()}
merge_rank = {pair: i for i, pair in enumerate(merges)}


# Apply merges in the order they were learned — lowest rank (earliest,
# most frequent) merge wins whenever multiple pairs in a word are mergeable.
def bpe_word(word: str) -> list[str]:
    symbols = word_to_symbols(word)
    while len(symbols) > 1:
        pairs = list(zip(symbols, symbols[1:]))
        ranked = [(merge_rank[p], p) for p in pairs if p in merge_rank]
        if not ranked:
            break
        _, best = min(ranked)
        a, b = best
        merged, out, i = a + b, [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


def encode(text: str) -> list[int]:
    ids = []
    for word in text.split():
        for sym in bpe_word(word):
            ids.append(stoi.get(sym, stoi[UNK]))
    return ids


def decode(ids: list[int]) -> str:
    text = "".join(itos[i] for i in ids)
    return text.replace(END, " ").strip()


trained_word = "the"
print(f"{trained_word!r} -> {bpe_word(trained_word)}")

unseen_word = "foxes"  # not in the corpus, but shares "fox" as a learned merge
print(f"{unseen_word!r} -> {bpe_word(unseen_word)} "
      "(falls back to sub-word pieces instead of <unk>)")

sentence = "the fox and the foxes jump"
ids = encode(sentence)
print(f"\n{sentence!r}\n -> {ids}\n -> {decode(ids)!r}")
