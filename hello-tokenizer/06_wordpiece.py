# $ venv/bin/python hello-tokenizer/06_wordpiece.py
#
# Goal: WordPiece (used by BERT) is close to BPE but differs in two ways:
# (1) subword pieces after the first character of a word are marked with a
# "##" continuation prefix, and (2) the merge criterion isn't raw pair
# frequency -- it's frequency normalized by how common each half already is,
# so it prefers merging pairs that co-occur more than chance would predict.
# Step 6: WordPiece training -- ## continuation marker + likelihood scoring

from collections import Counter

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barks at the fox",
    "the fox and the dog are friends",
]

word_freqs = Counter(w for line in corpus for w in line.split())


def word_to_symbols(word: str) -> list[str]:
    # First char plain, every char after gets a "##" continuation prefix.
    return [word[0]] + [f"##{c}" for c in word[1:]]


splits = {word: word_to_symbols(word) for word in word_freqs}


def symbol_freqs(splits) -> Counter:
    counts = Counter()
    for word, symbols in splits.items():
        for sym in symbols:
            counts[sym] += word_freqs[word]
    return counts


def pair_scores(splits) -> dict[tuple[str, str], float]:
    sym_freqs = symbol_freqs(splits)
    pair_freqs = Counter()
    for word, symbols in splits.items():
        freq = word_freqs[word]
        for a, b in zip(symbols, symbols[1:]):
            pair_freqs[(a, b)] += freq
    # BPE would just rank by pair_freqs. WordPiece instead scores by
    # freq(pair) / (freq(a) * freq(b)) -- how much more often this pair
    # co-occurs than its parts' individual frequencies would predict.
    return {
        pair: freq / (sym_freqs[pair[0]] * sym_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }


def merge_pair(pair, splits):
    a, b = pair
    merged = a + b[2:] if b.startswith("##") else a + b
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
merges = []
for step in range(NUM_MERGES):
    scores = pair_scores(splits)
    if not scores:
        break
    best_pair = max(scores, key=scores.get)
    merges.append(best_pair)
    splits = merge_pair(best_pair, splits)
    print(f"merge {step + 1:2d}: {best_pair} (score={scores[best_pair]:.4f})")

vocab = sorted({sym for symbols in splits.values() for sym in symbols})
print(f"\nvocab ({len(vocab)}): {vocab}")


# Encoding a new word: greedy longest-match-first against the vocab, WordPiece's
# signature difference from BPE's "replay the merge list" approach.
def wordpiece_encode(word: str, vocab: set[str]) -> list[str]:
    pieces, i = [], 0
    while i < len(word):
        j = len(word)
        match = None
        while j > i:
            piece = word[i:j] if i == 0 else f"##{word[i:j]}"
            if piece in vocab:
                match = piece
                break
            j -= 1
        if match is None:
            return ["<unk>"]
        pieces.append(match)
        i = j
    return pieces


vocab_set = set(vocab)
for word in ["the", "foxes", "unseen"]:
    print(f"  {word!r:10s} -> {wordpiece_encode(word, vocab_set)}")
