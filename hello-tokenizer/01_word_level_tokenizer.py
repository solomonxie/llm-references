# $ venv/bin/python hello-tokenizer/01_word_level_tokenizer.py
#
# Goal: the simplest possible tokenizer — split on whitespace/punctuation,
# assign each distinct word an id. Shows immediately why this breaks: any
# word not seen during vocab-building has nowhere to go.
# Step 1: Word-level tokenizer + vocab, and the out-of-vocabulary problem

import re

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barks at the fox",
]

# Split on runs of non-word characters, keep punctuation as its own token.
def word_split(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text)

# Vocab: every distinct word seen in the corpus, plus a reserved UNK slot.
UNK = "<unk>"
words = sorted({w for line in corpus for w in word_split(line)})
stoi = {UNK: 0, **{w: i + 1 for i, w in enumerate(words)}}
itos = {i: w for w, i in stoi.items()}

print(f"vocab size: {len(stoi)}")
print(f"vocab: {words}")


def encode(text: str) -> list[int]:
    return [stoi.get(w, stoi[UNK]) for w in word_split(text)]


def decode(ids: list[int]) -> str:
    return " ".join(itos[i] for i in ids)


seen = "the fox jumps"
unseen = "the fox sprints quickly"  # "sprints" and "quickly" never appeared in the corpus

print(f"\n{seen!r} -> {encode(seen)} -> {decode(encode(seen))!r}")
print(f"{unseen!r} -> {encode(unseen)} -> {decode(encode(unseen))!r}")
print("\nboth unseen words collapse to <unk> — all information about them is lost.")
print("subword tokenizers (03+) exist precisely to avoid this.")
