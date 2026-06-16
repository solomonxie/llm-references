# $ venv/bin/python hello-tokenizer/02_char_level_tokenizer.py
#
# Goal: the other extreme — tokenize by individual character. No word is
# ever unseen (the vocab is just "every character that can appear"), but
# sequences get much longer and each token carries far less meaning.
# Step 2: Character-level tokenizer — no OOV, but long sequences

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barks at the fox",
]

chars = sorted({c for line in corpus for c in line})
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

print(f"vocab size: {len(stoi)} (every distinct character)")


def encode(text: str) -> list[int]:
    return [stoi[c] for c in text]  # any char outside the corpus still fails here


def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)


sentence = "the fox jumps"
ids = encode(sentence)
print(f"\n{sentence!r} -> {len(ids)} tokens -> {decode(ids)!r}")

# Compare token count against word-level splitting of the same sentence.
word_count = len(sentence.split())
print(f"word-level would need ~{word_count} tokens for the same text")
print(f"char-level needs {len(ids)} — {len(ids) / word_count:.1f}x more tokens per word")

print("\nnever hits <unk> for words in the training language, but even 'sprints',")
print("a word never seen in the corpus, encodes fine since only characters matter:")
print(f"{'sprints'!r} -> {encode('sprints')}")
print("\nthe real cost: sequence length. A model has to learn 'th', 'the', 'the ',")
print("etc. as separate compositional steps instead of getting 'the' as one unit.")
