# $ venv/bin/python hello-llm-training/01_toy_corpus_and_tokenizer.py
#
# Goal: the raw material every later step trains on. A small, repetitive toy
# corpus (so a tiny model can plausibly learn its patterns in a few hundred
# CPU steps) and a char-level tokenizer built from scratch -- see
# `hello-tokenizer/02_char_level_tokenizer.py` for that mechanism in
# isolation; here it's just the first link in the pretraining pipeline.
# Step 1: Toy corpus + char-level tokenizer, encode/decode round trip

TOY_CORPUS = (
    """
the sun is up. the sky is blue. the cat is happy.
the moon is up. the sky is dark. the cat is sleepy.
the sun is down. the sky is pink. the dog is happy.
the moon is down. the sky is grey. the dog is sleepy.
"""
    .strip()
    + "\n"
) * 40  # repeated so there's enough data for many training crops (step 3)

VOCAB = sorted(set(TOY_CORPUS))
STOI = {ch: i for i, ch in enumerate(VOCAB)}
ITOS = {i: ch for ch, i in STOI.items()}


def encode(text: str) -> list[int]:
    return [STOI[ch] for ch in text]


def decode(ids: list[int]) -> str:
    return "".join(ITOS[i] for i in ids)


if __name__ == "__main__":
    print(f"corpus length: {len(TOY_CORPUS):,} chars")
    print(f"vocab size: {len(VOCAB)}  {VOCAB}")

    sample = "the cat is happy."
    ids = encode(sample)
    print(f"\nencode({sample!r}) = {ids}")
    print(f"decode(...)         = {decode(ids)!r}")
    assert decode(ids) == sample
