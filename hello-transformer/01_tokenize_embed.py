# $ venv/bin/python hello-transformer/01_tokenize_embed.py
#
# Goal: turn text into vectors. Every transformer starts here — a token is
# just an integer id, and an embedding table is just a lookup from id -> a
# learned vector. Nothing "attention"-related happens yet.
# Step 1: Char-level tokenizer + nn.Embedding lookup

import torch

torch.manual_seed(0)

text = "hello transformer"

# Character-level tokenizer: the whole vocab is just the distinct characters
# seen. Word-level or subword (BPE) tokenizers use the same id -> vector idea,
# just with a smarter vocabulary.
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)
print(f"vocab ({vocab_size} chars): {chars}")

token_ids = torch.tensor([stoi[ch] for ch in text])
print(f"text:      {text!r}")
print(f"token_ids: {token_ids.tolist()}")

# nn.Embedding is a (vocab_size, d_model) matrix — row i is token i's vector.
# Indexing it with a batch of ids does a lookup, no matrix multiply needed.
d_model = 8  # kept tiny so printed vectors stay readable; real models use 512-12288+
embed = torch.nn.Embedding(vocab_size, d_model)

#   embed.weight
#   ┌─────────────────────────┐
#   │ row 0 (token 'a')       │  each row: d_model floats
#   │ row 1 (token 'e')       │
#   │ ...                     │
#   │ row vocab_size-1        │
#   └─────────────────────────┘
#   embed(token_ids) gathers len(token_ids) of these rows, in order.

x = embed(token_ids)
print(f"\nembedding table shape: {tuple(embed.weight.shape)}  (vocab_size, d_model)")
print(f"embedded input shape:  {tuple(x.shape)}  (seq_len, d_model)")
print(f"vector for {text[0]!r}: {x[0]}")

# These vectors are random right now (nn.Embedding init) — training (see
# 09_train_toy_task.py) is what shapes them into something meaningful, via
# ordinary backprop like any other nn.Parameter.
