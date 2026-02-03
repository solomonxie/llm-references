# $ venv/bin/python hello-transformer/02_positional_encoding.py
#
# Goal: give the model a sense of word order. Attention (coming in step 3)
# treats its input as an unordered *set* of vectors — nothing about it looks
# at position. So position has to be baked into the vectors themselves,
# before attention ever runs.

import math

import torch

torch.manual_seed(0)

seq_len = 6
d_model = 8

# The original "Attention Is All You Need" scheme: a fixed (non-learned)
# sinusoid per dimension, sin on even dims and cos on odd dims, each pair
# oscillating at a different frequency. Two properties make this work:
#   - every position gets a unique vector
#   - PE(pos + k) is a fixed linear function of PE(pos) (from the angle-sum
#     identity), so the model can learn to attend by *relative* offset
#
#   pe[pos, 2i]   = sin(pos / 10000^(2i/d_model))
#   pe[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    dim = torch.arange(0, d_model, 2)  # (d_model/2,) — the "2i" indices
    freq = torch.exp(dim * (-math.log(10000.0) / d_model))  # 10000^(-2i/d_model)

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * freq)
    pe[:, 1::2] = torch.cos(position * freq)
    return pe


pe = sinusoidal_positional_encoding(seq_len, d_model)
print(f"positional encoding shape: {tuple(pe.shape)}  (seq_len, d_model)")
print("pe[0] (position 0):", pe[0])
print("pe[1] (position 1):", pe[1])

#   token embeddings          positional encodings         model input
#   ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
#   │ e(tok_0)      │         │ pe(0)         │         │ e(tok_0)+pe(0)│
#   │ e(tok_1)      │    +    │ pe(1)         │    =    │ e(tok_1)+pe(1)│
#   │ ...           │         │ ...           │         │ ...           │
#   └───────────────┘         └───────────────┘         └───────────────┘
#   same shape as token embeddings — added elementwise, not concatenated
embed = torch.nn.Embedding(20, d_model)
token_ids = torch.randint(0, 20, (seq_len,))
x = embed(token_ids) + pe
print(f"\nmodel input shape (token + position): {tuple(x.shape)}")

# Modern models (GPT, BERT, ...) often use a *learned* embedding table for
# positions instead — nn.Embedding(max_len, d_model), looked up by position
# index exactly like a token embedding. Same shape, same role; only the
# sinusoid formula is replaced by learned weights.
