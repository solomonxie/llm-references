# $ venv/bin/python hello-transformer/04_multi_head_attention.py
#
# Goal: run attention (step 3) several times in parallel, each with its own
# learned Q/K/V projections into a smaller subspace, then combine the
# results. One head might learn to track "what verb applies to me", another
# "what's the previous token" — a single head has to compromise across all
# of that in one attention pattern; multiple heads don't.
# Step 4: Splitting Q/K/V across heads, running attention in parallel, concatenating back

import math

import torch

torch.manual_seed(0)

seq_len = 5
d_model = 8
num_heads = 2
d_k = d_model // num_heads  # each head works in a smaller space; heads' outputs concat back to d_model

x = torch.randn(seq_len, d_model)

W_q = torch.nn.Linear(d_model, d_model, bias=False)  # projects to all heads at once — sliced apart below
W_k = torch.nn.Linear(d_model, d_model, bias=False)
W_v = torch.nn.Linear(d_model, d_model, bias=False)
W_o = torch.nn.Linear(d_model, d_model, bias=False)  # combines the concatenated heads back to d_model


def split_heads(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    seq_len, d_model = t.shape
    d_k = d_model // num_heads
    #   (seq_len, d_model) -> (seq_len, num_heads, d_k) -> (num_heads, seq_len, d_k)
    #
    #   d_model columns split into num_heads equal chunks of width d_k —
    #   head h owns columns [h*d_k : (h+1)*d_k], not a learned split, just a
    #   reshape (the projection W_q/W_k/W_v already learned what goes where).
    return t.view(seq_len, num_heads, d_k).transpose(0, 1)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # broadcasts over the leading heads dim
    weights = torch.softmax(scores, dim=-1)
    return weights @ V


Q = split_heads(W_q(x), num_heads)  # (num_heads, seq_len, d_k)
K = split_heads(W_k(x), num_heads)
V = split_heads(W_v(x), num_heads)
print(f"per-head Q/K/V shape: {tuple(Q.shape)}  (num_heads, seq_len, d_k)")

head_outputs = scaled_dot_product_attention(Q, K, V)  # (num_heads, seq_len, d_k), each head independent
print(f"per-head output shape: {tuple(head_outputs.shape)}")

#   head 0 output (seq_len, d_k)   head 1 output (seq_len, d_k)
#   ┌───────────────┐              ┌───────────────┐
#   │               │  concat      │               │        (seq_len, d_model)
#   │               │  ────────►   │               │  ───►  one row per token again
#   └───────────────┘              └───────────────┘
concatenated = head_outputs.transpose(0, 1).reshape(seq_len, d_model)
print(f"concatenated shape:    {tuple(concatenated.shape)}  (seq_len, d_model)")

output = W_o(concatenated)  # final learned mix across heads — without this, heads stay independent
print(f"multi-head output shape: {tuple(output.shape)}  (seq_len, d_model) — same shape the input came in as")

# This whole file is what nn.MultiheadAttention does internally. Real models
# use num_heads like 8-96; d_model // num_heads must divide evenly.
