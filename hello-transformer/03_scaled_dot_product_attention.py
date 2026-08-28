# $ venv/bin/python hello-transformer/03_scaled_dot_product_attention.py
#
# Goal: the actual mechanism the "Attention" in "Attention Is All You Need"
# refers to. Every token produces a Query ("what am I looking for?"), a Key
# ("what do I offer?"), and a Value ("what do I actually contribute?"). A
# token's new representation is a weighted sum of every token's Value,
# weighted by how well its Query matches each Key.

import math

import torch

torch.manual_seed(0)

seq_len = 5
d_model = 8

x = torch.randn(seq_len, d_model)  # stand-in for token+position embeddings from steps 1-2

# Q, K, V are just three separate learned linear projections of the same
# input — same shape in, same shape out here (real models often project to a
# smaller d_k, see 04_multi_head_attention.py).
W_q = torch.nn.Linear(d_model, d_model, bias=False)
W_k = torch.nn.Linear(d_model, d_model, bias=False)
W_v = torch.nn.Linear(d_model, d_model, bias=False)

Q = W_q(x)  # (seq_len, d_model)
K = W_k(x)
V = W_v(x)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.shape[-1]

    #   Q (seq_len, d_k) @ K.T (d_k, seq_len) = scores (seq_len, seq_len)
    #   scores[i, j] = how much token i's query matches token j's key
    #
    #        K.T
    #      ┌───────────┐
    #      │           │
    #  Q   │           │      scores
    # ┌───┐│           │    ┌───────────┐
    # │   ││           │  = │ i attends │  scores[i, j] = Q[i] · K[j]
    # └───┘│           │    │ to j here │
    #      └───────────┘    └───────────┘
    scores = Q @ K.transpose(-2, -1)

    # Without the 1/sqrt(d_k) scale, dot products grow with d_k and push
    # softmax into a near-one-hot regime (vanishing gradients into Q/K).
    scores = scores / math.sqrt(d_k)

    # Softmax over the *last* dim: each row of `weights` is a probability
    # distribution over "how much should token i attend to each token j".
    weights = torch.softmax(scores, dim=-1)

    # (seq_len, seq_len) @ (seq_len, d_v) = (seq_len, d_v) — each output row
    # is a weighted average of every token's Value.
    output = weights @ V
    return output, weights


output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Q/K/V shape:      {tuple(Q.shape)}  (seq_len, d_model)")
print(f"attention weights: {tuple(weights.shape)}  (seq_len, seq_len) — one row per query token")
print(f"output shape:      {tuple(output.shape)}  (seq_len, d_model) — same shape as the input")
print(f"\nrow 0 sums to {weights[0].sum():.4f} (softmax rows always sum to 1)")
print("attention weights (rounded):")
print(weights.round(decimals=2))

# This is "self-attention" specifically because Q, K, and V all come from the
# same input x — a token attending to other tokens in its own sequence.
# Cross-attention (step 7, the decoder attending to the encoder) is the exact
# same function with Q from one sequence and K/V from another.
