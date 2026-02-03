# $ venv/bin/python hello-transformer/06_encoder_block.py
#
# Goal: assemble steps 2-5 into one real encoder layer — multi-head
# self-attention, wrapped in Add & Norm, followed by a feedforward network,
# also wrapped in Add & Norm — then stack N of them. This is the left half
# of the original Transformer diagram (the right half, the decoder, is
# step 7).
# Step 6: One full encoder layer (self-attn + FFN, each Add & Norm'd), stacked N deep

import math

import torch

torch.manual_seed(0)


class MultiHeadAttention(torch.nn.Module):
    """Same mechanism as 04_multi_head_attention.py, packaged as a reusable
    module. Takes separate query/key/value inputs (all equal for
    self-attention; different for the decoder's cross-attention in step 7)
    plus an optional mask (used for causal masking in step 7)."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide evenly across heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[0]
        return t.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  # (num_heads, seq_len, d_k)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len_q = query.shape[0]
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            # Positions where mask is True get -inf *before* softmax, so
            # softmax sends their weight to (effectively) zero.
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        head_outputs = weights @ V  # (num_heads, seq_len_q, d_k)
        concatenated = head_outputs.transpose(0, 1).reshape(seq_len_q, -1)
        return self.W_o(concatenated)


class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff), torch.nn.ReLU(), torch.nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(torch.nn.Module):
    """One encoder layer: self-attention sublayer, then FFN sublayer, each
    wrapped in its own residual + LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #   x ──► self_attn(x,x,x) ──► + x ──► norm1 ──► ffn(·) ──► + (·) ──► norm2 ──► out
        attn_out = self.self_attn(x, x, x)  # query=key=value=x: self-attention
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class Encoder(torch.nn.Module):
    """N identical encoder layers stacked — each layer's output feeds the
    next. Stacking is what lets later layers build increasingly abstract
    representations on top of what earlier layers found."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    seq_len, d_model, num_heads, d_ff, num_layers = 5, 8, 2, 32, 3

    x = torch.randn(seq_len, d_model)  # stand-in for embed(tokens) + positional_encoding from steps 1-2
    encoder = Encoder(d_model, num_heads, d_ff, num_layers)
    out = encoder(x)

    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {tuple(out.shape)}  — unchanged shape after {num_layers} stacked layers")
    print(f"params in one encoder layer: {sum(p.numel() for p in encoder.layers[0].parameters()):,}")
    print(f"params in the full encoder:  {sum(p.numel() for p in encoder.parameters()):,}")
