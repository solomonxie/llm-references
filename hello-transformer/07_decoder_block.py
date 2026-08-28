# $ venv/bin/python hello-transformer/07_decoder_block.py
#
# Goal: the decoder side — same building blocks as the encoder (step 6), plus
# two changes:
#   - self-attention is *masked* (causal): position i may only attend to
#     positions <= i. Without this, predicting token i could "see" tokens
#     after it — fine at training time (the whole target is known), but
#     impossible to reproduce at inference time (tokens are generated one at
#     a time, later ones don't exist yet). Masking makes training match
#     inference.
#   - a second attention sublayer, *cross*-attention: queries come from the
#     decoder, but keys/values come from the encoder's output. This is how
#     the decoder actually looks at the source sequence.

import math

import torch

torch.manual_seed(0)


class MultiHeadAttention(torch.nn.Module):
    """Identical to 06_encoder_block.py — query/key/value are separate
    arguments specifically so this same class serves both self-attention
    (query=key=value) and cross-attention (query from the decoder,
    key=value from the encoder) below."""

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
        return t.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len_q = query.shape[0]
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        head_outputs = weights @ V
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


def causal_mask(seq_len: int) -> torch.Tensor:
    #   position i's row: True at j > i ("cannot see the future")
    #        j=0  j=1  j=2  j=3
    #   i=0 [ F,   T,   T,   T ]     row i has (seq_len - 1 - i) True entries
    #   i=1 [ F,   F,   T,   T ]     — position 0 sees only itself,
    #   i=2 [ F,   F,   F,   T ]       the last position sees everything
    #   i=3 [ F,   F,   F,   F ]
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, look_ahead_mask: torch.Tensor) -> torch.Tensor:
        #   x ─► masked self_attn(x,x,x) ─► +x ─► norm1 ─┐
        #        (causal: only sees target tokens so far) │
        #                                                  ▼
        #        cross_attn(·, encoder_out, encoder_out) ─► +(·) ─► norm2 ─┐
        #        (query=decoder, key/value=encoder — "attend to the source") │
        #                                                                     ▼
        #                                              ffn(·) ─► +(·) ─► norm3 ─► out
        self_attn_out = self.self_attn(x, x, x, mask=look_ahead_mask)
        x = self.norm1(x + self_attn_out)

        cross_attn_out = self.cross_attn(x, encoder_out, encoder_out)  # no mask — full source is visible
        x = self.norm2(x + cross_attn_out)

        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        mask = causal_mask(x.shape[0])
        for layer in self.layers:
            x = layer(x, encoder_out, mask)
        return x


if __name__ == "__main__":
    src_len, tgt_len, d_model, num_heads, d_ff, num_layers = 6, 4, 8, 2, 32, 2

    encoder_out = torch.randn(src_len, d_model)  # stand-in for step 6's Encoder(x) output
    target = torch.randn(tgt_len, d_model)  # stand-in for embed(target_tokens) + positional_encoding

    decoder = Decoder(d_model, num_heads, d_ff, num_layers)
    out = decoder(target, encoder_out)

    print(f"encoder output (source) shape: {tuple(encoder_out.shape)}  (src_len, d_model)")
    print(f"decoder input (target) shape:  {tuple(target.shape)}  (tgt_len, d_model)")
    print(f"decoder output shape:          {tuple(out.shape)}  (tgt_len, d_model — src_len only mattered for cross-attention)")

    print("\ncausal mask for tgt_len=4:")
    print(causal_mask(tgt_len))
