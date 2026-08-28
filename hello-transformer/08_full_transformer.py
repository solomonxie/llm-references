# $ venv/bin/python hello-transformer/08_full_transformer.py
#
# Goal: wire steps 1-7 together into one Transformer class end to end —
# tokenize -> embed + positional encoding -> encoder stack -> decoder stack
# -> project back to vocabulary logits. This is the exact architecture from
# "Attention Is All You Need," just at toy scale (d_model=16 instead of 512,
# 2 layers instead of 6). Nothing here is trained yet — that's step 9.

import math

import torch

torch.manual_seed(0)


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(seq_len).unsqueeze(1)
    dim = torch.arange(0, d_model, 2)
    freq = torch.exp(dim * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * freq)
    pe[:, 1::2] = torch.cos(position * freq)
    return pe


def causal_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        return t.view(t.shape[0], self.num_heads, self.d_k).transpose(0, 1)

    def forward(self, query, key, value, mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len_q = query.shape[0]
        Q, K, V = self._split_heads(self.W_q(query)), self._split_heads(self.W_k(key)), self._split_heads(self.W_v(value))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.W_o((weights @ V).transpose(0, 1).reshape(seq_len_q, -1))


class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_model, d_ff), torch.nn.ReLU(), torch.nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.self_attn(x, x, x))
        x = self.norm2(x + self.ffn(x))
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.self_attn(x, x, x, mask=mask))
        x = self.norm2(x + self.cross_attn(x, encoder_out, encoder_out))
        x = self.norm3(x + self.ffn(x))
        return x


class Transformer(torch.nn.Module):
    """The full encoder-decoder stack, plus the embedding/positional-encoding
    front end (steps 1-2) and a final Linear "output head" back to
    vocabulary-sized logits — one score per vocab entry, per target position,
    ready for softmax/argmax (greedy) or top-k/nucleus sampling."""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.token_embed = torch.nn.Embedding(vocab_size, d_model)
        self.register_buffer("pe", sinusoidal_positional_encoding(max_len, d_model))
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.output_head = torch.nn.Linear(d_model, vocab_size)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[0]
        # sqrt(d_model) scale (also from the original paper) makes the
        # embedding's magnitude comparable to the positional encoding's,
        # which is bounded to roughly [-1, 1] regardless of d_model.
        return self.token_embed(token_ids) * math.sqrt(self.d_model) + self.pe[:seq_len]

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(src_ids)
        for layer in self.encoder_layers:
            x = layer(x)
        encoder_out = x

        y = self.embed(tgt_ids)
        mask = causal_mask(tgt_ids.shape[0])
        for layer in self.decoder_layers:
            y = layer(y, encoder_out, mask)

        return self.output_head(y)  # (tgt_len, vocab_size) — logits, not yet probabilities


if __name__ == "__main__":
    vocab_size, d_model, num_heads, d_ff, num_layers = 30, 16, 4, 64, 2

    model = Transformer(vocab_size, d_model, num_heads, d_ff, num_layers)
    src_ids = torch.randint(0, vocab_size, (7,))
    tgt_ids = torch.randint(0, vocab_size, (5,))

    logits = model(src_ids, tgt_ids)
    print(f"src_ids: {src_ids.tolist()}")
    print(f"tgt_ids: {tgt_ids.tolist()}")
    print(f"logits shape: {tuple(logits.shape)}  (tgt_len, vocab_size)")

    predicted_next_tokens = logits.argmax(dim=-1)
    print(f"greedy-decoded token id per position: {predicted_next_tokens.tolist()}")
    print(f"total trainable params: {sum(p.numel() for p in model.parameters()):,}")

    # Meaningless output right now (random init, no training) — position i's
    # logits are meant to predict token i+1, but nothing has taught the model
    # that yet. See 09_train_toy_task.py.
