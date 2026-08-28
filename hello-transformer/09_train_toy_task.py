# $ venv/bin/python hello-transformer/09_train_toy_task.py
#
# Goal: prove the thing from step 8 actually learns. Task: reverse a
# sequence of random integers, e.g. [3, 7, 1, 9] -> [9, 1, 7, 3] — trivial
# for a human, but the model starts knowing nothing about it and has to pick
# up the pattern purely from (source, target) examples and gradient descent.
#
# Training a seq2seq model uses "teacher forcing": the decoder is fed the
# *correct* previous target tokens (shifted right by one, with a BOS token
# at the front) rather than its own previous predictions — this lets every
# position in a whole target sequence train in one parallel forward pass,
# instead of one slow token-at-a-time loop. Inference (the bottom of this
# file) can't do that — there's no "correct previous token" to feed, so it
# really does generate one token at a time, feeding each prediction back in.
#
# Also switches from single-sequence (seq_len, d_model) tensors, used in
# steps 1-8 for readability, to batched (batch, seq_len, d_model) ones —
# training on one example at a time would work but wastes the GPU/CPU's
# ability to do many examples' matmuls at once.

import math

import torch

torch.manual_seed(0)

VOCAB_SIZE = 12  # digits 0-9, plus BOS and PAD (see below)
BOS, PAD = 10, 11
SEQ_LEN = 6


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(max_len).unsqueeze(1)
    dim = torch.arange(0, d_model, 2)
    freq = torch.exp(dim * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * freq)
    pe[:, 1::2] = torch.cos(position * freq)
    return pe


def causal_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)


class MultiHeadAttention(torch.nn.Module):
    """Same math as steps 4/6/7 — the only change is an added leading batch
    dimension, so every shape below gains a `batch` in front."""

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
        batch, seq_len, _ = t.shape
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        return t.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len_q, _ = query.shape
        Q, K, V = self._split_heads(self.W_q(query)), self._split_heads(self.W_k(key)), self._split_heads(self.W_v(value))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (batch, num_heads, seq_len_q, seq_len_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))  # (seq_len_q, seq_len_k) broadcasts over batch/heads
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).reshape(batch, seq_len_q, -1)
        return self.W_o(out)


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
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.token_embed = torch.nn.Embedding(vocab_size, d_model)
        self.register_buffer("pe", sinusoidal_positional_encoding(max_len, d_model))
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.output_head = torch.nn.Linear(d_model, vocab_size)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        return self.token_embed(token_ids) * math.sqrt(self.d_model) + self.pe[:seq_len]

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(src_ids)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, tgt_ids: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y = self.embed(tgt_ids)
        mask = causal_mask(tgt_ids.shape[-1])
        for layer in self.decoder_layers:
            y = layer(y, encoder_out, mask)
        return self.output_head(y)

    def forward(self, src_ids: torch.Tensor, tgt_in_ids: torch.Tensor) -> torch.Tensor:
        return self.decode(tgt_in_ids, self.encode(src_ids))


def make_batch(batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One training batch: random digit sequences as source, their reverse as
    target. Returns (src, decoder_input, decoder_target) — decoder_input is
    the target shifted right with BOS prepended (teacher forcing);
    decoder_target is what the model's logits at each position should
    predict."""
    src = torch.randint(0, 10, (batch_size, seq_len))
    target = src.flip(dims=[1])
    decoder_input = torch.cat([torch.full((batch_size, 1), BOS), target[:, :-1]], dim=1)
    return src, decoder_input, target


@torch.no_grad()
def generate(model: Transformer, src: torch.Tensor, max_len: int) -> torch.Tensor:
    """Real (non-teacher-forced) inference: encode the source once, then
    generate target tokens one at a time, each new prediction fed back in as
    the next step's decoder input — the only way to decode when there's no
    ground-truth target to peek at."""
    model.eval()
    encoder_out = model.encode(src.unsqueeze(0))
    generated = torch.tensor([[BOS]])
    for _ in range(max_len):
        logits = model.decode(generated, encoder_out)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    model.train()
    return generated[0, 1:]  # drop the leading BOS


if __name__ == "__main__":
    d_model, num_heads, d_ff, num_layers = 32, 4, 128, 2
    model = Transformer(VOCAB_SIZE, d_model, num_heads, d_ff, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    print(f"training on: reverse a length-{SEQ_LEN} sequence of digits 0-9")
    for step in range(1, 301):
        src, decoder_input, decoder_target = make_batch(batch_size=32, seq_len=SEQ_LEN)
        logits = model(src, decoder_input)  # (batch, seq_len, vocab_size)

        # cross_entropy wants (N, C) logits vs (N,) targets — flatten the
        # batch and seq_len dims together, one prediction per row.
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), decoder_target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"step {step:>3}  loss {loss.item():.4f}")

    print("\ninference (real autoregressive decoding, no teacher forcing):")
    for _ in range(5):
        src = torch.randint(0, 10, (SEQ_LEN,))
        predicted = generate(model, src, max_len=SEQ_LEN)
        expected = src.flip(dims=[0])
        match = "✓" if torch.equal(predicted, expected) else "✗"
        print(f"  src={src.tolist()}  predicted={predicted.tolist()}  expected={expected.tolist()}  {match}")
