# $ venv/bin/python hello-llm-training/03_training_loop.py
#
# Goal: actually pretrain it. The objective is next-character prediction:
# for every position in a chunk of corpus, predict the character that comes
# next. Unlike `hello-transformer/09_train_toy_task.py`'s synthetic
# (source, target) pairs, the input and target here are the *same* text,
# offset by one -- this is the real self-supervised objective real LLM
# pretraining uses (at word-piece scale, on far more text).
# Step 3: Random corpus crops as (input, target) batches, cross-entropy loss, AdamW

import math

import torch

torch.manual_seed(0)

# --- Steps 1-2 recap: corpus/tokenizer + decoder-only GPT ---
TOY_CORPUS = (
    """
the sun is up. the sky is blue. the cat is happy.
the moon is up. the sky is dark. the cat is sleepy.
the sun is down. the sky is pink. the dog is happy.
the moon is down. the sky is grey. the dog is sleepy.
"""
    .strip()
    + "\n"
) * 40
VOCAB = sorted(set(TOY_CORPUS))
STOI = {ch: i for i, ch in enumerate(VOCAB)}
ITOS = {i: ch for ch, i in STOI.items()}


def encode(text: str) -> list[int]:
    return [STOI[ch] for ch in text]


def decode(ids: list[int]) -> str:
    return "".join(ITOS[i] for i in ids)


def causal_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = t.shape
        return t.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        Q, K, V = self._split_heads(self.W_q(x)), self._split_heads(self.W_k(x)), self._split_heads(self.W_v(x))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).reshape(batch, seq_len, -1)
        return self.W_o(out)


class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_model, d_ff), torch.nn.ReLU(), torch.nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.self_attn(x, mask))
        x = self.norm2(x + self.ffn(x))
        return x


class GPT(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_len: int):
        super().__init__()
        self.max_len = max_len
        self.token_embed = torch.nn.Embedding(vocab_size, d_model)
        self.pos_embed = torch.nn.Embedding(max_len, d_model)
        self.blocks = torch.nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.output_head = torch.nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = token_ids.shape
        positions = torch.arange(seq_len)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        mask = causal_mask(seq_len)
        for block in self.blocks:
            x = block(x, mask)
        return self.output_head(x)


@torch.no_grad()
def generate(model: GPT, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    model.eval()
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.max_len :]])
        next_id = model(context)[0, -1, :].argmax(dim=-1).item()
        ids.append(next_id)
    model.train()
    return ids


# Step 3: language-model batching -- every crop's input is corpus[i:i+seq_len],
# its target is the same window shifted one character right, i.e. "predict
# the next char at every position", not just at the end.
CORPUS_IDS = encode(TOY_CORPUS)


def make_batch(batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(CORPUS_IDS) - seq_len - 1, (batch_size,))
    inputs = torch.stack([torch.tensor(CORPUS_IDS[s : s + seq_len]) for s in starts])
    targets = torch.stack([torch.tensor(CORPUS_IDS[s + 1 : s + 1 + seq_len]) for s in starts])
    return inputs, targets


if __name__ == "__main__":
    D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_LEN = 64, 4, 128, 2, 32

    model = GPT(len(VOCAB), D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_LEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    print(f"training on: next-char prediction over a {len(CORPUS_IDS):,}-char toy corpus")
    for step in range(1, 501):
        inputs, targets = make_batch(batch_size=32, seq_len=MAX_LEN)
        logits = model(inputs)  # (batch, seq_len, vocab_size)

        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(VOCAB)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1:
            print(f"step {step:>3}  loss {loss.item():.4f}")

    prompt = "the cat is"
    generated = generate(model, encode(prompt), max_new_tokens=30)
    print(f"\nprompt:  {prompt!r}")
    print(f"trained generation:   {decode(generated)!r}  (should start resembling the corpus)")
