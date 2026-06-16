# $ venv/bin/python hello-llm-training/04_loss_curve_and_sampling.py
#
# Goal: two things every real pretraining run adds on top of step 3's bare
# loop -- a learning-rate schedule (warmup then cosine decay, instead of a
# single fixed lr) and a visible loss curve to confirm it's actually
# converging, not just running. Also swaps greedy decoding for temperature
# sampling at the end: greedy always picks the single most likely next
# character, which on a repetitive corpus like this one tends to lock onto
# one sentence and repeat it; sampling shows the model learned a
# *distribution* over plausible next characters, not one fixed answer.
# Step 4: LR warmup+cosine schedule, loss curve, temperature-sampled generation

import math

import torch

torch.manual_seed(0)

# --- Steps 1-3 recap: corpus/tokenizer, decoder-only GPT, LM batching ---
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


CORPUS_IDS = encode(TOY_CORPUS)


def make_batch(batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(CORPUS_IDS) - seq_len - 1, (batch_size,))
    inputs = torch.stack([torch.tensor(CORPUS_IDS[s : s + seq_len]) for s in starts])
    targets = torch.stack([torch.tensor(CORPUS_IDS[s + 1 : s + 1 + seq_len]) for s in starts])
    return inputs, targets


# Step 4: warmup (lr ramps 0 -> peak linearly) then cosine decay (peak -> ~0)
# -- a fixed lr either wastes early steps being too cautious or, if set high
# enough to train fast, destabilizes once the loss is already low. Real
# pretraining runs (GPT-2/3 included) use exactly this shape.
def lr_at_step(step: int, total_steps: int, peak_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * progress))


def loss_sparkline(losses: list[float], width: int = 60) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    bucket_size = max(1, len(losses) // width)
    buckets = [losses[i : i + bucket_size] for i in range(0, len(losses), bucket_size)]
    means = [sum(b) / len(b) for b in buckets]
    lo, hi = min(means), max(means)
    return "".join(blocks[int((hi - m) / (hi - lo + 1e-9) * (len(blocks) - 1))] for m in means)


@torch.no_grad()
def generate(model: GPT, prompt_ids: list[int], max_new_tokens: int, temperature: float = 0.0) -> list[int]:
    model.eval()
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.max_len :]])
        logits = model(context)[0, -1, :]
        if temperature == 0.0:
            next_id = logits.argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)
    model.train()
    return ids


if __name__ == "__main__":
    D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_LEN = 64, 4, 128, 2, 32
    TOTAL_STEPS, WARMUP_STEPS, PEAK_LR = 800, 50, 3e-3

    model = GPT(len(VOCAB), D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_LEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)

    loss_history = []
    for step in range(1, TOTAL_STEPS + 1):
        lr = lr_at_step(step, TOTAL_STEPS, PEAK_LR, WARMUP_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = make_batch(batch_size=32, seq_len=MAX_LEN)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(VOCAB)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if step % 100 == 0 or step == 1:
            print(f"step {step:>3}  lr {lr:.5f}  loss {loss.item():.4f}")

    print(f"\nloss curve (step 1 -> {TOTAL_STEPS}, high=bad, low=good):")
    print(loss_sparkline(loss_history))

    prompt = "the cat is"
    print(f"\nprompt: {prompt!r}")
    for temperature in [0.0, 0.5, 1.0]:
        generated = generate(model, encode(prompt), max_new_tokens=40, temperature=temperature)
        label = "greedy" if temperature == 0.0 else f"temperature={temperature}"
        print(f"  {label:<16} {decode(generated)!r}")
