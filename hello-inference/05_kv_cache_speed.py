# $ venv/bin/python 05_kv_cache_speed.py
#
# Goal: why real inference servers cache K/V instead of recomputing. Every
# decode step so far (01-04) recomputes attention over the ENTIRE sequence
# from scratch, even though tokens 0..n-1's Key/Value vectors never change
# once computed — only the newest token adds a new K/V pair. A KV-cache
# stores those vectors across steps and computes attention for just the new
# token against the cached K/V, instead of redoing the whole sequence.
#
# Uses a small hand-built decoder (same shape as hello-transformer's, random
# untrained weights — output quality isn't the point here, latency is) so
# the caching logic is plain tensor ops under direct control, not hidden
# inside a library's cache implementation.

import math
import time

import torch

torch.manual_seed(0)


class CachedSelfAttention(torch.nn.Module):
    """Single-head causal self-attention with an explicit, inspectable cache:
    each call appends the new token's K/V onto whatever was passed in and
    returns the grown cache alongside the output, for the caller to pass
    into the next call."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None):
        # x: (batch, new_len, d_model) — new_len is the whole prompt on the
        # first call, then exactly 1 (just the newest token) on every call
        # after, once a cache is being used.
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)

        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=1)  # (batch, seen_so_far + new_len, d_model)
            v = torch.cat([past_v, v], dim=1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_model)  # (batch, new_len, seen_so_far + new_len)
        # Causal mask over the FULL key range: query position i (within this
        # call's new_len) may attend to every cached key plus new keys up to
        # and including itself.
        new_len, total_len = scores.shape[-2], scores.shape[-1]
        offset = total_len - new_len  # how many cached (already-causal) positions precede this call's queries
        causal = torch.triu(torch.ones(new_len, total_len, dtype=torch.bool), diagonal=offset + 1)
        scores = scores.masked_fill(causal, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        return output, (k, v)


d_model, seq_len = 64, 200  # 200 decode steps makes the recompute-every-step cost obvious
model = CachedSelfAttention(d_model)
x = torch.randn(1, 1, d_model)  # stand-in for one token's embedding, fed in one at a time


@torch.no_grad()
def decode_without_cache(steps: int) -> float:
    """Recomputes from scratch every step — each step re-runs attention over
    every token generated so far, from position 0."""
    generated = x.clone()
    start = time.perf_counter()
    for _ in range(steps):
        output, _ = model(generated, cache=None)  # no cache in -> attends over the whole `generated` sequence
        next_token = output[:, -1:, :]  # stand-in for "sample/argmax the next token's embedding"
        generated = torch.cat([generated, next_token], dim=1)
    return time.perf_counter() - start


@torch.no_grad()
def decode_with_cache(steps: int) -> float:
    """Each step passes in only the ONE new token, plus the cache from the
    previous step — attention only computes new keys/values for that one
    token, reusing everything already cached."""
    current = x.clone()
    cache = None
    start = time.perf_counter()
    for _ in range(steps):
        output, cache = model(current, cache=cache)  # current is length 1 after the first step
        current = output[:, -1:, :]
    return time.perf_counter() - start


no_cache_time = decode_without_cache(seq_len)
cache_time = decode_with_cache(seq_len)

print(f"generating {seq_len} tokens:")
print(f"  without cache (recompute every step): {no_cache_time:.3f}s")
print(f"  with cache (reuse past K/V):          {cache_time:.3f}s")
print(f"  speedup: {no_cache_time / cache_time:.1f}x")

# The gap grows with sequence length — without a cache, step n redoes O(n)
# work it already did at step n-1, so total work across all steps is
# O(seq_len^2); with a cache, each step is O(1) new work, O(seq_len) total.
print("\nsame comparison at a shorter length (gap should be much smaller):")
short = 20
print(f"  without cache: {decode_without_cache(short):.3f}s   with cache: {decode_with_cache(short):.3f}s")
