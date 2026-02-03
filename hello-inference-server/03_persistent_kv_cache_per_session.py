# $ venv/bin/python 03_persistent_kv_cache_per_session.py
#
# Goal: a real conversation is a sequence of turns against the SAME growing
# context. Recomputing that whole context from scratch on every turn (what
# steps 1-2 do) wastes more and more work as the conversation grows. This
# keeps a KV cache alive per session_id between HTTP requests -- a later
# request in the same session sends only its NEW tokens and reuses the
# cache built by every earlier request in that session.
# Step 3: A KV cache kept alive across requests, keyed by session_id

import threading
import time

import requests
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

torch.manual_seed(0)

VOCAB_SIZE, D_MODEL, PORT = 20, 32, 8003


class CachedSelfAttention(nn.Module):
    """Same explicit, growable cache as hello-inference/05_kv_cache_speed.py."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model

    def forward(self, x, cache):
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        if cache is not None:
            past_k, past_v = cache
            k, v = torch.cat([past_k, k], dim=1), torch.cat([past_v, v], dim=1)
        scores = q @ k.transpose(-2, -1) / self.d_model**0.5
        new_len, total_len = scores.shape[-2], scores.shape[-1]
        offset = total_len - new_len
        mask = torch.triu(torch.ones(new_len, total_len, dtype=torch.bool), diagonal=offset + 1)
        scores = scores.masked_fill(mask, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        return out, (k, v)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.attn = CachedSelfAttention(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, token_ids, cache):
        x = self.embed(token_ids)
        out, cache = self.attn(x, cache)
        return self.head(out), cache


model = TinyLM()
model.eval()
app = FastAPI()

# session_id -> {"cache": (k, v) | None, "length": int}
SESSIONS: dict[str, dict] = {}


class GenerateRequest(BaseModel):
    session_id: str
    new_tokens: list[int]
    max_new_tokens: int = 5


@torch.no_grad()
def generate(session_id: str, new_tokens: list[int], max_new_tokens: int) -> list[int]:
    session = SESSIONS.setdefault(session_id, {"cache": None, "length": 0})

    # Feed only the NEW tokens through the model -- the cache already holds
    # every earlier token's K/V from previous requests in this session.
    logits, cache = model(torch.tensor([new_tokens]), session["cache"])
    session["cache"] = cache
    session["length"] += len(new_tokens)

    generated = []
    current = int(logits[0, -1].argmax())
    for _ in range(max_new_tokens):
        generated.append(current)
        logits, cache = model(torch.tensor([[current]]), session["cache"])
        session["cache"] = cache
        session["length"] += 1
        current = int(logits[0, -1].argmax())

    return generated


@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    generated = generate(req.session_id, req.new_tokens, req.max_new_tokens)
    return {"generated": generated, "session_cache_length": SESSIONS[req.session_id]["length"]}


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


threading.Thread(target=run_server, daemon=True).start()
time.sleep(1.0)

url = f"http://127.0.0.1:{PORT}/generate"

print("turn 1 (new session, cache starts empty):")
r1 = requests.post(url, json={"session_id": "conv-1", "new_tokens": [1, 2, 3], "max_new_tokens": 5}).json()
print(f"  {r1}")

print("turn 2 (same session -- only sends 1 new token, reuses turn 1's cache):")
r2 = requests.post(url, json={"session_id": "conv-1", "new_tokens": [7], "max_new_tokens": 5}).json()
print(f"  {r2}")

print(f"\ncache length grew from {r1['session_cache_length']} to {r2['session_cache_length']} -- turn 2")
print("never recomputed the first 4 tokens' attention, only extended the cache.")
