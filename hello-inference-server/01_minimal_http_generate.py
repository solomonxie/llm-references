# $ venv/bin/python hello-inference-server/01_minimal_http_generate.py
#
# Goal: the simplest possible inference server -- one HTTP endpoint, one
# request handled at a time, no caching, no concurrency. Every later step
# in this series adds exactly one thing production servers need that this
# doesn't have yet. Tokens here are just small integers (a toy 20-symbol
# vocabulary, untrained weights) -- the point is the SERVING mechanics, not
# language modeling quality; see `hello-tokenizer`/`hello-inference` for
# the text and generation-quality side.
# Step 1: A minimal FastAPI /generate endpoint, one request at a time

import threading
import time

import requests
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

torch.manual_seed(0)

VOCAB_SIZE, D_MODEL, PORT = 20, 32, 8001


class TinyLM(nn.Module):
    """A single self-attention layer + output head -- just enough structure
    to have a real forward pass and a real per-step generation loop."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.W_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)  # (B, T, D)
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = q @ k.transpose(-2, -1) / D_MODEL**0.5
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        return self.head(out)  # (B, T, VOCAB_SIZE)


model = TinyLM()
model.eval()


@torch.no_grad()
def generate(prompt: list[int], max_new_tokens: int) -> list[int]:
    tokens = list(prompt)
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([tokens]))
        next_token = int(logits[0, -1].argmax())
        tokens.append(next_token)
    return tokens


app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: list[int]
    max_new_tokens: int = 10


@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    return {"generated": generate(req.prompt, req.max_new_tokens)}


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


threading.Thread(target=run_server, daemon=True).start()
time.sleep(1.0)  # give uvicorn a moment to bind before the client below hits it

response = requests.post(f"http://127.0.0.1:{PORT}/generate", json={"prompt": [1, 2, 3], "max_new_tokens": 8})
print(f"status: {response.status_code}")
print(f"response: {response.json()}")

print("\nthis handles exactly one request at a time -- a second request sent while")
print("the first is still generating would just wait for it (steps 4-5 fix that).")
