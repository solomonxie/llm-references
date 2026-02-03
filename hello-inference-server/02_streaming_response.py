# $ venv/bin/python 02_streaming_response.py
#
# Goal: step 1's client waits for the ENTIRE completion before seeing
# anything -- fine for 8 tokens, unusable for a long response. Streaming
# sends each token to the client the moment it's generated, over Server-
# Sent Events (one `data: ...\n\n` line per token) instead of one JSON blob
# at the end. The generation loop itself doesn't change; only how each
# step's result gets flushed to the client does.
# Step 2: Streaming tokens back as they're generated, via SSE

import threading
import time

import requests
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

torch.manual_seed(0)

VOCAB_SIZE, D_MODEL, PORT = 20, 32, 8002


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.W_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = q @ k.transpose(-2, -1) / D_MODEL**0.5
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        return self.head(out)


model = TinyLM()
model.eval()
app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: list[int]
    max_new_tokens: int = 10


@torch.no_grad()
def token_stream(prompt: list[int], max_new_tokens: int):
    tokens = list(prompt)
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([tokens]))
        next_token = int(logits[0, -1].argmax())
        tokens.append(next_token)
        time.sleep(0.05)  # stand-in for real per-token compute latency, to make streaming visible
        yield f"data: {next_token}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    return StreamingResponse(token_stream(req.prompt, req.max_new_tokens), media_type="text/event-stream")


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


threading.Thread(target=run_server, daemon=True).start()
time.sleep(1.0)

start = time.perf_counter()
response = requests.post(
    f"http://127.0.0.1:{PORT}/generate", json={"prompt": [1, 2, 3], "max_new_tokens": 8}, stream=True,
)
print("tokens as they arrive:")
for line in response.iter_lines(decode_unicode=True):
    if line and line.startswith("data: "):
        elapsed = time.perf_counter() - start
        print(f"  [{elapsed:.2f}s] {line[len('data: '):]}")

print("\ncompare this arrival pattern to step 1, where nothing prints until the")
print("entire response is ready -- streaming trades 'one big wait' for 'many")
print("small ones', which is what makes a chat UI feel responsive token by token.")
