# $ venv/bin/python 04_continuous_batching.py
#
# Goal: steps 1-3 handle one request fully before starting the next --
# fine for a demo, but GPUs (and even CPUs, to a lesser extent) do a batch
# of N sequences in barely more time than one sequence, so serving requests
# one at a time wastes most of the hardware's throughput. This collects
# whatever requests arrive within a short time window into one batch and
# runs the decode loop once for all of them together.
#
# Simplification: batched requests here must share the same prompt length
# (no padding/attention-mask handling -- hello-inference/06_batching.py
# already covers that in isolation). The real limitation this step
# demonstrates: a batch only returns once its SLOWEST member finishes, so a
# short request stuck behind a long one still waits for it -- step 5 fixes
# that specific problem.
# Step 4: Batching concurrent requests collected within a short time window

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

torch.manual_seed(0)

VOCAB_SIZE, D_MODEL, PORT = 20, 32, 8004
BATCH_WINDOW_S = 0.05
MAX_BATCH = 8


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


class PendingRequest:
    def __init__(self, prompt: list[int], max_new_tokens: int):
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.event = threading.Event()
        self.result: list[int] = []


pending_lock = threading.Lock()
pending: list[PendingRequest] = []


@torch.no_grad()
def run_batch(batch: list[PendingRequest]) -> None:
    tokens = torch.tensor([r.prompt for r in batch])  # requires equal-length prompts
    results = [list(r.prompt) for r in batch]
    active = [True] * len(batch)
    max_steps = max(r.max_new_tokens for r in batch)

    for _ in range(max_steps):
        logits = model(tokens)  # one batched forward step for every sequence still in the batch
        next_tokens = logits[:, -1].argmax(dim=-1)
        tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)
        for i, req in enumerate(batch):
            if active[i]:
                results[i].append(int(next_tokens[i]))
                if len(results[i]) - len(req.prompt) >= req.max_new_tokens:
                    active[i] = False
        # Note: inactive rows still ride along in `tokens` and get computed
        # every remaining step -- wasted work for whichever request finished
        # first, since this loop can't drop a row out of the batch mid-flight.

    for i, req in enumerate(batch):
        req.result = results[i]
        req.event.set()


def batching_loop():
    while True:
        time.sleep(BATCH_WINDOW_S)
        with pending_lock:
            if not pending:
                continue
            batch, pending[:] = pending[:MAX_BATCH], pending[MAX_BATCH:]
        run_batch(batch)


class GenerateRequest(BaseModel):
    prompt: list[int]
    max_new_tokens: int = 10


@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    pending_req = PendingRequest(req.prompt, req.max_new_tokens)
    with pending_lock:
        pending.append(pending_req)
    pending_req.event.wait()
    return {"generated": pending_req.result}


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


threading.Thread(target=run_server, daemon=True).start()
threading.Thread(target=batching_loop, daemon=True).start()
time.sleep(1.0)

url = f"http://127.0.0.1:{PORT}/generate"
N_REQUESTS = 6


def fire_concurrent(n: int) -> float:
    with ThreadPoolExecutor(max_workers=n) as pool:
        start = time.perf_counter()
        futures = [pool.submit(requests.post, url, json={"prompt": [1, 2, 3], "max_new_tokens": 15}) for _ in range(n)]
        for f in futures:
            f.result()
        return time.perf_counter() - start


def fire_sequential(n: int) -> float:
    start = time.perf_counter()
    for _ in range(n):
        requests.post(url, json={"prompt": [1, 2, 3], "max_new_tokens": 15})
    return time.perf_counter() - start


concurrent_time = fire_concurrent(N_REQUESTS)
time.sleep(0.5)  # let any straggling batch drain before the sequential run
sequential_time = fire_sequential(N_REQUESTS)

print(f"{N_REQUESTS} requests, concurrent (batched together): {concurrent_time:.3f}s")
print(f"{N_REQUESTS} requests, sequential (one batch of 1 each): {sequential_time:.3f}s")
print(f"speedup: {sequential_time / concurrent_time:.1f}x")
