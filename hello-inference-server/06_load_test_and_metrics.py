# $ venv/bin/python 06_load_test_and_metrics.py
#
# Goal: the numbers that actually matter when evaluating an inference
# server -- not just "does it work", but throughput under concurrent load
# and the latency DISTRIBUTION (not just the average, which hides how bad
# the worst requests get). Runs step 5's continuous-batching server and
# hits it with concurrent load, reporting p50/p95/p99 latency and overall
# throughput.
# Step 6: A load-test client + latency percentiles / throughput metrics

import statistics
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

VOCAB_SIZE, D_MODEL, PORT = 20, 32, 8006
MAX_BATCH = 4
TICK_S = 0.02
PAD = 0


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


class ActiveSeq:
    def __init__(self, tokens, max_new_tokens):
        self.tokens = list(tokens)
        self.max_new_tokens = max_new_tokens
        self.n_generated = 0
        self.done_event = threading.Event()
        self.result: list[int] = []


queue_lock = threading.Lock()
incoming: list[ActiveSeq] = []
running: list[ActiveSeq] = []


@torch.no_grad()
def scheduler_step():
    with queue_lock:
        while incoming and len(running) < MAX_BATCH:
            running.append(incoming.pop(0))
    if not running:
        return
    max_len = max(len(s.tokens) for s in running)
    padded = torch.tensor([[PAD] * (max_len - len(s.tokens)) + s.tokens for s in running])
    logits = model(padded)
    next_tokens = logits[:, -1].argmax(dim=-1)
    finished = []
    for i, seq in enumerate(running):
        seq.tokens.append(int(next_tokens[i]))
        seq.n_generated += 1
        if seq.n_generated >= seq.max_new_tokens:
            seq.result = seq.tokens
            seq.done_event.set()
            finished.append(seq)
    for seq in finished:
        running.remove(seq)


def scheduler_loop():
    while True:
        time.sleep(TICK_S)
        scheduler_step()


class GenerateRequest(BaseModel):
    prompt: list[int]
    max_new_tokens: int = 10


@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    seq = ActiveSeq(req.prompt, req.max_new_tokens)
    with queue_lock:
        incoming.append(seq)
    seq.done_event.wait()
    return {"generated": seq.result}


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


threading.Thread(target=run_server, daemon=True).start()
threading.Thread(target=scheduler_loop, daemon=True).start()
time.sleep(1.0)

url = f"http://127.0.0.1:{PORT}/generate"


def one_request() -> tuple[float, int]:
    start = time.perf_counter()
    r = requests.post(url, json={"prompt": [1, 2, 3], "max_new_tokens": 20})
    elapsed = time.perf_counter() - start
    return elapsed, len(r.json()["generated"])


N_REQUESTS, CONCURRENCY = 40, 10
print(f"load test: {N_REQUESTS} requests, {CONCURRENCY} concurrent workers")

overall_start = time.perf_counter()
with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
    results = list(pool.map(lambda _: one_request(), range(N_REQUESTS)))
overall_elapsed = time.perf_counter() - overall_start

latencies = sorted(r[0] for r in results)
total_tokens = sum(r[1] for r in results)


def percentile(data: list[float], p: float) -> float:
    idx = min(int(len(data) * p), len(data) - 1)
    return data[idx]


print(f"\ntotal wall time:     {overall_elapsed:.3f}s")
print(f"throughput:          {N_REQUESTS / overall_elapsed:.1f} requests/sec")
print(f"token throughput:    {total_tokens / overall_elapsed:.1f} tokens/sec")
print(f"latency p50:         {percentile(latencies, 0.50):.3f}s")
print(f"latency p95:         {percentile(latencies, 0.95):.3f}s")
print(f"latency p99:         {percentile(latencies, 0.99):.3f}s")
print(f"latency mean/stdev:  {statistics.mean(latencies):.3f}s / {statistics.stdev(latencies):.3f}s")
