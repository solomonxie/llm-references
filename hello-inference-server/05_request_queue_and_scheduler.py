# $ venv/bin/python hello-inference-server/05_request_queue_and_scheduler.py
#
# Goal: fix step 4's specific problem -- a static batch waits for its
# slowest member before ANY new request can join, and a finished member
# still rides along wasting compute. Continuous batching (what real
# inference servers like vLLM do) instead runs one decode step at a time
# for whatever's currently active, pulling newly arrived requests into the
# running batch and dropping finished ones out, every single step -- so a
# short request queued behind a long one doesn't wait for the long one to
# finish, it just joins on the very next step there's room.
#
# Same padding simplification as step 4: no attention mask over pad
# positions (hello-inference/06_batching.py covers that in isolation).
# Step 5: A scheduler loop with per-step batch membership changes

import itertools
import threading
import time

import requests
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

torch.manual_seed(0)

VOCAB_SIZE, D_MODEL, PORT = 20, 32, 8005
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
seq_id_counter = itertools.count()


class ActiveSeq:
    def __init__(self, tokens: list[int], max_new_tokens: int):
        self.id = next(seq_id_counter)
        self.tokens = list(tokens)
        self.max_new_tokens = max_new_tokens
        self.n_generated = 0
        self.done_event = threading.Event()
        self.result: list[int] = []


queue_lock = threading.Lock()
incoming: list[ActiveSeq] = []
running: list[ActiveSeq] = []


@torch.no_grad()
def scheduler_step() -> None:
    with queue_lock:
        while incoming and len(running) < MAX_BATCH:
            joined = incoming.pop(0)
            running.append(joined)
            print(f"  [join] seq {joined.id} (batch size now {len(running)})")

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
        print(f"  [leave] seq {seq.id} done (batch size now {len(running)})")


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
results = {}


def fire(name: str, prompt: list[int], max_new_tokens: int, delay: float = 0.0):
    def worker():
        time.sleep(delay)
        start = time.perf_counter()
        r = requests.post(url, json={"prompt": prompt, "max_new_tokens": max_new_tokens})
        results[name] = (time.perf_counter() - start, r.json()["generated"])

    return threading.Thread(target=worker)


print("firing 2 long requests immediately, 2 short requests 0.1s later")
print("(the short ones should join the running batch mid-flight, not wait):\n")
threads = [
    fire("long-A", [1, 2, 3], 40),
    fire("long-B", [4, 5, 6], 40),
    fire("short-C", [7, 8], 5, delay=0.1),
    fire("short-D", [9, 1], 5, delay=0.1),
]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("\ncompletion times:")
for name, (elapsed, _) in sorted(results.items(), key=lambda kv: kv[1][0]):
    print(f"  {name:8s} finished in {elapsed:.3f}s")
print("\nthe short requests should finish well before the long ones despite")
print("arriving after them -- they joined the running batch and left again")
print("without waiting for long-A/long-B to complete first.")
