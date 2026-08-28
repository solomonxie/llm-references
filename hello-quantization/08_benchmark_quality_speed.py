# $ venv/bin/python 08_benchmark_quality_speed.py
#
# Goal: put every precision from this series in one table -- fp32, fp16,
# bf16, fake-int8, fake-int4 -- with the three numbers that actually
# matter when choosing one: memory footprint, generation latency, and
# quality (perplexity). No single format wins on all three; this is the
# tradeoff surface steps 1-6 built up piece by piece.
# Step 8: A systematic memory/latency/quality benchmark across precisions

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
eval_text = "The quick brown fox jumps over the lazy dog and runs into the forest."
prompt = "The history of computing began with"


def memory_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def perplexity(model) -> float:
    inputs = tokenizer(eval_text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()


def latency_s(model, n_runs: int = 3) -> float:
    inputs = tokenizer(prompt, return_tensors="pt")
    times = []
    for _ in range(n_runs):
        with torch.no_grad():
            start = time.perf_counter()
            model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def fake_quantize_(model, bits: int) -> None:
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for _, module in model.named_modules():
            if hasattr(module, "weight") and module.weight.dim() == 2:
                w = module.weight.data
                max_abs = w.abs().max().item()
                if max_abs == 0:
                    continue
                scale = max_abs / qmax
                module.weight.data = torch.round(w / scale).clamp(-qmax, qmax) * scale


configs = [
    ("fp32", torch.float32, None),
    ("fp16", torch.float16, None),
    ("bf16", torch.bfloat16, None),
    ("fake-int8", torch.float32, 8),
    ("fake-int4", torch.float32, 4),
]

print(f"{'precision':12s} {'memory (MB)':13s} {'latency (s)':13s} {'perplexity':10s}")
for name, dtype, fake_bits in configs:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.eval()
    if fake_bits is not None:
        fake_quantize_(model, fake_bits)

    mem = memory_mb(model)
    ppl = perplexity(model)
    lat = latency_s(model)
    print(f"{name:12s} {mem:<13.1f} {lat:<13.3f} {ppl:<10.3f}")

print("\nnote: fp16/bf16 report the SAME memory-footprint-per-parameter as fake-int8")
print("(2 bytes vs 1 byte would be the real story with true int8 storage -- 'fake'")
print("quantization here stores the rounded values back as float32 to simulate the")
print("rounding error without a real int8 kernel; step 7's bitsandbytes numbers are")
print("the ones that reflect real reduced storage.)")
