# $ venv/bin/python hello-quantization/02_fp16_bf16_cast.py
#
# Goal: the cheapest possible "quantization" -- just cast every weight to a
# narrower float format. fp16 (5 exponent bits, 10 mantissa bits) and bf16
# (8 exponent bits, 7 mantissa bits -- same exponent range as fp32, less
# precision) both halve memory versus fp32 with a single `.to(dtype)` call
# and (usually) barely move quality, since floats already concentrate
# precision where values are small, which is where most weights sit.
# Step 2: Casting to fp16/bf16 -- memory and quality vs. the fp32 baseline

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

eval_text = "The quick brown fox jumps over the lazy dog and runs into the forest."


def memory_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def perplexity(model) -> float:
    inputs = tokenizer(eval_text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()


print(f"{'dtype':10s} {'memory (MB)':12s} {'perplexity':10s}")
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.eval()
    # fp16/bf16 matmuls fall back to fp32 accumulation-unsupported paths on
    # plain CPU in some torch builds -- perplexity is still computed for
    # comparison, but real speedups from narrower floats show up on GPU/MPS.
    ppl = perplexity(model)
    print(f"{str(dtype):10s} {memory_mb(model):<12.1f} {ppl:<10.3f}")

print("\nfp16/bf16 halve memory vs fp32 for effectively free, quality-wise, on a")
print("model this small -- the real cost of going narrower shows up starting at")
print("int8/int4 (steps 3+), where the format is no longer a float at all.")
