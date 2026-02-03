# $ venv/bin/python 06_calibration_and_activation_quant.py
#
# Goal: steps 3-5 only quantized weights -- but activations (the values
# flowing *through* the network) matter too, and their range isn't known
# in advance the way a weight's range is (a weight tensor's min/max is
# just sitting there; an activation's range depends on whatever input the
# model happens to see). Two fixes: "dynamic" quantization computes the
# scale fresh from each batch's actual values; "static" quantization
# calibrates a fixed scale in advance from a representative sample of
# data, then reuses it for every future input -- cheaper at inference
# time, but only as good as the calibration sample was representative.
# Step 6: Calibrating activation ranges, static vs. dynamic quantization

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# The layer whose INPUT activations we'll quantize -- the first MLP block's
# expansion layer, a reasonable stand-in for "some linear layer deep in the
# network" the way a real quantization pass would treat every layer.
target_layer = model.transformer.h[0].mlp.c_fc


def quantize_dequantize(x: torch.Tensor, scale: float) -> torch.Tensor:
    quantized = torch.round(x / scale).clamp(-127, 127)
    return quantized * scale


# --- calibration: collect activation statistics from representative data ---
calibration_texts = [
    "The weather today is quite pleasant.",
    "Machine learning models require large amounts of data.",
    "She walked slowly through the old library.",
    "The stock market fluctuated wildly this week.",
]

observed_max = 0.0


def calibration_hook(module, inputs, output):
    global observed_max
    observed_max = max(observed_max, inputs[0].abs().max().item())


handle = target_layer.register_forward_hook(calibration_hook)
with torch.no_grad():
    for text in calibration_texts:
        model(**tokenizer(text, return_tensors="pt"))
handle.remove()

static_scale = observed_max / 127.0
print(f"calibrated static scale from {len(calibration_texts)} samples: {static_scale:.5f}")

eval_text = "The quick brown fox jumps over the lazy dog and runs into the forest."


def perplexity_with_activation_quant(scale_fn) -> float:
    def quant_hook(module, inputs):
        x = inputs[0]
        scale = scale_fn(x)
        return (quantize_dequantize(x, scale),)

    handle = target_layer.register_forward_pre_hook(quant_hook)
    try:
        inputs = tokenizer(eval_text, return_tensors="pt")
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        return torch.exp(loss).item()
    finally:
        handle.remove()


baseline_ppl = torch.exp(model(**tokenizer(eval_text, return_tensors="pt"),
                                labels=tokenizer(eval_text, return_tensors="pt")["input_ids"]).loss).item()
static_ppl = perplexity_with_activation_quant(lambda x: static_scale)
dynamic_ppl = perplexity_with_activation_quant(lambda x: x.abs().max().item() / 127.0)

print(f"\nperplexity, no activation quant:     {baseline_ppl:.3f}")
print(f"perplexity, static (calibrated):     {static_ppl:.3f}")
print(f"perplexity, dynamic (per-batch):     {dynamic_ppl:.3f}")
print("\ndynamic recomputes the exact scale for every input -- best accuracy, extra")
print("runtime cost. Static reuses one precomputed scale -- cheaper, but wrong if")
print("the eval input's activation range doesn't resemble the calibration sample's.")
