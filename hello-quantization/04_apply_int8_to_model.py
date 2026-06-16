# $ venv/bin/python hello-quantization/04_apply_int8_to_model.py
#
# Goal: apply step 3's quantize/dequantize round-trip to every 2D weight
# matrix in a real model, in place, and see how much perplexity actually
# moves. This "fake quantization" (quantize then immediately dequantize
# back to float, rather than running real int8 matmul kernels) isolates
# exactly the effect steps 3-6 care about -- the information lost by
# rounding to 8 bits -- without needing a real int8 compute backend.
# Step 4: Fake-quantizing every linear-like weight matrix, whole-model effect

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"


def quantize_dequantize_int8(weight: torch.Tensor) -> torch.Tensor:
    max_abs = weight.abs().max().item()
    if max_abs == 0:
        return weight
    scale = max_abs / 127.0
    quantized = torch.round(weight / scale).clamp(-127, 127)
    return quantized * scale


def perplexity(model, tokenizer, text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
eval_text = "The quick brown fox jumps over the lazy dog and runs into the forest."

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
fp32_ppl = perplexity(model, tokenizer, eval_text)

quantized_count, total_elements, total_abs_error = 0, 0, 0.0
with torch.no_grad():
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.dim() == 2:
            original = module.weight.data.clone()
            module.weight.data = quantize_dequantize_int8(original)
            total_abs_error += (original - module.weight.data).abs().sum().item()
            total_elements += original.numel()
            quantized_count += 1

int8_ppl = perplexity(model, tokenizer, eval_text)

print(f"quantized {quantized_count} weight matrices, {total_elements:,} total weights")
print(f"mean absolute quantization error: {total_abs_error / total_elements:.6f}")
print(f"\nperplexity fp32:        {fp32_ppl:.3f}")
print(f"perplexity fake-int8:   {int8_ppl:.3f}")
print(f"relative change:        {(int8_ppl - fp32_ppl) / fp32_ppl:+.2%}")
