# $ venv/bin/python hello-quantization/05_int4_quantization_from_scratch.py
#
# Goal: push step 3's approach further -- 4 bits per weight instead of 8,
# range [-7, 7] instead of [-127, 127]. Coarser range means more rounding
# error per weight, visibly so. This step also does the packing int8's
# `torch.int8` dtype did for free: two 4-bit values genuinely fit in one
# byte, so real int4 storage packs them together with bit shifts rather
# than "wasting" a whole byte per 4-bit value.
# Step 5: int4 quantization, with real 2-values-per-byte packing

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"


def quantize_int4(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    max_abs = weight.abs().max().item()
    scale = max_abs / 7.0  # 4-bit signed range: [-7, 7] (leaving -8 unused for symmetry)
    quantized = torch.round(weight / scale).clamp(-7, 7).to(torch.int8)
    return quantized, scale


def pack_int4(quantized: torch.Tensor) -> torch.Tensor:
    # Two 4-bit signed values (range -7..7, stored as 0..15 two's-complement-
    # style via + 8 offset) packed into each uint8 byte -- low nibble first.
    flat = (quantized.flatten() + 8).to(torch.uint8)  # shift to unsigned 0..15
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
    low, high = flat[0::2], flat[1::2]
    return (low | (high << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    interleaved = torch.stack([low, high], dim=1).flatten()[:numel]
    return interleaved.to(torch.int8) - 8  # shift back to signed -7..7


weight = torch.randn(64, 64) * 0.3
q4, scale = quantize_int4(weight)
packed = pack_int4(q4)
unpacked = unpack_int4(packed, weight.numel()).reshape(weight.shape)

print(f"original size:  {weight.numel() * weight.element_size()} bytes (fp32)")
print(f"int8 would be:  {weight.numel()} bytes")
print(f"packed int4:    {packed.numel()} bytes (2 values/byte) + 1 float32 scale")
print(f"round-trip pack/unpack lossless: {torch.equal(q4, unpacked)}")

dequantized = unpacked.float() * scale
error4 = (weight - dequantized).abs()
print(f"\nint4 mean absolute error: {error4.mean().item():.6f}")

# Compare against int8's error on the identical tensor for scale.
max_abs = weight.abs().max().item()
q8 = torch.round(weight / (max_abs / 127.0)).clamp(-127, 127)
error8 = (weight - q8 * (max_abs / 127.0)).abs()
print(f"int8 mean absolute error: {error8.mean().item():.6f}  "
      f"(int4 has {error4.mean().item() / error8.mean().item():.1f}x more error)")

# Whole-model effect, same fake-quantize-in-place approach as step 4.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
eval_text = "The quick brown fox jumps over the lazy dog and runs into the forest."


def perplexity(model) -> float:
    inputs = tokenizer(eval_text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
fp32_ppl = perplexity(model)

with torch.no_grad():
    for _, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.dim() == 2:
            w = module.weight.data
            q, s = quantize_int4(w)
            module.weight.data = q.float() * s

int4_ppl = perplexity(model)
print(f"\nperplexity fp32:  {fp32_ppl:.3f}")
print(f"perplexity int4:  {int4_ppl:.3f}  ({(int4_ppl - fp32_ppl) / fp32_ppl:+.2%})")
