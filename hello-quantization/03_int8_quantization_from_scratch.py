# $ venv/bin/python 03_int8_quantization_from_scratch.py
#
# Goal: int8 quantization, in isolation, on one weight matrix. The idea:
# find a scale that maps the weight's value range onto the 8-bit signed
# integer range [-127, 127], round every weight to the nearest representable
# integer, and store that (plus the single scale float) instead of the
# original floats. Dequantizing (int * scale) never perfectly recovers the
# original values -- the gap between them is the quantization error this
# whole series is about measuring.
# Step 3: Symmetric per-tensor int8 quantization, by hand, with error

import torch

torch.manual_seed(0)


def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    # Symmetric: the same scale maps both positive and negative values,
    # anchored on the single largest-magnitude weight in the tensor.
    max_abs = weight.abs().max().item()
    scale = max_abs / 127.0
    quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def dequantize_int8(quantized: torch.Tensor, scale: float) -> torch.Tensor:
    return quantized.to(torch.float32) * scale


weight = torch.randn(64, 64) * 0.3  # a stand-in for one real linear layer's weight matrix

quantized, scale = quantize_int8(weight)
dequantized = dequantize_int8(quantized, scale)

print(f"original dtype/size:   {weight.dtype}, {weight.numel() * weight.element_size()} bytes")
print(f"quantized dtype/size:  {quantized.dtype}, {quantized.numel() * quantized.element_size()} bytes "
      f"+ 1 float32 scale ({scale:.6f})")
print(f"compression: {weight.element_size() / quantized.element_size():.0f}x smaller per weight")

error = (weight - dequantized).abs()
print(f"\nmax absolute error:  {error.max().item():.6f}")
print(f"mean absolute error: {error.mean().item():.6f}")
print(f"relative error (mean/scale): {(error.mean() / abs(weight).mean()).item():.2%}")

print("\nsample values:")
for i in range(5):
    print(f"  original={weight[0, i].item():+.4f}  int8={quantized[0, i].item():+4d}  "
          f"dequantized={dequantized[0, i].item():+.4f}")
