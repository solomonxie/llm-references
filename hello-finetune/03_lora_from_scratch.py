# $ venv/bin/python 03_lora_from_scratch.py
#
# Goal: LoRA (Low-Rank Adaptation) in isolation, no model attached yet. The
# idea: instead of updating a weight matrix W directly (out_features x
# in_features parameters), freeze W and learn a *low-rank* update
# delta_W = B @ A, where A is (rank x in_features) and B is (out_features x
# rank). For rank << in/out_features, A and B together have far fewer
# parameters than W -- and B is initialized to zero, so delta_W starts at
# exactly zero (training starts from the frozen model's exact behavior).
# Step 3: A LoRA-adapted linear layer, implemented from scratch

import torch
import torch.nn as nn

torch.manual_seed(0)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False  # frozen -- LoRA never touches the original weights

        out_features, in_features = base.weight.shape
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))  # zero init -> delta_W starts at 0
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.A.T) @ self.B.T * self.scaling
        return base_out + lora_out


base_layer = nn.Linear(64, 64)
lora_layer = LoRALinear(base_layer, rank=4)

base_params = sum(p.numel() for p in base_layer.parameters())
lora_params = lora_layer.A.numel() + lora_layer.B.numel()
print(f"base layer:  {base_params:,} parameters (frozen)")
print(f"LoRA A + B:  {lora_params:,} parameters (trainable) -- "
      f"{lora_params / base_params:.1%} of the base layer's size")

x = torch.randn(2, 64)
with torch.no_grad():
    base_only = base_layer(x)
    lora_output = lora_layer(x)
print(f"\nbefore any training, LoRA output == base output: "
      f"{torch.allclose(base_only, lora_output)}  (B starts at zero)")

# One training step to show delta_W stops being zero once B moves.
target = torch.randn(2, 64)
optimizer = torch.optim.SGD([lora_layer.A, lora_layer.B], lr=0.1)
loss = ((lora_layer(x) - target) ** 2).mean()
loss.backward()
optimizer.step()

with torch.no_grad():
    after = lora_layer(x)
print(f"after one training step, LoRA output == base output: "
      f"{torch.allclose(base_only, after)}")
print(f"B is no longer all-zero: {not torch.allclose(lora_layer.B, torch.zeros_like(lora_layer.B))}")
