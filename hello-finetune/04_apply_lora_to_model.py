# $ venv/bin/python 04_apply_lora_to_model.py
#
# Goal: wire step 3's LoRA wrapper into a real model. GPT-2's attention
# projections (`c_attn`, `c_proj`) are `Conv1D` layers, not `nn.Linear` --
# same idea (a learned linear map) but with the weight matrix stored
# transposed relative to `nn.Linear`. Rather than special-case that, this
# wrapper takes in/out feature counts explicitly and treats the base layer
# as any callable -- works identically for either layer type.
# Step 4: Wrapping GPT-2's attention projections with LoRA, base frozen

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
torch.manual_seed(0)


class LoRAWrapped(nn.Module):
    def __init__(self, base: nn.Module, in_features: int, out_features: int, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T) @ self.B.T * self.scaling


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

n_embd = model.config.n_embd  # 768 for distilgpt2

# Freeze the whole model first, then attach trainable LoRA adapters only to
# each block's attention projections -- everything else (embeddings, MLPs,
# layernorms, the final head) stays exactly as pretrained.
for p in model.parameters():
    p.requires_grad = False

for block in model.transformer.h:
    block.attn.c_attn = LoRAWrapped(block.attn.c_attn, in_features=n_embd, out_features=3 * n_embd)
    block.attn.c_proj = LoRAWrapped(block.attn.c_proj, in_features=n_embd, out_features=n_embd)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total parameters:     {total:,}")
print(f"trainable parameters: {trainable:,} ({trainable / total:.2%} of the model)")
print(f"frozen parameters:    {total - trainable:,}")

# Sanity check: output should be identical to the un-adapted model, since
# every LoRA B matrix still starts at zero (step 3's zero-init property).
inputs = tokenizer("The weather today is", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
print(f"\nforward pass still runs with LoRA attached: logits shape {tuple(logits.shape)}")
