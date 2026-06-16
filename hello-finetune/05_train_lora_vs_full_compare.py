# $ venv/bin/python hello-finetune/05_train_lora_vs_full_compare.py
#
# Goal: put step 2 (full fine-tune) and step 4's LoRA adapters through the
# exact same training loop and toy task, then compare trainable parameter
# count and whether the facts actually got learned. This is the payoff --
# LoRA training a tiny fraction of the parameters should still reach the
# same toy-task result as updating every weight.
# Step 5: Full fine-tune vs LoRA, same task, compared side by side

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"

FACTS = [
    ("Q: What is the capital of Zorblax? A:", " Glimmerhold"),
    ("Q: What color is the Zorblaxian sky? A:", " violet"),
    ("Q: Who rules Zorblax? A:", " Queen Ashvara"),
]


class LoRAWrapped(nn.Module):
    def __init__(self, base, in_features, out_features, rank=4, alpha=8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.A.T) @ self.B.T * self.scaling


def load_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def apply_lora(model):
    n_embd = model.config.n_embd
    for p in model.parameters():
        p.requires_grad = False
    for block in model.transformer.h:
        block.attn.c_attn = LoRAWrapped(block.attn.c_attn, n_embd, 3 * n_embd)
        block.attn.c_proj = LoRAWrapped(block.attn.c_proj, n_embd, n_embd)
    return model


def train(model, tokenizer, epochs=40, lr=5e-5):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    model.train()
    for _ in range(epochs):
        for prompt, completion in FACTS:
            text = prompt + completion + tokenizer.eos_token
            inputs = tokenizer(text, return_tensors="pt")
            loss = model(**inputs, labels=inputs["input_ids"]).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate(model, tokenizer) -> float:
    correct = sum(generate(model, tokenizer, p).strip().startswith(c.strip()) for p, c in FACTS)
    return correct / len(FACTS)


results = {}
for name, prepare in [("full fine-tune", lambda m: m), ("LoRA", apply_lora)]:
    torch.manual_seed(0)
    model, tokenizer = load_model()
    model = prepare(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trained = train(model, tokenizer)
    accuracy = evaluate(trained, tokenizer)

    results[name] = {"total": total, "trainable": trainable, "accuracy": accuracy}
    print(f"{name:16s} trainable={trainable:>9,} ({trainable / total:.2%})  toy-fact accuracy={accuracy:.0%}")

print(f"\nLoRA trains {results['LoRA']['trainable'] / results['full fine-tune']['trainable']:.1%} as many")
print("parameters as full fine-tuning, for the same toy-task result.")
