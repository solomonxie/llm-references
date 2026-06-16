# $ venv/bin/python hello-finetune/02_full_finetune_toy_task.py
#
# Goal: full fine-tuning -- every one of the model's 82M parameters gets a
# gradient and can move. On a handful of examples this overfits fast (the
# whole point here: prove the facts get learned), but it also means storing
# a full new copy of the model per fine-tune, and touches weights the base
# model needed for everything else it could do.
# Step 2: Full fine-tune (all parameters trainable) on the toy fact set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
torch.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

FACTS = [
    ("Q: What is the capital of Zorblax? A:", " Glimmerhold"),
    ("Q: What color is the Zorblaxian sky? A:", " violet"),
    ("Q: Who rules Zorblax? A:", " Queen Ashvara"),
]

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable parameters: {trainable:,} (100% of the model)\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(40):
    total_loss = 0.0
    for prompt, completion in FACTS:
        # Standard causal LM loss on the full prompt+completion sequence --
        # simplified (not masking the prompt tokens out of the loss) since
        # the point here is the LoRA-vs-full comparison, not label masking.
        text = prompt + completion + tokenizer.eos_token
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0 or epoch == 39:
        print(f"epoch {epoch:2d}  avg loss {total_loss / len(FACTS):.4f}")

model.eval()


def generate(prompt: str, max_new_tokens: int = 8) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


print()
for prompt, correct in FACTS:
    print(f"{prompt}\n  model says: {generate(prompt)!r}  (correct: {correct!r})")
