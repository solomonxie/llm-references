# $ venv/bin/python 01_load_small_model.py
#
# Goal: baseline behavior before any fine-tuning. distilgpt2 (82M params,
# a distilled GPT-2) has never seen the made-up facts used throughout this
# series -- a fictional planet with its own capital, sky color, and ruler.
# Every later step measures fine-tuning by whether the model starts getting
# these specific, unlearnable-by-guessing facts right.
# Step 1: Loading a small pretrained model, baseline (wrong) answers

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"  # 82M params, ~350MB download, cached after first run

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# The toy facts every step in this series trains toward -- entirely made up,
# so the base model has no way to know them; any resemblance to a correct
# answer is coincidence, not knowledge.
FACTS = [
    ("Q: What is the capital of Zorblax? A:", " Glimmerhold"),
    ("Q: What color is the Zorblaxian sky? A:", " violet"),
    ("Q: Who rules Zorblax? A:", " Queen Ashvara"),
]


def generate(prompt: str, max_new_tokens: int = 8) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


print(f"loaded {MODEL_NAME}, {sum(p.numel() for p in model.parameters()):,} parameters\n")
for prompt, correct in FACTS:
    completion = generate(prompt)
    print(f"{prompt}\n  model says: {completion!r}")
    print(f"  correct:    {correct!r}\n")

print("the base model has no way to know any of these -- fine-tuning (steps 2+)")
print("is what teaches it these specific facts.")
