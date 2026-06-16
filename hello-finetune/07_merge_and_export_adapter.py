# $ venv/bin/python hello-finetune/07_merge_and_export_adapter.py
#
# Goal: two different ways to ship a LoRA-tuned model. (1) Merge: fold
# B @ A * scaling directly into the frozen base weights, producing one
# ordinary model with no adapter machinery left -- simplest to deploy, but
# back to a full-size model per fine-tune. (2) Keep separate: ship only the
# small adapter file and load it onto the (already-deployed) base model at
# request time -- many fine-tunes can share one base this way.
# Step 7: Merging a LoRA adapter into the base vs. keeping it separate

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
torch.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM")
model = get_peft_model(base_model, lora_config)

FACTS = [
    ("Q: What is the capital of Zorblax? A:", " Glimmerhold"),
    ("Q: What color is the Zorblaxian sky? A:", " violet"),
    ("Q: Who rules Zorblax? A:", " Queen Ashvara"),
]

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
model.train()
for _ in range(60):
    for prompt, completion in FACTS:
        text = prompt + completion + tokenizer.eos_token
        inputs = tokenizer(text, return_tensors="pt")
        loss = model(**inputs, labels=inputs["input_ids"]).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
model.eval()


def generate(m, prompt, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


print("trained PEFT model:")
for prompt, correct in FACTS:
    print(f"  {prompt} -> {generate(model, prompt)!r}  (correct: {correct!r})")

# Path 1: save just the adapter (a few hundred KB, not the whole model).
model.save_pretrained("lora_adapter")
print("\nsaved adapter-only weights to lora_adapter/")

# Path 2: merge LoRA into the base -- collapses A/B/scaling into ordinary
# weights, so the result is a plain model with no PEFT wrapper needed to use it.
merged = model.merge_and_unload()
merged.save_pretrained("merged_model")
print("saved merged, adapter-free model to merged_model/")
print(f"merged model output matches trained model: "
      f"{generate(merged, FACTS[0][0]) == generate(model, FACTS[0][0])}")

# Reloading path 1: fresh base + the saved adapter, loaded back on demand --
# this is how one base model can serve many different LoRA fine-tunes.
fresh_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
reloaded = PeftModel.from_pretrained(fresh_base, "lora_adapter")
print(f"\nreloaded adapter reproduces training output: "
      f"{generate(reloaded, FACTS[0][0]) == generate(model, FACTS[0][0])}")
