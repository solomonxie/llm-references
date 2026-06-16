# $ venv/bin/python hello-finetune/06_quantized_base_qlora.py
#
# Goal: QLoRA -- LoRA (steps 3-5) applied on top of a base model whose
# frozen weights are stored in 4-bit (nf4), not fp32/fp16. The frozen base
# barely needs gradients at all (LoRA is the only trainable part), so
# quantizing it trades a little numerical precision for a much smaller
# memory footprint, with the LoRA adapters still trained in full precision.
# This step uses `peft` + `bitsandbytes` (the real libraries) instead of
# steps 3-4's from-scratch wrapper, now that the mechanism is understood.
# Step 6: QLoRA via peft + bitsandbytes -- 4-bit frozen base, LoRA on top
#
# Requires a CUDA GPU -- bitsandbytes' 4-bit kernels have no CPU or Apple
# Silicon (MPS) backend. On CPU-only/Apple Silicon this raises at
# from_pretrained(); read this file alongside steps 3-5 either way, since
# the LoRA math is identical, just applied via a library instead of by hand.

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "distilgpt2"

if not torch.cuda.is_available():
    raise SystemExit(
        "bitsandbytes 4-bit quantization needs a CUDA GPU (none detected here). "
        "See steps 3-5 for the same LoRA mechanism running anywhere."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # QLoRA's paper-recommended quant type
    bnb_4bit_compute_dtype=torch.bfloat16,  # LoRA math still runs in bf16, only storage is 4-bit
    bnb_4bit_use_double_quant=True,       # quantize the quantization constants too -- extra memory savings
)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")

lora_config = LoraConfig(
    r=4, lora_alpha=8,
    target_modules=["c_attn"],  # same attention projection steps 3-4 targeted by hand
    lora_dropout=0.0,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

FACTS = [
    ("Q: What is the capital of Zorblax? A:", " Glimmerhold"),
    ("Q: What color is the Zorblaxian sky? A:", " violet"),
    ("Q: Who rules Zorblax? A:", " Queen Ashvara"),
]

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
for epoch in range(40):
    for prompt, completion in FACTS:
        text = prompt + completion + tokenizer.eos_token
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        loss = model(**inputs, labels=inputs["input_ids"]).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"epoch {epoch:2d}  loss {loss.item():.4f}")

model.save_pretrained("qlora_adapter")
print("\nsaved adapter (LoRA weights only, base stays 4-bit and untouched) to qlora_adapter/")
