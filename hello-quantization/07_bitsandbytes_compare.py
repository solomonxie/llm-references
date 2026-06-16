# $ venv/bin/python hello-quantization/07_bitsandbytes_compare.py
#
# Goal: compare the from-scratch quantization (steps 3-6) against a real
# production quantization library. `bitsandbytes` does the same conceptual
# thing -- map float weights onto a narrower representation with a scale --
# but with real int8/4-bit compute kernels (not the "fake" quantize-then-
# immediately-dequantize-back-to-float simulation used so far) and more
# refined range-finding (per-block scales, not one scale for the whole
# tensor) for better quality at the same bit width.
# Step 7: Real int8/nf4 quantization via bitsandbytes, vs. the from-scratch version
#
# Requires a CUDA GPU -- bitsandbytes' int8/4-bit kernels have no CPU or
# Apple Silicon (MPS) backend. Read alongside steps 3-6 either way; the
# quantization concepts are identical, just executed by real kernels here.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "distilgpt2"

if not torch.cuda.is_available():
    raise SystemExit(
        "bitsandbytes needs a CUDA GPU (none detected here). "
        "See steps 3-6 for the same quantization concepts running anywhere."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
eval_text = "The quick brown fox jumps over the lazy dog and runs into the forest."


def perplexity(model) -> float:
    inputs = tokenizer(eval_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()


fp32_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda().eval()
fp32_ppl = perplexity(fp32_model)
del fp32_model
torch.cuda.empty_cache()

int8_config = BitsAndBytesConfig(load_in_8bit=True)
int8_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=int8_config, device_map="auto")
int8_ppl = perplexity(int8_model)
del int8_model
torch.cuda.empty_cache()

nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
nf4_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=nf4_config, device_map="auto")
nf4_ppl = perplexity(nf4_model)

print(f"{'method':20s} {'perplexity':10s}")
print(f"{'fp32':20s} {fp32_ppl:<10.3f}")
print(f"{'bitsandbytes int8':20s} {int8_ppl:<10.3f}")
print(f"{'bitsandbytes nf4':20s} {nf4_ppl:<10.3f}")
print("\ncompare these against step 4's fake-int8 and step 5's fake-int4 perplexity")
print("numbers -- bitsandbytes' per-block scales and (for nf4) a non-uniform")
print("codebook tuned for roughly-normal weight distributions usually do better")
print("than this series' single-scale-per-tensor from-scratch version.")
