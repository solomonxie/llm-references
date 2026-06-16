# $ venv/bin/python hello-inference/04_repetition_penalty.py
#
# Goal: greedy decoding's classic degenerate failure mode, made visible on
# purpose, then fixed. Once a phrase repeats, its tokens are (by definition)
# high-probability continuations again — greedy has no memory of "I already
# said this" and can loop forever. Repetition penalty fixes this directly:
# lower the logit of any token that already appeared in the generated text,
# in proportion to a penalty factor, before picking the next one.
# Step 4: Fixing greedy's repetition-loop failure mode

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

# A prompt/length combo picked specifically because greedy GPT-2 loops on it —
# not every prompt triggers this, but short factual-sounding prompts often do.
prompt = "The weather today is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids


def greedy_decode(input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated).logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    return generated


def greedy_decode_with_repetition_penalty(input_ids: torch.Tensor, max_new_tokens: int, penalty: float) -> torch.Tensor:
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated).logits[:, -1, :]

            # For every token id already in `generated`: a positive logit
            # gets divided by penalty (pushed down), a negative logit gets
            # MULTIPLIED by penalty (also pushed down — dividing a negative
            # number would push it UP toward zero, the wrong direction).
            # This is the exact scheme from Keskar et al. 2019 ("CTRL"),
            # which transformers' own repetition_penalty implements.
            seen = torch.unique(generated[0])
            seen_logits = logits[0, seen]
            logits[0, seen] = torch.where(seen_logits > 0, seen_logits / penalty, seen_logits * penalty)

            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    return generated


print(f"prompt: {prompt!r}\n")

plain = greedy_decode(input_ids, max_new_tokens=40)
print(f"plain greedy (watch it loop):\n  {tokenizer.decode(plain[0])!r}")

penalized = greedy_decode_with_repetition_penalty(input_ids, max_new_tokens=40, penalty=1.3)
print(f"\nwith repetition_penalty=1.3:\n  {tokenizer.decode(penalized[0])!r}")

# In practice this is `model.generate(..., do_sample=False,
# repetition_penalty=1.3)` — the library applies the identical CTRL-style
# penalty implemented by hand above.
