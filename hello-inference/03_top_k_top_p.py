# $ venv/bin/python 03_top_k_top_p.py
#
# Goal: temperature alone (step 2) still leaves the *entire* vocabulary
# eligible to be sampled — at high temperature, even a token with tiny
# probability can occasionally get picked, producing outright gibberish.
# Top-k and top-p (nucleus) sampling both restrict WHICH tokens are eligible
# before sampling among them:
#   - top-k: keep only the k highest-probability tokens, zero out the rest
#   - top-p: keep the smallest set of tokens whose probabilities sum to >= p
#     (adapts to the distribution's shape — a confident distribution keeps
#     few tokens, a flat/uncertain one keeps more)
# Step 3: Top-k and nucleus (top-p) filtering before sampling

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    top_k_values, _ = torch.topk(logits, k)
    threshold = top_k_values[:, -1, None]  # the k-th largest value — the cutoff
    return logits.masked_fill(logits < threshold, float("-inf"))  # softmax(-inf) = 0: excluded from sampling


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Keep tokens up to (and including) the first one that crosses p. Shift
    # the "remove" mask right by one first, so that first crossing token
    # itself is always kept — without the shift, hitting cumulative == p
    # exactly at token i would drop token i too.
    sorted_remove = cumulative_probs > p
    sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
    sorted_remove[:, 0] = False

    remove_mask = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove_mask, float("-inf"))


def sample(input_ids: torch.Tensor, max_new_tokens: int, filter_fn=None, **filter_kwargs) -> torch.Tensor:
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated).logits[:, -1, :]
            if filter_fn is not None:
                logits = filter_fn(logits, **filter_kwargs)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
    return generated


prompt = "The best way to learn a new programming language is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(f"prompt: {prompt!r}\n")

torch.manual_seed(0)
unfiltered = sample(input_ids, max_new_tokens=20)
print(f"unfiltered sampling (full vocab eligible): {tokenizer.decode(unfiltered[0])!r}")

for k in [5, 50]:
    torch.manual_seed(0)
    result = sample(input_ids, max_new_tokens=20, filter_fn=top_k_filter, k=k)
    print(f"\ntop-k={k}: {tokenizer.decode(result[0])!r}")

for p in [0.5, 0.9]:
    torch.manual_seed(0)
    result = sample(input_ids, max_new_tokens=20, filter_fn=top_p_filter, p=p)
    print(f"\ntop-p={p}: {tokenizer.decode(result[0])!r}")

# In practice, both filters above are just `model.generate(..., do_sample=True,
# top_k=50)` or `top_p=0.9` — the library applies the identical filter-then-
# sample logic implemented by hand above.
