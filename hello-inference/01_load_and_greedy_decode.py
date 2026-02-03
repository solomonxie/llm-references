# $ venv/bin/python 01_load_and_greedy_decode.py
#
# Goal: generation, by hand, on a real pretrained model — hello-transformer
# built the architecture from scratch but never trained anything beyond a
# toy task; this uses GPT-2 (124M params, OpenAI, 2019), small enough to run
# on CPU, to show what "generating text" actually does at each step: one
# forward pass produces logits for the NEXT token only, you pick one, append
# it, and repeat. `model.generate()` (used from here on) does exactly this
# loop internally — writing it out once here makes it not a black box.
# Step 1: Loading GPT-2; a by-hand greedy decode loop (argmax, repeat)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"  # the smallest GPT-2 checkpoint — 124M params, ~500MB download, cached after first run

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()  # disables dropout — irrelevant for training, but affects generation quality if left on

prompt = "The best way to learn a new programming language is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(f"prompt: {prompt!r}")
print(f"input_ids: {input_ids.tolist()}  ({input_ids.shape[1]} tokens)")


def greedy_decode_by_hand(input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    generated = input_ids
    with torch.no_grad():  # no backward pass needed — saves memory/time, same as inference always
        for _ in range(max_new_tokens):
            outputs = model(generated)
            # outputs.logits: (batch, seq_len_so_far, vocab_size) — one row of
            # vocab-sized scores per position. Only the LAST position's row
            # matters: that's "what comes after everything so far".
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy: always the single highest-scoring token
            generated = torch.cat([generated, next_token], dim=1)
    return generated


by_hand = greedy_decode_by_hand(input_ids, max_new_tokens=15)
print(f"\nby-hand greedy:   {tokenizer.decode(by_hand[0])!r}")

# In practice you'd just call the library's own loop instead of writing it
# yourself: `model.generate(input_ids, max_new_tokens=15, do_sample=False,
# pad_token_id=tokenizer.eos_token_id)` does exactly the steps above and
# returns the same token ids. Every file from here on keeps using the
# by-hand loop instead, specifically so each new mechanism (temperature,
# top-k/top-p, repetition penalty, caching, batching, beam search) is
# visible as plain tensor ops rather than a flag on a library call.

# The obvious flaw with greedy decoding, on display: it's fully deterministic
# (same prompt -> same output, every time) and tends toward repetitive,
# "safe" continuations — see 02_temperature_sampling.py for the fix.
by_hand_again = greedy_decode_by_hand(input_ids, max_new_tokens=15)
print(f"\nsame prompt, run again: {tokenizer.decode(by_hand_again[0])!r}  (identical — greedy has no randomness)")
