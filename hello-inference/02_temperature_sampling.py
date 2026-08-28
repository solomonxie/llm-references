# $ venv/bin/python 02_temperature_sampling.py
#
# Goal: fix greedy decoding's biggest flaw — always picking the single
# highest-probability token makes output deterministic and often bland/
# repetitive ("the cat sat on the mat and the cat sat on the mat...").
# Sampling instead draws randomly from the probability distribution, and
# *temperature* controls how sharply peaked that distribution is before
# sampling: divide logits by T before softmax.
#   - T < 1 sharpens the distribution (more confident, closer to greedy)
#   - T = 1 uses the model's own probabilities, unmodified
#   - T > 1 flattens it (more random, more likely to pick a low-probability token)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

prompt = "The best way to learn a new programming language is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids


def sample_with_temperature(input_ids: torch.Tensor, max_new_tokens: int, temperature: float) -> torch.Tensor:
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated).logits[:, -1, :]
            scaled = logits / temperature  # dividing BEFORE softmax — this is the entire mechanism
            probs = torch.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # draw one token, weighted by probs
            generated = torch.cat([generated, next_token], dim=1)
    return generated


def greedy_decode(input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated).logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    return generated


print(f"prompt: {prompt!r}\n")

greedy = greedy_decode(input_ids, max_new_tokens=20)
print(f"greedy (T=0, deterministic): {tokenizer.decode(greedy[0])!r}")

for temperature in [0.3, 1.0, 1.5]:
    torch.manual_seed(0)  # same seed across temperatures — differences below are the temperature's effect, not luck
    result = sample_with_temperature(input_ids, max_new_tokens=20, temperature=temperature)
    print(f"\nT={temperature}: {tokenizer.decode(result[0])!r}")

# Same T=1.0, different random draws — sampling means the same prompt no
# longer gives the same output every time, unlike greedy.
print("\nT=1.0, three runs with different randomness:")
for seed in [1, 2, 3]:
    torch.manual_seed(seed)
    result = sample_with_temperature(input_ids, max_new_tokens=15, temperature=1.0)
    print(f"  seed={seed}: {tokenizer.decode(result[0])!r}")
