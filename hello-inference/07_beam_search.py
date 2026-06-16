# $ venv/bin/python hello-inference/07_beam_search.py
#
# Goal: greedy decoding (step 1) picks the single best NEXT token at each
# step, but the best next token doesn't always lead to the best overall
# sequence — a slightly worse token now might open up a much better
# continuation later, and greedy can never see that (no backtracking). Beam
# search hedges: keep the top `num_beams` candidate sequences (by total
# log-probability) at every step, not just one, expanding all of them and
# keeping only the best `num_beams` survivors each round.
# Step 7: Keeping top-k candidate sequences by cumulative log-probability, not just the single best token

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()


@torch.no_grad()
def beam_search(prompt: str, num_beams: int, max_new_tokens: int) -> list[tuple[str, float]]:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Start with `num_beams` IDENTICAL copies of the prompt — they only
    # diverge once step 1 expands each one differently.
    beams = input_ids.repeat(num_beams, 1)  # (num_beams, prompt_len)
    beam_scores = torch.zeros(num_beams)  # cumulative log-prob per beam, starts at log(1) = 0
    # The whole zeros-initialized set of beams is identical the first time
    # through — force only ONE of them to expand on step 0 (as if the other
    # num_beams-1 didn't exist yet), or every beam would just pick the same
    # single best next token again, wasting num_beams-1 slots on duplicates.
    beam_scores[1:] = float("-inf")

    for _ in range(max_new_tokens):
        logits = model(beams).logits[:, -1, :]  # (num_beams, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Each beam's total score if extended by each possible next token —
        # broadcasting beam_scores (num_beams,) against every vocab entry.
        candidate_scores = beam_scores.unsqueeze(1) + log_probs  # (num_beams, vocab_size)

        # Flatten to pick the global top `num_beams` across ALL beams' ALL
        # candidate tokens at once — a beam can win 0, 1, or multiple of the
        # new slots; a poor beam can be dropped entirely.
        vocab_size = log_probs.shape[-1]
        top_scores, top_indices = candidate_scores.view(-1).topk(num_beams)
        beam_indices = top_indices // vocab_size  # which beam each winner came from
        token_indices = top_indices % vocab_size  # which token it picked

        beams = torch.cat([beams[beam_indices], token_indices.unsqueeze(1)], dim=1)
        beam_scores = top_scores

    return [(tokenizer.decode(beams[i]), beam_scores[i].item()) for i in range(num_beams)]


def greedy_score(prompt: str, max_new_tokens: int) -> tuple[str, float]:
    """Greedy is beam search with num_beams=1 — same function, smaller beam."""
    return beam_search(prompt, num_beams=1, max_new_tokens=max_new_tokens)[0]


prompt = "The scientists discovered that"

greedy_text, greedy_score_ = greedy_score(prompt, max_new_tokens=15)
print(f"greedy (num_beams=1):\n  {greedy_text!r}  (log-prob: {greedy_score_:.2f})")

print("\nbeam search (num_beams=4), all 4 final beams, best first:")
for text, score in beam_search(prompt, num_beams=4, max_new_tokens=15):
    print(f"  {score:.2f}  {text!r}")

# The top beam's total log-probability should be >= greedy's — beam search
# explores strictly more of the search space, so it can only do as well or
# better BY THIS METRIC (it optimizes total sequence log-probability, not
# any notion of "quality" — often correlated, not the same thing; very high
# num_beams is also known to produce duller, more generic text in practice).
best_beam_text, best_beam_score = beam_search(prompt, num_beams=4, max_new_tokens=15)[0]
print(f"\nbest beam score ({best_beam_score:.2f}) >= greedy score ({greedy_score_:.2f}): {best_beam_score >= greedy_score_}")
