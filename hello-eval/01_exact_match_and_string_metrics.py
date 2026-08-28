# $ venv/bin/python 01_exact_match_and_string_metrics.py
#
# Goal: the cheapest possible way to grade an LLM's output against a
# reference answer -- string comparison. Exact match is strict (right
# answer, wrong casing/whitespace still fails); token-F1 gives partial
# credit for overlapping words, which matters once answers are full
# sentences instead of single facts.
# Step 1: Exact match and token-F1 scoring on toy QA pairs

import re

qa_pairs = [
    {"question": "What is the capital of France?", "reference": "Paris", "prediction": "Paris"},
    {"question": "What is the capital of Japan?", "reference": "Tokyo", "prediction": "The capital of Japan is Tokyo."},
    {"question": "Who wrote Hamlet?", "reference": "William Shakespeare", "prediction": "Shakespeare"},
    {"question": "What is 2 + 2?", "reference": "4", "prediction": "The answer is 4."},
]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # strip punctuation
    return re.sub(r"\s+", " ", text)


def exact_match(prediction: str, reference: str) -> bool:
    return normalize(prediction) == normalize(reference)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    overlap = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


print(f"{'question':35s} {'EM':5s} {'F1':5s}")
em_total, f1_total = 0, 0.0
for pair in qa_pairs:
    em = exact_match(pair["prediction"], pair["reference"])
    f1 = token_f1(pair["prediction"], pair["reference"])
    em_total += em
    f1_total += f1
    print(f"{pair['question']:35s} {str(em):5s} {f1:.2f}")

print(f"\naggregate: EM={em_total / len(qa_pairs):.0%}  F1={f1_total / len(qa_pairs):.2f}")
print("\nrows 2 and 3 are both clearly correct answers, phrased differently from")
print("the reference -- EM scores both 0, F1 gives partial credit for the")
print("overlapping words. Neither metric understands meaning, though -- a")
print("prediction with zero word overlap but the right meaning still scores 0.")
