# $ venv/bin/python hello-eval/04_pairwise_comparison_eval.py
#
# Goal: absolute 1-5 scoring (step 3) is notoriously inconsistent between
# runs -- judges anchor differently depending on what they've seen. Pairwise
# comparison ("which of these two answers is better?") is an easier, more
# reliable judgment for a model to make, and is what most production model
# comparisons (and RLHF preference data, see hello-rlhf) actually use.
# Step 4: A/B pairwise comparison, aggregated into a win rate

import json
import random

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "qwen2.5:7b"

COMPARE_PROMPT = """Question: {question}

Response A: {response_a}

Response B: {response_b}

Which response better answers the question? Reply with ONLY this JSON:
{{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""


def compare(question: str, response_a: str, response_b: str) -> dict:
    prompt = COMPARE_PROMPT.format(question=question, response_a=response_a, response_b=response_b)
    response = requests.post(
        OLLAMA_URL,
        json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False},
    )
    try:
        return json.loads(response.json()["message"]["content"])
    except json.JSONDecodeError:
        return {"winner": "tie", "reason": "unparseable judge output"}


# Two systems being compared -- "model_a" is deliberately terser/vaguer.
cases = [
    {
        "question": "How do I reverse a list in Python?",
        "model_a": "Use reversed().",
        "model_b": "Use `my_list[::-1]` for a reversed copy, or `my_list.reverse()` to reverse it in place.",
    },
    {
        "question": "What's the time complexity of binary search?",
        "model_a": "O(log n), since each step halves the remaining search space.",
        "model_b": "It's fast.",
    },
    {
        "question": "What does 'idempotent' mean in an API?",
        "model_a": "Calling it multiple times has the same effect as calling it once.",
        "model_b": "An operation that produces the same result no matter how many times you repeat it.",
    },
]

random.seed(0)
wins = {"model_a": 0, "model_b": 0, "tie": 0}

for case in cases:
    # Randomize A/B order per case to avoid position bias skewing results --
    # a real harness should always do this, not just trust judge symmetry.
    swap = random.random() < 0.5
    resp_a, resp_b = (case["model_b"], case["model_a"]) if swap else (case["model_a"], case["model_b"])
    result = compare(case["question"], resp_a, resp_b)

    winner_label = {"A": "model_b" if swap else "model_a",
                    "B": "model_a" if swap else "model_b",
                    "tie": "tie"}.get(result.get("winner"), "tie")
    wins[winner_label] += 1
    print(f"{case['question']}\n  winner: {winner_label} ({result.get('reason')})\n")

total = len(cases)
print(f"model_a win rate: {wins['model_a'] / total:.0%}")
print(f"model_b win rate: {wins['model_b'] / total:.0%}")
print(f"ties: {wins['tie'] / total:.0%}")
