# $ venv/bin/python hello-eval/02_rubric_based_scoring.py
#
# Goal: open-ended answers (no single "reference string" exists) need a
# different check -- a rubric of pass/fail criteria written by a human,
# each checking for something the answer must/must-not contain. Still no
# model involved in the grading; it's rule-based, but the rules can be
# richer than plain string equality.
# Step 2: Hand-written rubric checks (regex/keyword presence) for open answers

import re

test_cases = [
    {
        "prompt": "Write a Python function that returns the square of a number.",
        "answer": "def square(x):\n    return x * x",
        "rubric": [
            ("has a def statement", lambda a: bool(re.search(r"\bdef\s+\w+\(", a))),
            ("returns a value", lambda a: "return" in a),
            ("uses multiplication or **2", lambda a: "*" in a),
            ("no print() side effect", lambda a: "print(" not in a),
        ],
    },
    {
        "prompt": "Explain what a for loop does, in one sentence, without code.",
        "answer": "A for loop runs a block of code once for each item in a sequence.",
        "rubric": [
            ("mentions 'loop'", lambda a: "loop" in a.lower()),
            ("mentions repetition (each/every/repeat)", lambda a: re.search(r"\beach\b|\bevery\b|\brepeat", a.lower())),
            ("contains no code block", lambda a: "def " not in a and "```" not in a),
            ("is a single sentence", lambda a: a.strip().count(".") <= 1),
        ],
    },
]

for case in test_cases:
    print(f"prompt: {case['prompt']}")
    print(f"answer: {case['answer']!r}")
    results = [(name, bool(check(case["answer"]))) for name, check in case["rubric"]]
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    score = sum(p for _, p in results) / len(results)
    print(f"  rubric score: {score:.0%}\n")
