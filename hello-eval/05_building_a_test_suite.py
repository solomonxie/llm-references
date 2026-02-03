# $ venv/bin/python 05_building_a_test_suite.py
#
# Goal: steps 1-4 hard-coded a handful of cases inline. A real eval suite
# lives in a data file (YAML/JSON) separate from the harness code, so
# adding a test case is a data edit, not a code change -- and the same
# harness can run against any suite file. This writes a small suite to
# disk, then loads and runs it, mixing the exact-match (step 1) and
# rubric (step 2) checks by case type.
# Step 5: A YAML test suite + a harness that runs it and aggregates results

import re
from pathlib import Path

import yaml

SUITE_PATH = Path("eval_suite.yaml")

SUITE_YAML = """
- id: capital-france
  type: exact_match
  question: What is the capital of France?
  reference: Paris
  prediction: Paris

- id: capital-japan
  type: exact_match
  question: What is the capital of Japan?
  reference: Tokyo
  prediction: Kyoto

- id: square-function
  type: rubric
  question: Write a Python function that returns the square of a number.
  prediction: |
    def square(x):
        return x * x
  rubric:
    - has_def
    - has_return
    - no_print
"""
SUITE_PATH.write_text(SUITE_YAML.strip() + "\n")

RUBRIC_CHECKS = {
    "has_def": lambda a: bool(re.search(r"\bdef\s+\w+\(", a)),
    "has_return": lambda a: "return" in a,
    "no_print": lambda a: "print(" not in a,
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def run_case(case: dict) -> tuple[bool, str]:
    if case["type"] == "exact_match":
        passed = normalize(case["prediction"]) == normalize(case["reference"])
        return passed, "exact match" if passed else f"expected {case['reference']!r}, got {case['prediction']!r}"
    elif case["type"] == "rubric":
        checks = [RUBRIC_CHECKS[name](case["prediction"]) for name in case["rubric"]]
        passed = all(checks)
        detail = ", ".join(f"{n}={'ok' if c else 'FAIL'}" for n, c in zip(case["rubric"], checks))
        return passed, detail
    raise ValueError(f"unknown case type: {case['type']}")


suite = yaml.safe_load(SUITE_PATH.read_text())

results = []
for case in suite:
    passed, detail = run_case(case)
    results.append({"id": case["id"], "passed": passed})
    print(f"[{'PASS' if passed else 'FAIL'}] {case['id']:20s} {detail}")

pass_rate = sum(r["passed"] for r in results) / len(results)
print(f"\n{sum(r['passed'] for r in results)}/{len(results)} passed ({pass_rate:.0%})")
