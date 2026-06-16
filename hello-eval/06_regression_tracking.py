# $ venv/bin/python hello-eval/06_regression_tracking.py
#
# Goal: a single eval run tells you today's pass rate; it doesn't tell you
# whether things got *worse*. Regression tracking stores every run's
# per-case results to disk, keyed by timestamp, and diffs the latest run
# against a designated baseline run to flag exactly which cases flipped
# from pass to fail (or vice versa) -- the useful signal after a prompt or
# model change.
# Step 6: Storing eval runs over time and diffing against a baseline

import json
import re
import time
from pathlib import Path

HISTORY_PATH = Path("eval_history.json")

suite = [
    {"id": "capital-france", "reference": "Paris", "prediction": "Paris"},
    {"id": "capital-japan", "reference": "Tokyo", "prediction": "Kyoto"},
    {"id": "capital-italy", "reference": "Rome", "prediction": "Rome"},
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def run_suite(suite: list[dict]) -> dict[str, bool]:
    return {case["id"]: normalize(case["prediction"]) == normalize(case["reference"]) for case in suite}


def load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text())
    return []


def save_run(history: list[dict], results: dict[str, bool]) -> None:
    history.append({"timestamp": time.time(), "results": results})
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


def diff_runs(baseline: dict[str, bool], current: dict[str, bool]) -> list[str]:
    lines = []
    for case_id in current:
        before = baseline.get(case_id)
        after = current[case_id]
        if before is None:
            lines.append(f"  NEW    {case_id}: {'PASS' if after else 'FAIL'}")
        elif before and not after:
            lines.append(f"  REGRESSED  {case_id}: PASS -> FAIL")
        elif not before and after:
            lines.append(f"  FIXED      {case_id}: FAIL -> PASS")
    return lines


history = load_history()

if not history:
    # First run ever -- simulate an earlier baseline where "capital-italy"
    # used to fail (e.g. an older model/prompt version), to give the diff
    # below something to report.
    baseline_results = dict(run_suite(suite))
    baseline_results["capital-italy"] = False
    save_run(history, baseline_results)
    print("no prior history -- recorded a synthetic baseline run")

baseline_run = history[0]["results"]
current_results = run_suite(suite)
save_run(history, current_results)

print(f"\nbaseline (run 1): {baseline_run}")
print(f"current  (run {len(history)}): {current_results}")

diff_lines = diff_runs(baseline_run, current_results)
print(f"\ndiff vs baseline ({len(diff_lines)} changes):")
for line in diff_lines or ["  (no changes)"]:
    print(line)
