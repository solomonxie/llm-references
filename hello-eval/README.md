# hello-eval

Goal: how to actually grade LLM outputs, from plain string comparison through LLM-as-judge,
pairwise comparison, a data-driven test suite, and tracking pass/fail over time to catch
regressions. Steps 3-4's judge calls run against a local [Ollama](https://ollama.com) model.

Each file is a complete, standalone, runnable script -- later files re-declare code from
earlier ones rather than importing across numbered files.

| File | Demonstrates |
|---|---|
| `01_exact_match_and_string_metrics.py` | Exact match and token-F1 scoring on toy QA pairs |
| `02_rubric_based_scoring.py` | Hand-written rubric checks (regex/keyword presence) for open-ended answers |
| `03_llm_as_judge.py` | An LLM scoring another model's output 1-5, forced into structured JSON |
| `04_pairwise_comparison_eval.py` | A/B pairwise comparison judging, with position-bias-safe randomization, aggregated to a win rate |
| `05_building_a_test_suite.py` | Eval cases in a YAML file + a harness that loads and runs them |
| `06_regression_tracking.py` | Storing eval runs over time and diffing the latest against a baseline |

## Setup

```sh
ollama serve &
ollama pull qwen2.5:7b     # steps 3-4 (judge model)

python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_exact_match_and_string_metrics.py
```

## Notes

- `05` and `06` write small data files (`eval_suite.yaml`, `eval_history.json`) into this
  directory when run -- that's the point (a suite/history that persists between runs), not
  a side effect to clean up.
