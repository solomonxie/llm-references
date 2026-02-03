# hello-eval

Goal: how to actually grade LLM outputs, from plain string comparison through LLM-as-judge,
pairwise comparison, a data-driven test suite, and tracking pass/fail over time to catch
regressions. Steps 3-4's judge calls run against a local [Ollama](https://ollama.com) model.

Each file is a complete, standalone, runnable script -- later files re-declare code from
earlier ones rather than importing across numbered files.

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
