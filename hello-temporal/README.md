# hello-temporal

Goal: durable workflow orchestration for agent pipelines, via [Temporal](https://temporal.io) --
retries, long-running human-in-the-loop pauses, and crash-resumable multi-step execution,
without hand-rolling any of it. `hello-eval`/`hello-inference-server` cover making individual
calls and serving reliable; this covers making a whole multi-step agent *run* reliable.

Each file is a complete, standalone, runnable script (worker and client both run in-process
here for a runnable demo; in production the worker is normally its own long-lived process).

## Setup

```sh
# Temporal CLI, for the local dev server -- https://docs.temporal.io/cli
brew install temporal   # or see the docs above for other platforms
temporal server start-dev &

export ANTHROPIC_API_KEY=...     # step 2 only

python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_hello_world_workflow.py
```

Open http://localhost:8233 (the dev server's built-in Web UI) to watch workflow executions,
their event history, and pending activities while any step runs.

## Notes

- Step 4's real payoff needs a manual step: start it, Ctrl+C after the first tool's activity
  logs, then rerun. Temporal replays the workflow from its server-side event history, so the
  completed activity is not re-executed -- the new worker resumes at the next step.
- Step 5 uses a 30-second interval instead of a real cron cadence so the demo observes an
  actual firing; swap `ScheduleIntervalSpec` for a `ScheduleSpec(cron_expressions=[...])` for
  real cron syntax.
