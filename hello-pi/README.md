# hello-pi

Goal: rebuild [Pi](https://github.com/badlogic/pi-mono) in miniature -- the minimal coding
agent (four tools, a sub-1000-token system prompt, a self-extension system) that powers
OpenClaw. `hello-agent` builds a generic ReAct loop; this one is specifically about Pi's
bet that a strong model plus a tiny, fixed toolset beats a large bespoke framework.

Each file is a complete, standalone, runnable script against the real Claude API (no local
model -- Pi's minimalism assumes a competent model underneath).

## Setup

```sh
export ANTHROPIC_API_KEY=...

python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_minimal_system_prompt.py
```

## Notes

- Steps 2-5 write into a `scratch/` subdirectory here -- that's the point, not a side effect
  to clean up.
- Steps 4-5 take `<session> "<task>"` as arguments and persist history under `.sessions/` --
  run the same session name twice to see it resume.
- Step 5's `write_skill` `exec`s whatever code the model writes with no isolation -- fine for
  a toy agent talking to yourself, not for anything handling untrusted input. See `hello-kern`
  for how to actually sandbox agent-written code before running it.
