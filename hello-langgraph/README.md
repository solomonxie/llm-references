# hello-langgraph

Goal: LangGraph's core mechanics — a graph of nodes and edges, with state threaded through — one
piece at a time. Same "hello world -> full thing" build-up as `hello-transformer` and
`hello-langchain`; reads naturally as a follow-on to `hello-langchain` (tools, agent loops) since
steps 4-7 use those same pieces as graph nodes instead of a hand-managed loop.

Each file is a complete, standalone, runnable script.

## Setup

```sh
ollama serve &                        # if not already running
ollama pull llama3.2:3b               # steps 4, 6-7
ollama pull qwen2.5:7b                # step 5 only — see hello-langchain/README

# from the repo root
python3 -m venv venv && venv/bin/pip install -r hello-langgraph/requirements.txt
venv/bin/python hello-langgraph/01_state_graph_basics.py
```

## Notes

- `01`-`03` have no LLM dependency at all — they're pure graph mechanics, worth understanding
  before an LLM call is one of the nodes.
- `create_react_agent` (used by early LangGraph docs/tutorials) is deprecated as of LangGraph
  1.0 in favor of `langchain.agents.create_agent`, used here in `05` — same shape, moved package.
