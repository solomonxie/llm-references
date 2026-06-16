# hello-agent

Goal: the tool-calling/ReAct agent loop, raw -- against a local [Ollama](https://ollama.com)
model over its plain HTTP API, no framework, no API key or cost. `hello-langchain` builds the
same loop through LangChain's abstractions; this is what those abstractions are doing underneath.

Each file is a complete, standalone, runnable script.

## Setup

```sh
ollama serve &
ollama pull llama3.2:3b     # steps 1-3
ollama pull qwen2.5:7b      # steps 5-7 -- needs more reliable multi-step tool reasoning

# from the repo root
python3 -m venv venv && venv/bin/pip install -r hello-agent/requirements.txt
venv/bin/python hello-agent/01_plain_chat_loop.py
```

## Notes

- Step 4 also needs `qwen2.5:7b` or another tool-calling-capable model for reliable native
  `tool_calls` output (`llama3.2:3b` supports the field but is less consistent about using it).
- See `hello-langchain/08_agent_loop.py`'s note on small-model tool-call failure modes -- the
  same class of problem (a model requesting a malformed or wrong-named call) motivates step 6 here.
