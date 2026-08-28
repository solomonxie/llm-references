# hello-agent

Goal: the tool-calling/ReAct agent loop, raw -- against a local [Ollama](https://ollama.com)
model over its plain HTTP API, no framework, no API key or cost. `hello-langchain` builds the
same loop through LangChain's abstractions; this is what those abstractions are doing underneath.
Steps 8-11 pick the loop back up through the vendor agent SDKs (Claude Agent SDK, OpenAI Agents
SDK, Google ADK) to see what each one buys you over the raw version in steps 1-7.

Each file is a complete, standalone, runnable script.

## Setup

```sh
ollama serve &
ollama pull llama3.2:3b     # steps 1-3
ollama pull qwen2.5:7b      # steps 5-7 -- needs more reliable multi-step tool reasoning

# from the repo root
python3 -m venv venv && venv/bin/pip install -r hello-agent/requirements.txt
venv/bin/python hello-agent/01_plain_chat_loop.py

export ANTHROPIC_API_KEY=...   # step 8
export OPENAI_API_KEY=...      # step 9
export GOOGLE_API_KEY=...      # step 10 (a Gemini API key)
```

## Notes

- Step 4 also needs `qwen2.5:7b` or another tool-calling-capable model for reliable native
  `tool_calls` output (`llama3.2:3b` supports the field but is less consistent about using it).
- See `hello-langchain/08_agent_loop.py`'s note on small-model tool-call failure modes -- the
  same class of problem (a model requesting a malformed or wrong-named call) motivates step 6 here.
- Steps 8-10 need their respective vendor API key set; step 11 runs all three back to back and
  skips any vendor whose key isn't set, rather than requiring all three.
- What differs across vendors (steps 8-10): how a tool is declared (an in-process MCP server +
  `tool()` decorator for Claude, a `@function_tool`-decorated plain function for OpenAI, a plain
  function with a typed dict return for ADK), and what drives the loop (`query()` as an async
  generator, `Runner.run_sync`, or an explicit `Runner` + `SessionService` pair). ADK is the only
  one of the three that makes session state an explicit, swappable dependency rather than
  something implicit in the call.
