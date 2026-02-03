# hello-agent

Goal: the tool-calling/ReAct agent loop, raw -- against a local [Ollama](https://ollama.com)
model over its plain HTTP API, no framework, no API key or cost. `hello-langchain` builds the
same loop through LangChain's abstractions; this is what those abstractions are doing underneath.

Each file is a complete, standalone, runnable script.

| File | Demonstrates |
|---|---|
| `01_plain_chat_loop.py` | Multi-turn chat, raw HTTP, no tools -- the baseline every agent loop extends |
| `02_function_schema_and_manual_dispatch.py` | Hand-rolled tool schema in the prompt, manually parsing/dispatching a JSON tool call |
| `03_react_prompt_from_scratch.py` | The ReAct Thought/Action/Observation loop, via plain-text prompting |
| `04_native_tool_calling_api.py` | The same tool, via Ollama's native `tools` field and `tool_calls`/`tool` message role |
| `05_multi_tool_agent_loop.py` | Looping native tool calls -- multiple tools, until the model gives a final answer |
| `06_error_handling_and_retries.py` | Malformed calls, unknown tools, exceptions -- fed back as observations, with a failure budget |
| `07_memory_and_multi_turn.py` | Persisting conversation + tool history across separate top-level questions |

## Setup

```sh
ollama serve &
ollama pull llama3.2:3b     # steps 1-3
ollama pull qwen2.5:7b      # steps 5-7 -- needs more reliable multi-step tool reasoning

python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_plain_chat_loop.py
```

## Notes

- Step 4 also needs `qwen2.5:7b` or another tool-calling-capable model for reliable native
  `tool_calls` output (`llama3.2:3b` supports the field but is less consistent about using it).
- See `hello-langchain/08_agent_loop.py`'s note on small-model tool-call failure modes -- the
  same class of problem (a model requesting a malformed or wrong-named call) motivates step 6 here.
