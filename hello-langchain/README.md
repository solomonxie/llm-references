# hello-langchain

Goal: LangChain's core abstractions, one at a time — same "hello world -> full thing" build-up as
`hello-transformer`. Runs entirely against a local [Ollama](https://ollama.com) model, no API key
or cost.

Each file is a complete, standalone, runnable script.

| File | Demonstrates |
|---|---|
| `01_llm_call.py` | `ChatOllama.invoke()` — the base chat-model interface every provider implements |
| `02_prompt_template.py` | `ChatPromptTemplate` — reusable prompts with placeholders and message roles |
| `03_lcel_chain.py` | LCEL: piping `prompt \| llm \| output_parser` into one composed `Runnable` |
| `04_structured_output.py` | `.with_structured_output(PydanticModel)` — typed data out, not prose |
| `05_memory_conversation.py` | Multi-turn conversation — history is just a message list you resend |
| `06_tools_function_calling.py` | `@tool` + `.bind_tools()` — the model requests calls, your code executes them |
| `07_rag_retrieval.py` | Embeddings + an in-memory vector store — grounding answers in your own docs |
| `08_agent_loop.py` | The tool-calling loop from `06`, run repeatedly until there's a final answer (ReAct) |

## Setup

```sh
ollama serve &                        # if not already running
ollama pull llama3.2:3b               # chat model used in most steps
ollama pull qwen2.5:7b                # step 8 only — needs more reliable multi-tool-call reasoning
ollama pull nomic-embed-text          # step 7 only — embeddings

python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_llm_call.py
```

## Notes

- `03`-`08` build on `01`/`02` — read them in order.
- Model choice matters for tool-use reliability: `llama3.2:3b` occasionally requests a tool with
  another tool's *name* as its argument instead of waiting for that tool's result (see the
  try/except in `08_agent_loop.py`, and its comment on why that step uses `qwen2.5:7b` instead).
  This is a real, common small-model failure mode, not specific to this example — production
  agent loops need to handle malformed tool-call args regardless of model size.
