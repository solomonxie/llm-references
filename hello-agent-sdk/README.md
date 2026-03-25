# hello-agent-sdk

Goal: the same one-tool task (a `get_weather` tool, asked "what's the weather in Paris?")
through the three major vendor agent SDKs -- Anthropic's Claude Agent SDK, OpenAI's Agents
SDK, and Google's Agent Development Kit -- to see how each one's abstractions differ for
identical work. `hello-agent` builds the loop these SDKs wrap by hand; `hello-langchain` /
`hello-langgraph` cover the leading third-party framework instead of a vendor's own SDK.

Each file is a complete, standalone, runnable script.

## Setup

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt

export ANTHROPIC_API_KEY=...   # step 1
export OPENAI_API_KEY=...      # step 2
export GOOGLE_API_KEY=...      # step 3 (a Gemini API key)

venv/bin/python 01_claude_agent_sdk.py
```

## Notes

- Step 4 runs all three back to back and skips any vendor whose API key isn't set --
  it doesn't need all three to demonstrate the comparison.
- What differs across vendors: how a tool is declared (an in-process MCP server + `tool()`
  decorator for Claude, a `@function_tool`-decorated plain function for OpenAI, a plain
  function with a typed dict return for ADK), and what drives the loop (`query()` as an
  async generator, `Runner.run_sync`, or an explicit `Runner` + `SessionService` pair).
  ADK is the only one of the three that makes session state an explicit, swappable
  dependency rather than something implicit in the call.
