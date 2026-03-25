# $ venv/bin/python 01_minimal_system_prompt.py
#
# Goal: [Pi](https://github.com/badlogic/pi-mono), the coding agent behind
# OpenClaw, has the shortest system prompt of any known agent -- its whole
# behavioral spec fits under 1000 tokens, with no elaborate scaffolding.
# This series rebuilds Pi's design in miniature: this step is just the
# prompt, measured with the real tokenizer instead of guessing.
# Step 1: A sub-1000-token system prompt, sent as a plain (tool-less) call

import anthropic

MODEL = "claude-opus-5"

# The whole "spec": be terse, prefer editing over rewriting, use tools
# instead of asking -- everything later steps do follows from this alone.
SYSTEM_PROMPT = """You are a minimal coding agent. Rules:
- Be concise. No preamble, no summary of what you're about to do.
- Prefer editing an existing file over rewriting it.
- If a tool can answer the question, use it instead of guessing.
- Never explain your reasoning unless asked."""

client = anthropic.Anthropic()

token_count = client.messages.count_tokens(
    model=MODEL,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": "hello"}],
)
print(f"system prompt: {token_count.input_tokens} tokens (Pi's is under 1000)")

response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": "What's 12 * 7?"}],
)
print(next(b.text for b in response.content if b.type == "text"))
