# $ venv/bin/python 03_agent_loop_with_tool_calls.py
#
# Goal: wire step 1's system prompt and step 2's four tools into an actual
# agentic loop against Claude's native tool-calling API -- send a real
# task, execute whatever tools come back, feed results in, repeat until
# the model stops calling tools. This is pi-agent-core's job in the real
# Pi: the loop is the entire "framework".
# Step 3: The four tools + system prompt, driven by a manual tool-call loop

import subprocess
from pathlib import Path

import anthropic

MODEL = "claude-opus-5"
WORKDIR = Path(__file__).parent / "scratch"
WORKDIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a minimal coding agent. Rules:
- Be concise. No preamble, no summary of what you're about to do.
- Prefer editing an existing file over rewriting it.
- If a tool can answer the question, use it instead of guessing.
- Never explain your reasoning unless asked."""


def read_file(path: str) -> str:
    return (WORKDIR / path).read_text()


def write_file(path: str, content: str) -> str:
    (WORKDIR / path).write_text(content)
    return f"wrote {len(content)} bytes to {path}"


def edit_file(path: str, old_str: str, new_str: str) -> str:
    target = WORKDIR / path
    text = target.read_text()
    if old_str not in text:
        return f"error: old_str not found in {path}"
    target.write_text(text.replace(old_str, new_str, 1))
    return f"replaced 1 occurrence in {path}"


def run_bash(command: str) -> str:
    result = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=10)
    return result.stdout + result.stderr


TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file's full contents",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace the first occurrence of old_str with new_str in a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "run_bash",
        "description": "Run a shell command in the working directory",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
]

DISPATCH = {"read_file": read_file, "write_file": write_file, "edit_file": edit_file, "run_bash": run_bash}

client = anthropic.Anthropic()
task = "Create hello.txt containing 'hi from pi', then read it back and tell me what it says."
messages = [{"role": "user", "content": task}]

while True:
    response = client.messages.create(model=MODEL, max_tokens=4096, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages)

    tool_uses = [b for b in response.content if b.type == "tool_use"]
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":
        break

    tool_results = []
    for call in tool_uses:
        try:
            result = DISPATCH[call.name](**call.input)
            tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": result})
        except Exception as exc:
            tool_results.append(
                {"type": "tool_result", "tool_use_id": call.id, "content": str(exc), "is_error": True}
            )
        print(f"-> {call.name}({call.input})")
    messages.append({"role": "user", "content": tool_results})

print(next(b.text for b in response.content if b.type == "text"))
