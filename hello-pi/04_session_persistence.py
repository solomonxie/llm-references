# $ venv/bin/python 04_session_persistence.py <session> "<task>"
# $ venv/bin/python 04_session_persistence.py mysession "create foo.txt with 'x'"
# $ venv/bin/python 04_session_persistence.py mysession "now read foo.txt back"   # resumes
#
# Goal: Pi's extension system can persist state into sessions, so a
# conversation survives a process restart. Same loop as step 3, but the
# message history now round-trips through a JSON file keyed by session
# name instead of living only in a local variable.
# Step 4: Save/load the message history to disk, keyed by session name

import json
import subprocess
import sys
from pathlib import Path

import anthropic

MODEL = "claude-opus-5"
WORKDIR = Path(__file__).parent / "scratch"
SESSIONS_DIR = Path(__file__).parent / ".sessions"
WORKDIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

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


# New in step 4: load/save the plain-dict history so a session survives
# a process restart. Every entry is JSON-serializable by construction --
# assistant turns are model_dump()'d before being appended (see below),
# so there's nothing SDK-specific left to serialize here.
def session_path(name: str) -> Path:
    return SESSIONS_DIR / f"{name}.json"


def load_session(name: str) -> list:
    path = session_path(name)
    return json.loads(path.read_text()) if path.exists() else []


def save_session(name: str, messages: list) -> None:
    session_path(name).write_text(json.dumps(messages, indent=2))


session_name, task = sys.argv[1], sys.argv[2]
messages = load_session(session_name)
messages.append({"role": "user", "content": task})

client = anthropic.Anthropic()

while True:
    response = client.messages.create(model=MODEL, max_tokens=4096, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages)

    # New in step 4: model_dump() the response content immediately so
    # `messages` stays JSON-serializable at every point, not just at exit.
    messages.append({"role": "assistant", "content": [b.model_dump(mode="json") for b in response.content]})

    if response.stop_reason != "tool_use":
        break

    tool_results = []
    for call in [b for b in response.content if b.type == "tool_use"]:
        try:
            result = DISPATCH[call.name](**call.input)
            tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": result})
        except Exception as exc:
            tool_results.append(
                {"type": "tool_result", "tool_use_id": call.id, "content": str(exc), "is_error": True}
            )
        print(f"-> {call.name}({call.input})")
    messages.append({"role": "user", "content": tool_results})

save_session(session_name, messages)
print(next(b.text for b in response.content if b.type == "text"))
