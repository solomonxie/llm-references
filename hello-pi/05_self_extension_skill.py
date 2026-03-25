# $ venv/bin/python 05_self_extension_skill.py <session> "<task>"
# $ venv/bin/python 05_self_extension_skill.py mysession "write yourself a tool that reverses a string, then use it on 'pi'"
# $ venv/bin/python 05_self_extension_skill.py mysession "use the reverse_string skill on 'again'"   # skill persisted from last run
#
# Goal: Pi's real power isn't the four built-in tools, it's that
# extensions can add more -- "software building software". Adds a fifth,
# meta tool, write_skill, that lets the agent define a brand-new Python
# tool for itself at runtime; new skills are registered into this same
# loop immediately and persisted to disk so later sessions inherit them.
# Step 5: A write_skill tool that extends the agent's own tool set

import json
import subprocess
import sys
from pathlib import Path

import anthropic

MODEL = "claude-opus-5"
WORKDIR = Path(__file__).parent / "scratch"
SESSIONS_DIR = Path(__file__).parent / ".sessions"
SKILLS_FILE = Path(__file__).parent / ".sessions" / "skills.json"
WORKDIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a minimal coding agent. Rules:
- Be concise. No preamble, no summary of what you're about to do.
- Prefer editing an existing file over rewriting it.
- If a tool can answer the question, use it instead of guessing.
- If no existing tool fits and the task would benefit from a reusable
  helper, write one with write_skill instead of solving it inline.
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


BASE_TOOLS = [
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

BASE_DISPATCH = {"read_file": read_file, "write_file": write_file, "edit_file": edit_file, "run_bash": run_bash}

# New in step 5: the meta tool. code must define exactly one top-level
# function named `name`; write_skill execs it and wires the resulting
# callable straight into DISPATCH/TOOLS for the rest of this run.
WRITE_SKILL_TOOL = {
    "name": "write_skill",
    "description": "Define a new tool for yourself: a Python function plus its JSON Schema, available immediately and in future sessions",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "function name, also the tool name"},
            "description": {"type": "string"},
            "input_schema": {"type": "object", "description": "JSON Schema for the function's arguments"},
            "code": {"type": "string", "description": "full `def <name>(...):` source, stdlib only"},
        },
        "required": ["name", "description", "input_schema", "code"],
    },
}


def load_skills() -> dict:
    return json.loads(SKILLS_FILE.read_text()) if SKILLS_FILE.exists() else {}


def save_skills(skills: dict) -> None:
    SKILLS_FILE.write_text(json.dumps(skills, indent=2))


def compile_skill(name: str, code: str):
    namespace: dict = {}
    exec(code, namespace)  # trusted input in this toy -- a real Pi sandboxes this (see hello-kern)
    return namespace[name]


def register_skill(skills: dict, dispatch: dict, tools: list, name: str, description: str, input_schema: dict, code: str) -> str:
    dispatch[name] = compile_skill(name, code)
    tools.append({"name": name, "description": description, "input_schema": input_schema})
    skills[name] = {"description": description, "input_schema": input_schema, "code": code}
    save_skills(skills)
    return f"registered new skill: {name}"


def session_path(name: str) -> Path:
    return SESSIONS_DIR / f"{name}.json"


def load_session(name: str) -> list:
    path = session_path(name)
    return json.loads(path.read_text()) if path.exists() else []


def save_session(name: str, messages: list) -> None:
    session_path(name).write_text(json.dumps(messages, indent=2))


# Start from the four built-ins, then re-register every skill taught in a
# previous run -- self-extension persists across processes, same as the
# session history does.
skills = load_skills()
dispatch = dict(BASE_DISPATCH)
tools = list(BASE_TOOLS) + [WRITE_SKILL_TOOL]
for skill_name, skill in skills.items():
    dispatch[skill_name] = compile_skill(skill_name, skill["code"])
    tools.append({"name": skill_name, "description": skill["description"], "input_schema": skill["input_schema"]})

session_name, task = sys.argv[1], sys.argv[2]
messages = load_session(session_name)
messages.append({"role": "user", "content": task})

client = anthropic.Anthropic()

while True:
    response = client.messages.create(model=MODEL, max_tokens=4096, system=SYSTEM_PROMPT, tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": [b.model_dump(mode="json") for b in response.content]})

    if response.stop_reason != "tool_use":
        break

    tool_results = []
    for call in [b for b in response.content if b.type == "tool_use"]:
        try:
            if call.name == "write_skill":
                result = register_skill(skills, dispatch, tools, **call.input)
            else:
                result = dispatch[call.name](**call.input)
            tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": result})
        except Exception as exc:
            tool_results.append(
                {"type": "tool_result", "tool_use_id": call.id, "content": str(exc), "is_error": True}
            )
        print(f"-> {call.name}({call.input})")
    messages.append({"role": "user", "content": tool_results})

save_session(session_name, messages)
print(next(b.text for b in response.content if b.type == "text"))
