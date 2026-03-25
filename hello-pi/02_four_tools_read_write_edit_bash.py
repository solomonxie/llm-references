# $ venv/bin/python 02_four_tools_read_write_edit_bash.py
#
# Goal: Pi ships exactly four tools -- Read, Write, Edit, Bash -- and
# nothing else; every other capability (searching, testing, git) is
# expected to go through Bash. Defined here as plain functions with JSON
# Schema, smoke-tested directly with no LLM call yet -- step 3 wires them
# into a real agentic loop.
# Step 2: The four tools, implemented and self-tested

import subprocess
from pathlib import Path

WORKDIR = Path(__file__).parent / "scratch"
WORKDIR.mkdir(exist_ok=True)


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
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
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
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]

DISPATCH = {"read_file": read_file, "write_file": write_file, "edit_file": edit_file, "run_bash": run_bash}

if __name__ == "__main__":
    print(write_file("greeting.txt", "hi from pi"))
    print(read_file("greeting.txt"))
    print(edit_file("greeting.txt", "hi from pi", "hello from pi"))
    print(read_file("greeting.txt"))
    print(run_bash("wc -c greeting.txt"))
