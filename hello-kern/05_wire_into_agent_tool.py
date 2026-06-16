# $ venv/bin/python 05_wire_into_agent_tool.py
#
# Goal: compose steps 2-4 into one run_code_sandboxed() and expose it as a
# single tool in a real agent loop -- this is the `run_bash` tool from
# `hello-pi`, hardened. The command below is kern's own documented pattern
# for untrusted/AI-generated code: a container (step 3's isolation) +
# --security-profile untrusted (step 4's seccomp allowlist) + a memory cap
# (step 2's resource limiting), together in one invocation.
# Step 5: One hardened run_code tool, wired into an agent loop
#
# Linux/WSL2 only -- same requirements as step 2. A real agent framework
# would more likely use kern's own SDK (`pip install kern-sandbox`,
# `from kern_sandbox import run_code`) instead of shelling out to the CLI
# as this file does -- shown here because it makes every flag explicit.

import shutil
import subprocess
from pathlib import Path

import anthropic

if shutil.which("kern") is None:
    raise SystemExit("kern not found on PATH -- see step 2's header for the install command")

MODEL = "claude-opus-5"
JOB_DIR = Path(__file__).parent / "job"
JOB_DIR.mkdir(exist_ok=True)
(JOB_DIR.parent / "secret.txt").write_text("sk-not-a-real-key-but-pretend-this-matters\n")


def run_code_sandboxed(code: str) -> str:
    (JOB_DIR / "x.py").write_text(code)
    cmd = [
        "kern", "box", "job", "--image", "python:3.12-slim",
        "--security-profile", "untrusted", "--memory", "256m",
        "-v", f"{JOB_DIR}:/w",
        "--", "python3", "/w/x.py",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout + result.stderr


TOOLS = [
    {
        "name": "run_code",
        "description": "Run Python code in a hardened kern sandbox: its own container, no network route out, memory-capped, seccomp allowlist",
        "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    }
]

client = anthropic.Anthropic()
messages = [
    {
        "role": "user",
        "content": "Run some code that tries to read ../secret.txt and also tries to open a network socket. Report what happens to each.",
    }
]

while True:
    response = client.messages.create(model=MODEL, max_tokens=2048, tools=TOOLS, messages=messages)
    messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason != "tool_use":
        break

    tool_results = []
    for call in [b for b in response.content if b.type == "tool_use"]:
        result = run_code_sandboxed(call.input["code"])
        print(f"-> run_code:\n{call.input['code']}\n== output ==\n{result}")
        tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": result})
    messages.append({"role": "user", "content": tool_results})

print(next(b.text for b in response.content if b.type == "text"))
