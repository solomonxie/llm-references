# $ venv/bin/python 06_wire_into_agent_tool.py
#
# Goal: compose steps 2-5 into one run_code_sandboxed() and expose it as a
# single tool in a real agent loop -- this is the `run_bash` tool from
# `hello-pi`, hardened. The mount/PID/net namespaces (needing the `mount`
# and `unshare` syscalls) are set up by a trusted outer wrapper *before*
# rlimits and the seccomp filter are applied to the untrusted code's own
# interpreter -- applying seccomp any earlier would block the setup step's
# own use of `mount`.
# Step 6: One hardened run_code tool, wired into an agent loop
#
# Linux only -- same requirements as steps 3-5.

import shlex
import subprocess
from pathlib import Path

import anthropic

MODEL = "claude-opus-5"
SCRATCH = Path(__file__).parent / "scratch"
SCRATCH.mkdir(exist_ok=True)
SECRET = SCRATCH.parent / "secret.txt"
SECRET.write_text("sk-not-a-real-key-but-pretend-this-matters\n")

DANGEROUS_SYSCALLS = ["ptrace", "mount", "umount2", "reboot", "init_module", "delete_module", "socket"]

# Runs *inside* the already-namespaced sandbox: locks itself down with
# rlimits + seccomp, then execs the untrusted code -- this is the "inner
# stage" that step 5's filter needs to apply to, not the outer wrapper.
INNER_HARNESS = f"""
import errno, resource
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))
resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024,) * 2)
import seccomp
f = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
for name in {DANGEROUS_SYSCALLS!r}:
    try:
        f.add_rule(seccomp.ERRNO(errno.EPERM), name)
    except Exception:
        pass
f.load()
exec(compile({{CODE}}, "<sandboxed>", "exec"))
"""


def run_code_sandboxed(code: str) -> str:
    inner = INNER_HARNESS.replace("{CODE}", repr(code))
    setup = f"mount --bind /dev/null {shlex.quote(str(SECRET))} && exec python3 -c {shlex.quote(inner)}"
    cmd = [
        "unshare", "--mount", "--user", "--map-root-user",
        "--net", "--pid", "--fork", "--mount-proc",
        "--", "bash", "-c", setup,
    ]
    result = subprocess.run(cmd, cwd=SCRATCH, capture_output=True, text=True, timeout=10)
    return result.stdout + result.stderr


TOOLS = [
    {
        "name": "run_code",
        "description": "Run Python code in a sandbox with no network, its own PID namespace, and rlimits/seccomp applied",
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
