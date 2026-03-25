# $ venv/bin/python 01_subprocess_no_sandbox.py
#
# Goal: the starting point for any "let the agent run code" tool --
# subprocess with zero isolation. Shows the actual danger: agent-generated
# code inherits the full filesystem, network, and process table of the
# host process. Every later step in this series closes one of these gaps.
# Step 1: run_code() with no sandboxing at all

import subprocess
from pathlib import Path

SCRATCH = Path(__file__).parent / "scratch"
SCRATCH.mkdir(exist_ok=True)

# Stand-in for something the agent's sandbox is supposed to protect --
# a real deployment might have API keys or SSH keys nearby instead.
(SCRATCH.parent / "secret.txt").write_text("sk-not-a-real-key-but-pretend-this-matters\n")


def run_code(code: str) -> str:
    result = subprocess.run(["python3", "-c", code], cwd=SCRATCH, capture_output=True, text=True, timeout=10)
    return result.stdout + result.stderr


# "Agent-generated" code that only needed to read a file in its own
# directory -- but nothing stops it from walking up and reading anything
# the host process can see.
leaky_code = """
from pathlib import Path
print(Path('../secret.txt').read_text())
"""

print(run_code(leaky_code))
