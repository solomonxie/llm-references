# $ venv/bin/python 04_network_and_pid_isolation.py
#
# Goal: step 3 sealed one file path; the code can still open sockets and
# see the host's entire process table. Two more `unshare` namespaces close
# both: `--net` gives a fresh network stack with only a loopback interface
# (no route out, no DNS), and `--pid --fork --mount-proc` gives a private
# PID namespace where the sandboxed code is PID 1 and `ps` shows nothing
# else.
# Step 4: run_code() with its own network and PID namespaces added
#
# Linux only -- same requirements as step 3, plus `--mount-proc` needs
# /proc to be mountable in the new namespace (true by default).

import shlex
import subprocess
from pathlib import Path

SCRATCH = Path(__file__).parent / "scratch"
SCRATCH.mkdir(exist_ok=True)
SECRET = SCRATCH.parent / "secret.txt"
SECRET.write_text("sk-not-a-real-key-but-pretend-this-matters\n")


def run_code(code: str) -> str:
    inner = f"mount --bind /dev/null {shlex.quote(str(SECRET))} && exec python3 -c {shlex.quote(code)}"
    cmd = [
        "unshare",
        "--mount",
        "--user",
        "--map-root-user",
        "--net",  # new, isolated network stack -- loopback only
        "--pid",
        "--fork",  # new PID namespace; --fork so the shell becomes PID 1 correctly
        "--mount-proc",  # so `ps`/`/proc` inside reflect the new PID namespace, not the host's
        "--",
        "bash",
        "-c",
        inner,
    ]
    result = subprocess.run(cmd, cwd=SCRATCH, capture_output=True, text=True, timeout=10)
    return result.stdout + result.stderr


no_network_code = """
import socket
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2)
    print("reached the network (sandbox failed)")
except OSError as e:
    print(f"blocked as expected: {e}")
"""

only_itself_code = """
import subprocess
print(subprocess.run(["ps", "-e"], capture_output=True, text=True).stdout)
"""

print(run_code(no_network_code))
print(run_code(only_itself_code))
