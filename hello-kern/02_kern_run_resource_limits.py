# $ venv/bin/python 02_kern_run_resource_limits.py
#
# Goal: `kern run` -- kern's lightest mode, deliberately *not* a full
# sandbox. It caps CPU/memory/process resources on the command it runs
# (like `resource.setrlimit` would), but doesn't containerize it: no new
# rootfs, no network/PID namespace, no seccomp filter. Same leak as step 1
# still works here -- this step exists to show that resource limits alone
# aren't isolation; step 3's `kern box` is what actually closes the gap.
# Step 2: run_code() with `kern run --memory/--cpus`, still leaky
#
# Linux/WSL2 only -- kern has no native macOS binary (README: "macOS
# requires a Linux VM"). Install: `curl -fsSL
# https://raw.githubusercontent.com/getkern/kern/main/install.sh | sh`

import shutil
import subprocess
from pathlib import Path

if shutil.which("kern") is None:
    raise SystemExit("kern not found on PATH -- see the install command in this file's header")

SCRATCH = Path(__file__).parent / "scratch"
SCRATCH.mkdir(exist_ok=True)
(SCRATCH.parent / "secret.txt").write_text("sk-not-a-real-key-but-pretend-this-matters\n")


def run_code(code: str) -> str:
    cmd = ["kern", "run", "--memory", "256M", "--cpus", "0.5", "--", "python3", "-c", code]
    result = subprocess.run(cmd, cwd=SCRATCH, capture_output=True, text=True, timeout=10)
    return result.stdout + result.stderr


# Same code as step 1 -- reads a file outside its own directory. `kern run`
# adds no filesystem boundary, so this still succeeds.
leaky_code = """
from pathlib import Path
print(Path('../secret.txt').read_text())
"""

print(run_code(leaky_code))

# A CPU-bound loop, capped at 0.5 cores -- this is what `kern run` is
# actually for: bounding a resource, not sealing a boundary.
busy_code = """
import time
start = time.time()
x = 0
while time.time() - start < 1:
    x += 1
print(f"iterations in ~1s wall-clock under --cpus 0.5: {x}")
"""

print(run_code(busy_code))
