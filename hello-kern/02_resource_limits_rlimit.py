# $ venv/bin/python 02_resource_limits_rlimit.py
#
# Goal: step 1 had no limit on CPU time, memory, or process count -- an
# infinite loop or a fork bomb takes the host down with it. `resource.
# setrlimit`, applied in a `preexec_fn` that runs in the forked child right
# before exec, caps all three. Doesn't touch the filesystem-escape problem
# from step 1 yet -- that's step 3.
# Step 2: run_code() with CPU/memory/process rlimits

import resource
import subprocess
from pathlib import Path

SCRATCH = Path(__file__).parent / "scratch"
SCRATCH.mkdir(exist_ok=True)


def _limit_resources():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))  # 2 CPU seconds
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)  # 256MB address space
    resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))  # no fork bombs
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024,) * 2)  # 10MB max file size


def run_code(code: str) -> str:
    # preexec_fn runs in the child after fork(), before exec() -- exactly
    # where a limit needs to be in place to bind the process that's about
    # to run untrusted code, not the parent that's launching it.
    result = subprocess.run(
        ["python3", "-c", code], cwd=SCRATCH, capture_output=True, text=True, timeout=10, preexec_fn=_limit_resources
    )
    return f"[exit {result.returncode}] " + result.stdout + result.stderr


print(run_code("while True: pass"))  # killed by RLIMIT_CPU, not the 10s timeout
print(run_code("x = bytearray(10**9)"))  # MemoryError from RLIMIT_AS, no OOM-killed host
