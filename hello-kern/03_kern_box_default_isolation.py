# $ venv/bin/python 03_kern_box_default_isolation.py
#
# Goal: `kern box` -- a real OCI container: its own root filesystem (from
# an image, via overlay + `pivot_root`), its own PID/mount/network/UTS/IPC
# namespaces, always rootless. Unlike step 2's `kern run`, this actually
# seals the boundary -- the same leaky_code from steps 1-2 now fails,
# because "../secret.txt" doesn't exist inside the container's own rootfs
# at all. No flags needed for this baseline; it's `kern box`'s default.
# Step 3: run_code() inside `kern box`, the leak finally closed
#
# Linux/WSL2 only -- same requirements as step 2.

import shutil
import subprocess
from pathlib import Path

if shutil.which("kern") is None:
    raise SystemExit("kern not found on PATH -- see step 2's header for the install command")

SCRATCH = Path(__file__).parent / "scratch"
SCRATCH.mkdir(exist_ok=True)
(SCRATCH.parent / "secret.txt").write_text("sk-not-a-real-key-but-pretend-this-matters\n")


def run_code(code: str) -> str:
    # `-v SCRATCH:/w` is the only thing the container can see of the host --
    # secret.txt, one directory up on the host, simply isn't mounted in.
    cmd = [
        "kern", "box", "job", "--image", "python:3.12-slim",
        "-v", f"{SCRATCH}:/w",
        "--", "python3", "-c", code,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout + result.stderr


leaky_code = """
from pathlib import Path
print(Path('../secret.txt').read_text())
"""
print("attempting the same leak as steps 1-2:")
print(run_code(leaky_code))

# The host's process table is invisible too -- `kern box` gives the
# container its own PID namespace, so the sandboxed code is PID 1 (or
# close to it) and sees nothing else.
ps_code = "import subprocess; print(subprocess.run(['ps', '-e'], capture_output=True, text=True).stdout)"
print("process table as seen from inside the box:")
print(run_code(ps_code))
