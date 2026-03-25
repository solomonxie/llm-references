# $ venv/bin/python 03_filesystem_isolation_mount_namespace.py
#
# Goal: rlimits (step 2) bound CPU/memory but do nothing about the
# filesystem-escape problem from step 1. A private mount namespace
# (`unshare --mount`) lets this process change what's mounted where
# *without affecting the host or any other process* -- bind-mounting
# /dev/null over a sensitive path makes it unreadable from inside the
# sandbox only. `--user --map-root-user` makes this work without root,
# via an unprivileged user namespace.
# Step 3: run_code() inside a private mount namespace that masks secret.txt
#
# Linux only -- needs util-linux's `unshare`/`mount` and unprivileged user
# namespaces enabled (the default on most distros; check with
# `sysctl kernel.unprivileged_userns_clone` if this fails with EPERM).

import shlex
import subprocess
from pathlib import Path

SCRATCH = Path(__file__).parent / "scratch"
SCRATCH.mkdir(exist_ok=True)
SECRET = SCRATCH.parent / "secret.txt"
SECRET.write_text("sk-not-a-real-key-but-pretend-this-matters\n")


def run_code(code: str) -> str:
    inner = f"mount --bind /dev/null {shlex.quote(str(SECRET))} && exec python3 -c {shlex.quote(code)}"
    cmd = ["unshare", "--mount", "--user", "--map-root-user", "--", "bash", "-c", inner]
    result = subprocess.run(cmd, cwd=SCRATCH, capture_output=True, text=True, timeout=10)
    return result.stdout + result.stderr


# Same leaky code as step 1 -- this time the traversal reads /dev/null's
# empty content instead of the real secret, and only inside this process's
# own mount namespace. `cat ../secret.txt` on the host still shows the key.
leaky_code = """
from pathlib import Path
print(repr(Path('../secret.txt').read_text()))
"""

print(run_code(leaky_code))
