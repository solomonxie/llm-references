# $ venv/bin/python hello-kern/04_untrusted_security_profile.py
#
# Goal: namespaces (step 3) are userspace-visible boundaries -- a
# container-escape bug bypasses all of them. kern's `--security-profile
# untrusted` adds the kernel's own layer on top: a seccomp *allowlist*
# (mobo's default profile minus 35 known escape-prone syscalls), checked
# on every syscall the process makes, regardless of what namespace it
# thinks it's in -- a syscall outside the vetted set returns ENOSYS
# instead of running. This is the profile kern's own docs recommend for
# untrusted/AI-generated code specifically.
# Step 4: run_code() with `--security-profile untrusted`, seccomp allowlist added
#
# Linux/WSL2 only -- same requirements as step 2.

import shutil
import subprocess
from pathlib import Path

if shutil.which("kern") is None:
    raise SystemExit("kern not found on PATH -- see step 2's header for the install command")

SCRATCH = Path(__file__).parent / "scratch"
SCRATCH.mkdir(exist_ok=True)


def run_code(code: str, *, untrusted: bool) -> str:
    cmd = ["kern", "box", "job", "--image", "python:3.12-slim", "-v", f"{SCRATCH}:/w"]
    if untrusted:
        cmd += ["--security-profile", "untrusted"]
    cmd += ["--", "python3", "-c", code]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout + result.stderr


# `mount`/`ptrace`/etc. are exactly the kind of syscall a container-escape
# CVE walks through -- allowed by default (a normal container image may
# legitimately need some of them), denied under the untrusted profile.
escape_attempt = """
import ctypes, errno
libc = ctypes.CDLL(None, use_errno=True)
rc = libc.ptrace(0, 0, 0, 0)  # PTRACE_TRACEME
print("ptrace allowed" if rc == 0 else f"ptrace blocked: errno={errno.errorcode.get(ctypes.get_errno())}")
"""

print("default profile:")
print(run_code(escape_attempt, untrusted=False))

print("--security-profile untrusted:")
print(run_code(escape_attempt, untrusted=True))
