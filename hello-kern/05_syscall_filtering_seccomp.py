# $ venv/bin/python 05_syscall_filtering_seccomp.py
#
# Goal: namespaces (steps 3-4) and rlimits (step 2) are userspace-visible
# boundaries -- a container/namespace escape bug bypasses all of them.
# seccomp-bpf is the kernel's own layer: it filters which syscalls a
# process is allowed to make at all, checked on every syscall regardless
# of what namespace the process thinks it's in. Applied here as a denylist
# of specifically dangerous syscalls (an allowlist would be stricter but
# means enumerating every syscall python3's own startup needs -- doable
# via `strace -f -c`, out of scope for this lesson).
# Step 5: run_code() with a seccomp filter blocking high-risk syscalls
#
# Linux only -- needs `pip install pyseccomp` and a kernel with
# CONFIG_SECCOMP_FILTER (effectively all modern kernels).

import errno
import subprocess
from pathlib import Path

SCRATCH = Path(__file__).parent / "scratch"
SCRATCH.mkdir(exist_ok=True)

# ptrace/mount/module-loading/power-control are the classic container/
# namespace-escape primitives; socket is blocked here as a second,
# kernel-level backstop on top of step 4's network namespace.
DANGEROUS_SYSCALLS = ["ptrace", "mount", "umount2", "reboot", "init_module", "delete_module", "socket"]


def _apply_seccomp():
    import seccomp

    f = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
    for name in DANGEROUS_SYSCALLS:
        try:
            f.add_rule(seccomp.ERRNO(errno.EPERM), name)
        except Exception:
            pass  # syscall not defined for this arch/libseccomp version -- skip, don't crash the sandbox
    f.load()


def run_code(code: str) -> str:
    result = subprocess.run(
        ["python3", "-c", code], cwd=SCRATCH, capture_output=True, text=True, timeout=10, preexec_fn=_apply_seccomp
    )
    return result.stdout + result.stderr


blocked_socket_code = """
import socket
try:
    socket.socket()
    print("socket() succeeded (seccomp filter failed)")
except PermissionError as e:
    print(f"blocked by seccomp: {e}")
"""

print(run_code(blocked_socket_code))
