# hello-kern

Goal: build a sandbox runtime for executing agent-generated code from the actual Linux kernel
primitives up -- rlimits, mount/network/PID namespaces, then seccomp-bpf syscall filtering --
instead of reaching for Docker/gVisor/Firecracker/a hosted sandbox API. `hello-pi`'s
`write_skill` tool execs model-written code with zero isolation; this is what closes that gap.

Each file is a complete, standalone, runnable script.

## Setup

**Linux only** -- steps 3-6 use `unshare`/`mount` (util-linux, preinstalled on virtually every
distro) and unprivileged user namespaces (default-enabled on most distros; check with
`sysctl kernel.unprivileged_userns_clone` if a step fails with `EPERM`). Steps 1-2 run
anywhere Python does.

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # step 6 only

venv/bin/python 01_subprocess_no_sandbox.py
```

## Notes

- Each step adds one isolation layer over the last: rlimits (CPU/memory/process-count caps) ->
  a private mount namespace (masks a sensitive path) -> network + PID namespaces (no route out,
  no visibility into other processes) -> seccomp (kernel-enforced syscall denylist, a backstop
  even if a namespace escape bug existed).
- Step 5's filter is a *denylist* of specifically dangerous syscalls, not a strict allowlist --
  enumerating every syscall a Python interpreter's startup needs is realistic for production
  (trace it with `strace -f -c`) but out of scope here.
- Step 6 loads rlimits/seccomp in an *inner* stage that runs after the outer `unshare`/`mount`
  setup, not before -- the filter blocks `mount`, which the setup step still needs.
