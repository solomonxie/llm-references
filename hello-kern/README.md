# hello-kern

Goal: sandbox agent-generated code with [kern](https://github.com/getkern/kern), a real
rootless container runtime purpose-built for this ("a real, kernel-enforced container in
~3.5 ms from an OCI image, no daemon, one static binary") -- instead of hand-rolling the
rlimits/namespaces/seccomp it's built on. `hello-pi`'s `write_skill` tool execs model-written
code with zero isolation; this is what closes that gap.

Each file is a complete, standalone, runnable script.

## Setup

**Linux/WSL2 only** -- kern has no native macOS binary (its own docs: "macOS requires a Linux
VM"); needs unprivileged user namespaces and cgroup v2 (default on most modern distros).

```sh
curl -fsSL https://raw.githubusercontent.com/getkern/kern/main/install.sh | sh
kern doctor                      # verifies kernel prerequisites

# from the repo root
python3 -m venv venv && venv/bin/pip install -r hello-kern/requirements.txt
export ANTHROPIC_API_KEY=...     # step 5 only

venv/bin/python hello-kern/01_subprocess_no_sandbox.py
```

## Notes

- Step 1 is the danger, unsandboxed. Step 2's `kern run` caps CPU/memory/processes but adds
  no filesystem/network/process boundary -- same leak still works, on purpose, to show that
  resource limits alone aren't isolation. Step 3's `kern box` is what actually closes it: a
  full OCI container (own rootfs, own PID/mount/net namespaces), rootless by default, no flags
  needed for that baseline. Step 4 adds `--security-profile untrusted`'s seccomp allowlist on
  top -- a kernel-enforced backstop even if a namespace-escape bug existed.
- Step 5's command is kern's own documented pattern for untrusted/AI-generated code:
  `kern box job --image ... --security-profile untrusted --memory 256m -v ./job:/w -- python3
  /w/x.py`. A real integration would more likely use kern's own SDK (`kern-sandbox` on PyPI)
  instead of shelling out to the CLI, as this file does to keep every flag visible.
- None of these scripts were run in this repo's own (macOS) dev environment -- kern requires
  Linux; verify on a Linux host/VM/WSL2 before relying on this series' exact output.
