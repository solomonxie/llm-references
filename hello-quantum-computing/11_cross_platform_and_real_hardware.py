# $ venv/bin/python hello-quantum-computing/11_cross_platform_and_real_hardware.py
#
# Goal: not a new algorithm -- a reference. Steps 8-10 all ran on free
# local simulators; this is what changes to point the identical circuits
# at real cloud hardware, and why it's left as inert strings rather than
# live calls (an account, billing, and a queued job don't belong in a
# script meant to run unattended).
# Step 11: local simulator -> real QPU per platform, and what each vendor actually hosts

COMPARISON = """
                  Google Cirq          Microsoft Q# (qsharp)      AWS Braket
circuit unit      cirq.Circuit         Q# operation               braket.circuits.Circuit
qubit alloc       LineQubit.range(n)   use qs = Qubit[n]           implicit int indices
superposition     cirq.H(q)            H(q)                        Circuit().h(0)
entangle          cirq.CNOT(q0, q1)    CNOT(q0, q1)                Circuit().cnot(0, 1)
local simulator   cirq.Simulator()     qsharp.eval(...)            LocalSimulator()
"""
print(COMPARISON)

# Google -- Cirq targets Google's own superconducting processors through
# cirq_google.Engine, gated behind a Google Cloud project enrolled in the
# Quantum Computing Service (limited access, invite-only as of this writing):
#
#   import cirq_google
#   engine = cirq_google.Engine(project_id="your-gcp-project")
#   result = engine.run(program=bell_circuit, processor_id="rainbow", repetitions=1000)

# Microsoft -- Azure Quantum is a marketplace, not Microsoft's own
# hardware (their in-house approach is still research-stage topological
# qubits): it routes your Q# job to a partner QPU you choose and pay for
# through your Azure subscription:
#
#   import qsharp.azure
#   qsharp.azure.connect(resourceId="/subscriptions/.../Microsoft.Quantum/Workspaces/...")
#   qsharp.azure.target("ionq.qpu")  # or quantinuum.qpu.h1-1, rigetti.qpu.ankaa-3, ...
#   qsharp.azure.execute(GroverMarked11, shots=1000)

# AWS -- Braket's LocalSimulator swaps for an AwsDevice ARN pointing at a
# managed cloud simulator or a partner QPU, billed per-shot/per-task
# through your AWS account:
#
#   from braket.aws import AwsDevice
#   device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Harmony")
#   result = device.run(bell, shots=1000).result()

print("Real hardware calls above are commented out -- each needs a cloud account,")
print("billing enabled, and returns a queued (not instant) result. Nothing here")
print("executes them; this file is a map of what changes, not a job submission.")
