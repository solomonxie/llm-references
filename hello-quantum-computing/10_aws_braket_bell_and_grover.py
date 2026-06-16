# $ venv/bin/python hello-quantum-computing/10_aws_braket_bell_and_grover.py
#
# Goal: the same Bell state and Grover search, now on AWS Braket -- built
# via Braket's chainable `Circuit` API and run on `LocalSimulator` (free,
# no AWS account needed). `LocalSimulator` is a drop-in stand-in for
# `AwsDevice`, the class step 11 uses to point the identical circuit at
# real cloud hardware instead.
# Step 10: AWS Braket SDK -- Bell state + Grover's algorithm, local simulator

from braket.circuits import Circuit
from braket.devices import LocalSimulator

device = LocalSimulator()

# --- Bell state ---
bell = Circuit().h(0).cnot(0, 1)
print("Bell circuit:")
print(bell)

bell_result = device.run(bell, shots=1000).result()
print(f"\ncounts over 1000 shots: {dict(bell_result.measurement_counts)}")
print("(only 00 and 11 should appear -- same entanglement as step 5)\n")

# --- Grover's algorithm, 2 qubits, marked item |11> ---
# Braket has a direct 2-qubit `cz` gate, so like step 9's Q# version this
# oracle is a single gate -- no X-sandwich needed to mark |11>.
grover = Circuit()
grover.h(0).h(1)
grover.cz(0, 1)                    # oracle: marks |11>
grover.h(0).h(1).x(0).x(1)
grover.cz(0, 1)
grover.x(0).x(1).h(0).h(1)         # diffusion
print("Grover circuit (marked = |11>, 1 iteration -- optimal for N=4):")
print(grover)

grover_result = device.run(grover, shots=1000).result()
counts = dict(grover_result.measurement_counts)
print(f"\ncounts over 1000 shots: {counts}  (11 should dominate)")
