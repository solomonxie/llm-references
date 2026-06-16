# $ venv/bin/python hello-quantum-computing/04_multi_qubit_systems.py
#
# Goal: n qubits live in a 2^n-dimensional vector, built from single-qubit
# states via the tensor (Kronecker) product -- not concatenation. Gates on
# one qubit within a multi-qubit system are the gate tensored with
# identity on every other qubit, which is also why state space (and
# simulation cost) explodes exponentially with qubit count.
# Step 4: multi-qubit systems, tensor products, basis states

import numpy as np

ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def tensor(*states: np.ndarray) -> np.ndarray:
    result = states[0]
    for s in states[1:]:
        result = np.kron(result, s)
    return result


# Two-qubit basis states: |00>, |01>, |10>, |11> -- 4 = 2^2 amplitudes.
for a, b, label in [(ZERO, ZERO, "|00>"), (ZERO, ONE, "|01>"), (ONE, ZERO, "|10>"), (ONE, ONE, "|11>")]:
    print(f"{label} = {tensor(a, b).real.astype(int)}")

# 3 qubits: 8 = 2^3 amplitudes. This doubling per qubit is the reason
# classical simulation of quantum circuits becomes intractable past ~40-50
# qubits, and why real hardware exists at all.
q3 = tensor(ZERO, ZERO, ZERO)
print(f"\n|000> has {len(q3)} amplitudes (2^3)")
for n in [10, 20, 30, 50]:
    print(f"  {n} qubits -> 2^{n} = {2 ** n:,} amplitudes")

# Applying a gate to qubit 0 of a 2-qubit system: tensor the gate with
# identity on the untouched qubit, then matrix-multiply the combined state.
H_on_qubit0 = tensor(H, I)
state = tensor(ZERO, ZERO)
result = H_on_qubit0 @ state
print(f"\nH on qubit 0 of |00>: {np.round(result, 3)}  (superposition of |00> and |10>)")

X_on_qubit1 = tensor(I, X)
result = X_on_qubit1 @ tensor(ZERO, ZERO)
print(f"X on qubit 1 of |00>: {result.real.astype(int)}  (|00> -> |01>)")

# Product states factor cleanly back into per-qubit states; not every
# 4-amplitude vector does -- step 5 builds one that doesn't.
factorable = tensor(H @ ZERO, ZERO)
print(f"\n|+>|0> = {np.round(factorable, 3)} -- factors as (H|0>) tensor |0>")
