# $ venv/bin/python hello-quantum-computing/02_single_qubit_gates.py
#
# Goal: gates are unitary matrices; applying a gate is a matrix-vector
# multiply against the state from step 1. X/Y/Z are Pauli rotations, H
# is the gate that actually makes superposition (a classical bit has no
# equivalent), and gates compose by matrix multiplication in the order
# applied.
# Step 2: single-qubit gates (X, Y, Z, H) as unitary matrices

import numpy as np

ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)

X = np.array([[0, 1], [1, 0]], dtype=complex)          # bit flip: |0><->|1>
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)        # bit + phase flip
Z = np.array([[1, 0], [0, -1]], dtype=complex)          # phase flip: |1> -> -|1>
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # superposition


def apply(gate: np.ndarray, state: np.ndarray) -> np.ndarray:
    return gate @ state


def show(name: str, state: np.ndarray) -> None:
    p0, p1 = abs(state[0]) ** 2, abs(state[1]) ** 2
    print(f"{name}: {np.round(state, 3)}  P(0)={p0:.3f} P(1)={p1:.3f}")


show("|0>", ZERO)
show("X|0>", apply(X, ZERO))     # flips to |1>
show("Z|0>", apply(Z, ZERO))     # |0> has no |1> component, so Z does nothing visible
show("H|0>", apply(H, ZERO))     # equal superposition, step 1's |+>

print()
plus = apply(H, ZERO)
show("H|0>       ", plus)
show("Z(H|0>)     ", apply(Z, plus))       # same probabilities, opposite relative phase
show("H(Z(H|0>))  ", apply(H, apply(Z, plus)))  # H-Z-H composes to X -- verified below

# Every gate here is unitary: U applied then U's conjugate transpose
# undoes it exactly, which is what keeps quantum computation reversible.
for name, gate in [("X", X), ("Y", Y), ("Z", Z), ("H", H)]:
    identity_check = gate.conj().T @ gate
    print(f"\n{name} unitary (U^dagger U = I): {np.allclose(identity_check, np.eye(2))}")

hzh = H @ Z @ H
print(f"\nH.Z.H == X: {np.allclose(hzh, X)}")
