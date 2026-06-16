# $ venv/bin/python hello-quantum-computing/05_entanglement_bell_state.py
#
# Goal: H|0> tensor |0>, then CNOT, produces a Bell state -- a 2-qubit
# state that provably cannot be written as (qubit A) tensor (qubit B).
# Measuring either qubit is still 50/50 individually, but the two
# outcomes are perfectly correlated: this is entanglement, the resource
# every 2+ qubit quantum algorithm actually exploits.
# Step 5: CNOT, Bell states, and verifying non-factorability

import numpy as np

ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I = np.eye(2, dtype=complex)

# CNOT: flips the target qubit iff the control qubit is |1>. Basis order
# is |00>,|01>,|10>,|11>; qubit 0 is control, qubit 1 is target.
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)


def tensor(*states):
    result = states[0]
    for s in states[1:]:
        result = np.kron(result, s)
    return result


H_on_qubit0 = tensor(H, I)
bell = CNOT @ (H_on_qubit0 @ tensor(ZERO, ZERO))
print(f"Bell state |Phi+> = {np.round(bell, 3)}")
print("(only |00> and |11> have nonzero amplitude -- |01> and |10> are impossible)")

# Non-factorability: if bell == tensor(a, b) for some single-qubit a, b,
# then bell[1] (the |01> amplitude, a0*b1) times bell[2] (|10>, a1*b0)
# would equal bell[0] (|00>, a0*b0) times bell[3] (|11>, a1*b1). It doesn't.
cross = bell[1] * bell[2]
diag = bell[0] * bell[3]
print(f"\nfactorability check: a0*b1 * a1*b0 = {cross:.3f}  vs  a0*b0 * a1*b1 = {diag:.3f}")
print(f"equal (would mean factorable): {np.isclose(cross, diag)}")

# Each qubit alone is still 50/50 -- entanglement is about the CORRELATION
# between outcomes, not either qubit individually being more predictable.
rng = np.random.default_rng(seed=0)
probabilities = np.abs(bell) ** 2  # over |00>,|01>,|10>,|11>
outcomes = rng.choice(4, size=10_000, p=probabilities)
q0 = (outcomes >> 1) & 1
q1 = outcomes & 1
print(f"\nqubit 0 alone: P(0)={np.mean(q0 == 0):.3f} P(1)={np.mean(q0 == 1):.3f}  (still 50/50)")
print(f"P(q0 == q1) = {np.mean(q0 == q1):.3f}  (should be ~1.0 -- always agree)")
