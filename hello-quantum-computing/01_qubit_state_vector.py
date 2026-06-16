# $ venv/bin/python hello-quantum-computing/01_qubit_state_vector.py
#
# Goal: a qubit is a 2-element complex vector [alpha, beta] with
# |alpha|^2 + |beta|^2 = 1 -- alpha is "how much |0>", beta is "how much
# |1>". Everything later (gates, entanglement, algorithms) is just linear
# algebra on vectors like these. No quantum SDK yet -- plain numpy, so the
# mechanism is fully visible.
# Step 1: qubit state vector, computational basis, superposition

import numpy as np

ZERO = np.array([1, 0], dtype=complex)   # |0>
ONE = np.array([0, 1], dtype=complex)    # |1>


def is_normalized(state: np.ndarray) -> bool:
    return np.isclose(np.sum(np.abs(state) ** 2), 1.0)


print(f"|0> = {ZERO}, normalized: {is_normalized(ZERO)}")
print(f"|1> = {ONE}, normalized: {is_normalized(ONE)}")

# Equal superposition: alpha = beta = 1/sqrt(2) -- 50/50 chance of
# measuring |0> or |1>, unlike a classical bit which is one or the other.
plus = (ZERO + ONE) / np.sqrt(2)
print(f"\n|+> = {plus}, normalized: {is_normalized(plus)}")
print(f"P(0) = {abs(plus[0]) ** 2:.3f}, P(1) = {abs(plus[1]) ** 2:.3f}")

# An unequal superposition still has to be normalized.
skewed = np.array([np.sqrt(0.9), np.sqrt(0.1)], dtype=complex)
print(f"\nskewed = {skewed}, normalized: {is_normalized(skewed)}")
print(f"P(0) = {abs(skewed[0]) ** 2:.3f}, P(1) = {abs(skewed[1]) ** 2:.3f}")

# A vector that doesn't sum to 1 isn't a valid quantum state.
invalid = np.array([1, 1], dtype=complex)
print(f"\ninvalid = {invalid}, normalized: {is_normalized(invalid)}  (not a real state)")
