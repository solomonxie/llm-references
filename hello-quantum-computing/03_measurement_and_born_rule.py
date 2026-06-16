# $ venv/bin/python hello-quantum-computing/03_measurement_and_born_rule.py
#
# Goal: a state vector isn't directly observable -- measuring collapses
# it to one basis outcome, with probability |amplitude|^2 (the Born
# rule). One measurement tells you almost nothing; this step runs many
# and shows the observed frequencies converge on the theoretical
# probabilities as shot count grows.
# Step 3: measurement, the Born rule, and convergence over repeated shots

import numpy as np

ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

rng = np.random.default_rng(seed=0)


def measure(state: np.ndarray, shots: int) -> np.ndarray:
    probabilities = np.abs(state) ** 2
    return rng.choice([0, 1], size=shots, p=probabilities)


# Skewed state: 90% |0>, 10% |1>.
skewed = np.array([np.sqrt(0.9), np.sqrt(0.1)], dtype=complex)
print(f"state: {skewed}, theoretical P(0)=0.900 P(1)=0.100\n")

for shots in [10, 100, 1_000, 100_000]:
    outcomes = measure(skewed, shots)
    p0 = np.mean(outcomes == 0)
    print(f"{shots:>7} shots -> observed P(0)={p0:.3f} P(1)={1 - p0:.3f}")

# H|0> is exactly 50/50 -- a single measurement can't tell you the state
# was ever in superposition at all, only which basis outcome it collapsed to.
plus = H @ ZERO
outcomes = measure(plus, 10_000)
print(f"\nH|0> over 10,000 shots -> P(0)={np.mean(outcomes == 0):.3f} P(1)={np.mean(outcomes == 1):.3f}")
