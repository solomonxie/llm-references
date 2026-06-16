# $ venv/bin/python hello-quantum-computing/07_grover_search.py
#
# Goal: unstructured search over N items takes O(N) classical queries on
# average. Grover's algorithm finds a marked item in O(sqrt(N)) by
# alternating an oracle (flips the marked item's phase) with a diffusion
# step (reflects every amplitude about the mean), which amplifies the
# marked item's probability each round -- interference doing the work
# classical search can't.
# Step 7: Grover's search algorithm, from scratch, general n qubits

import numpy as np

N_QUBITS = 3
N = 2 ** N_QUBITS          # 8 items to search over
MARKED = 5                 # the "needle" -- binary 101


def uniform_superposition(n: int) -> np.ndarray:
    return np.full(n, 1 / np.sqrt(n), dtype=complex)


def oracle(n: int, marked: int) -> np.ndarray:
    # Flips the sign of exactly the marked basis state's amplitude.
    diag = np.ones(n, dtype=complex)
    diag[marked] = -1
    return np.diag(diag)


def diffusion(n: int) -> np.ndarray:
    # 2|s><s| - I, where |s> is the uniform superposition -- reflects
    # every amplitude about their mean, boosting whatever the oracle
    # just suppressed.
    mean_projector = np.full((n, n), 2 / n, dtype=complex)
    return mean_projector - np.eye(n, dtype=complex)


Uf = oracle(N, MARKED)
D = diffusion(N)
optimal_iterations = round(np.pi / 4 * np.sqrt(N))
print(f"N={N} items, marked={MARKED} (binary {MARKED:0{N_QUBITS}b}), "
      f"optimal iterations = {optimal_iterations}")

state = uniform_superposition(N)
print(f"\nstart: P(marked) = {abs(state[MARKED]) ** 2:.3f}  (baseline: 1/N = {1 / N:.3f})")

for i in range(1, optimal_iterations + 1):
    state = D @ (Uf @ state)
    print(f"after iteration {i}: P(marked) = {abs(state[MARKED]) ** 2:.3f}")

# One more iteration past optimal overshoots -- probability comes back down.
overshoot = D @ (Uf @ state)
print(f"after 1 extra (overshoot): P(marked) = {abs(overshoot[MARKED]) ** 2:.3f}")

rng = np.random.default_rng(seed=0)
probabilities = np.abs(state) ** 2
outcomes = rng.choice(N, size=1000, p=probabilities)
hit_rate = np.mean(outcomes == MARKED)
print(f"\n1000 shots -> found marked item {hit_rate:.1%} of the time "
      f"(classical random guess: {1 / N:.1%})")
