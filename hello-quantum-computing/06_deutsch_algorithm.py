# $ venv/bin/python hello-quantum-computing/06_deutsch_algorithm.py
#
# Goal: the first taste of quantum advantage. Given a black-box function
# f: {0,1} -> {0,1}, "constant" (f(0)==f(1)) vs "balanced" (f(0)!=f(1))
# needs 2 classical queries in the worst case -- but Deutsch's algorithm
# answers it with exactly 1 quantum query, by putting the input in
# superposition and using phase kickback so the answer ends up encoded in
# an interference pattern rather than in f's output directly.
# Step 6: Deutsch's algorithm -- 1 quantum query vs. 2 classical queries

import numpy as np

ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def tensor(*states):
    result = states[0]
    for s in states[1:]:
        result = np.kron(result, s)
    return result


def oracle_matrix(f) -> np.ndarray:
    # Uf: |x>|y> -> |x>|y XOR f(x)>, built as a 4x4 permutation matrix
    # over basis order |00>,|01>,|10>,|11>.
    matrix = np.zeros((4, 4), dtype=complex)
    for x in (0, 1):
        for y in (0, 1):
            in_idx = 2 * x + y
            out_idx = 2 * x + (y ^ f(x))
            matrix[out_idx, in_idx] = 1
    return matrix


FUNCTIONS = {
    "constant-0": lambda x: 0,
    "constant-1": lambda x: 1,
    "identity (balanced)": lambda x: x,
    "negation (balanced)": lambda x: 1 - x,
}


def deutsch(f) -> int:
    Uf = oracle_matrix(f)
    state = tensor(ZERO, ONE)          # |0>|1>
    state = tensor(H, H) @ state       # superposition on both qubits
    state = Uf @ state                 # phase kickback: f(x) ends up as a phase on |x>
    state = tensor(H, np.eye(2)) @ state  # interfere the input qubit back down

    # Measuring qubit 0: 0 -> constant, 1 -> balanced. The state is exact
    # (no superposition left on qubit 0), so this "measurement" is
    # deterministic -- unlike steps 3 and 5 there's nothing to sample.
    probabilities = np.abs(state) ** 2
    p_qubit0_is_1 = probabilities[2] + probabilities[3]  # |10> + |11>
    return round(p_qubit0_is_1)


for name, f in FUNCTIONS.items():
    verdict = "balanced" if deutsch(f) else "constant"
    actual = "balanced" if f(0) != f(1) else "constant"
    print(f"{name:>22}: quantum verdict = {verdict:<9} (actual: {actual}, 1 query used)")
