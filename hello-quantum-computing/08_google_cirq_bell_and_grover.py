# $ venv/bin/python hello-quantum-computing/08_google_cirq_bell_and_grover.py
#
# Goal: the same Bell state (step 5) and Grover search (step 7), now on
# Google's Cirq -- a real SDK built by Google Quantum AI, run here on
# Cirq's built-in local simulator (free, no Google Cloud account). Same
# physics, real vocabulary: qubits, gates, circuits, measurement.
# Step 8: Google Cirq -- Bell state + Grover's algorithm, local simulator

import cirq

simulator = cirq.Simulator()

# --- Bell state ---
q0, q1 = cirq.LineQubit.range(2)
bell_circuit = cirq.Circuit([
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key="result"),
])
print("Bell circuit:")
print(bell_circuit)

bell_result = simulator.run(bell_circuit, repetitions=1000)
print(f"\ncounts over 1000 shots: {dict(bell_result.histogram(key='result'))}")
print("(only 00 and 11 should appear -- same entanglement as step 5)\n")

# --- Grover's algorithm, 2 qubits, marked item |11> ---
def grover_circuit(marked: str) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(len(marked))
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(*qubits))

    def phase_flip_marked() -> list:
        ops = [cirq.X(q) for q, bit in zip(qubits, marked) if bit == "0"]
        ops.append(cirq.Z(qubits[-1]).controlled_by(*qubits[:-1]))
        ops += [cirq.X(q) for q, bit in zip(qubits, marked) if bit == "0"]
        return ops

    circuit.append(phase_flip_marked())  # oracle

    circuit.append(cirq.H.on_each(*qubits))
    circuit.append(cirq.X.on_each(*qubits))
    circuit.append(cirq.Z(qubits[-1]).controlled_by(*qubits[:-1]))
    circuit.append(cirq.X.on_each(*qubits))
    circuit.append(cirq.H.on_each(*qubits))  # diffusion

    circuit.append(cirq.measure(*qubits, key="result"))
    return circuit


grover = grover_circuit("11")
print("Grover circuit (marked = |11>, 1 iteration -- optimal for N=4):")
print(grover)

grover_result = simulator.run(grover, repetitions=1000)
counts = dict(grover_result.histogram(key="result"))
print(f"\ncounts over 1000 shots: {counts}  (3 == |11>, should dominate)")
