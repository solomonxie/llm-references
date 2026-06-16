# $ venv/bin/python hello-quantum-computing/09_microsoft_qsharp_bell_and_grover.py
#
# Goal: the same Bell state and Grover search, now in Q# -- Microsoft's
# dedicated quantum programming language, called from Python via the
# `qsharp` package's local full-state simulator (free, no Azure account).
# Q# reads more like a real programming language than Cirq/Braket's
# circuit-builder style: `use` allocates qubits, functions read top to
# bottom as the actual circuit.
# Step 9: Microsoft Q# (qsharp package) -- Bell state + Grover's algorithm

import qsharp

# --- Bell state ---
qsharp.eval("""
    operation BellPair() : (Result, Result) {
        use q0 = Qubit();
        use q1 = Qubit();
        H(q0);
        CNOT(q0, q1);
        let r0 = M(q0);
        let r1 = M(q1);
        Reset(q0);
        Reset(q1);
        return (r0, r1);
    }
""")

counts = {"00": 0, "01": 0, "10": 0, "11": 0}
for _ in range(1000):
    r0, r1 = qsharp.eval("BellPair()")
    key = f"{int(r0 == qsharp.Result.One)}{int(r1 == qsharp.Result.One)}"
    counts[key] += 1
print(f"Bell state counts over 1000 shots: {counts}")
print("(only 00 and 11 should appear -- same entanglement as step 5)\n")

# --- Grover's algorithm, 2 qubits, marked item |11> ---
# CZ marks |11> directly (phase -1 exactly when both qubits are |1>), so
# this oracle needs no X-sandwich, unlike step 8's general-target version.
qsharp.eval("""
    operation ReflectAboutUniform(qs : Qubit[]) : Unit {
        within {
            ApplyToEachA(H, qs);
            ApplyToEachA(X, qs);
        } apply {
            Controlled Z(qs[0..Length(qs) - 2], qs[Length(qs) - 1]);
        }
    }

    operation GroverMarked11() : (Result, Result) {
        use qs = Qubit[2];
        ApplyToEach(H, qs);
        Controlled Z([qs[0]], qs[1]);  // oracle: marks |11>
        ReflectAboutUniform(qs);       // diffusion
        let r0 = M(qs[0]);
        let r1 = M(qs[1]);
        ResetAll(qs);
        return (r0, r1);
    }
""")

grover_counts = {"00": 0, "01": 0, "10": 0, "11": 0}
for _ in range(1000):
    r0, r1 = qsharp.eval("GroverMarked11()")
    key = f"{int(r0 == qsharp.Result.One)}{int(r1 == qsharp.Result.One)}"
    grover_counts[key] += 1
print(f"Grover counts over 1000 shots: {grover_counts}  (11 should dominate)")
