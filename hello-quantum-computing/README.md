# hello-quantum-computing

Goal: quantum computing mechanics from first principles -- qubit state vectors, gates,
measurement, entanglement, and two real algorithms (Deutsch, Grover) built with nothing but
numpy -- then the same mechanics reimplemented on the three major cloud quantum SDKs (Google
Cirq, Microsoft Q#, AWS Braket), all run on their free local simulators so no cloud account is
needed to work through the series.

Each file is a complete, standalone, runnable script -- later files re-declare code from
earlier ones rather than importing across numbered files.

## Phases

- Phase 1 -- mechanics from scratch (numpy only, no quantum SDK): qubit state vectors and
  superposition, single-qubit gates, measurement and the Born rule, multi-qubit systems and
  tensor products, entanglement and Bell states, Deutsch's algorithm, Grover's search algorithm.
- Phase 2 -- the same Bell state and Grover circuits, reimplemented per platform on its free
  local simulator: Google Cirq, Microsoft Q# (`qsharp` package), AWS Braket SDK.
- Phase 3 -- cross-platform vocabulary comparison, and what changes (account, target device,
  billing) to move each platform's local-simulator code onto real cloud QPU hardware.

## Setup

```sh
# from the repo root
python3 -m venv venv && venv/bin/pip install -r hello-quantum-computing/requirements.txt
venv/bin/python hello-quantum-computing/01_qubit_state_vector.py
```

## Notes

- Phase 1 uses full state-vector simulation (explicit numpy arrays and matrices), which is why
  it's capped at 2-3 qubits -- state size doubles per qubit, and this is the same wall real
  quantum hardware exists to get around.
- All three platform SDKs in phase 2 run on local, in-process simulators -- no AWS/Azure/Google
  Cloud account, no billing, no network call. Phase 3 documents (without executing) what each
  vendor's real-hardware path looks like: Google's own processors are limited-access; Microsoft
  and AWS both route to third-party QPU vendors (IonQ, Quantinuum, Rigetti, and others) rather
  than hardware they build themselves.
