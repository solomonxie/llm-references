# hello-neuralnet

Goal: neural networks from the ground up — a single neuron through backprop, activations,
mini-batch training, arbitrary depth, and convolution — the layer underneath `hello-transformer`'s
architecture. Same "hello world -> full thing" build-up as the rest of this repo. NumPy for the
by-hand calculus (steps 1-2, 6-7), torch from step 3 onward once autograd has been proven to match.

Each file is a complete, standalone, runnable script.

Run any file directly:

```sh
# from the repo root
python3 -m venv venv && venv/bin/pip install -r hello-neuralnet/requirements.txt
venv/bin/python hello-neuralnet/01_single_neuron.py
# ...
venv/bin/python hello-neuralnet/08_train_real_digits.py
```

## Notes

- `02` and `06` solve the SAME problem (XOR) at two depths — worth running back to back to see the
  generalized `DeepMLP` reproduce the fixed 2-layer version's result.
- `08` needs no download — `sklearn.datasets.load_digits()` ships the data locally.
