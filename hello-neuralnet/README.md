# hello-neuralnet

Goal: neural networks from the ground up — a single neuron through backprop, activations,
mini-batch training, arbitrary depth, and convolution — the layer underneath `hello-transformer`'s
architecture. Same "hello world -> full thing" build-up as the rest of this repo. NumPy for the
by-hand calculus (steps 1-2, 6-7), torch from step 3 onward once autograd has been proven to match.

Each file is a complete, standalone, runnable script.

| File | Demonstrates |
|---|---|
| `01_single_neuron.py` | One neuron, sigmoid, gradient descent by hand — learns AND |
| `02_manual_backprop_mlp.py` | A 2-layer MLP, forward + backward by hand, chain rule through a hidden layer — learns XOR (not linearly separable, unlike AND) |
| `03_autograd_equivalent.py` | Cross-checks step 2's hand gradients against `loss.backward()` exactly, then trains the normal way (autograd + `torch.optim`) |
| `04_activation_functions.py` | Sigmoid vs. tanh vs. ReLU gradients, and why deep sigmoid/tanh stacks vanish |
| `05_mini_batch_sgd.py` | Shuffled mini-batches vs. full-batch gradient descent, same data |
| `06_deeper_network_backprop.py` | Backprop generalized to a `Layer`/`DeepMLP` stack of any depth |
| `07_conv_layer_by_hand.py` | 2D convolution (sliding-window dot product) by hand, cross-checked against `torch.nn.functional.conv2d` |
| `08_train_real_digits.py` | The full stack, trained on real handwritten digits (scikit-learn's bundled 8x8 digits set) — 95%+ test accuracy |

Run any file directly:

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_single_neuron.py
# ...
venv/bin/python 08_train_real_digits.py
```

## Notes

- `02` and `06` solve the SAME problem (XOR) at two depths — worth running back to back to see the
  generalized `DeepMLP` reproduce the fixed 2-layer version's result.
- `08` needs no download — `sklearn.datasets.load_digits()` ships the data locally.
