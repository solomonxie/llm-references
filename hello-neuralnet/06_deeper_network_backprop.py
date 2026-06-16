# $ venv/bin/python hello-neuralnet/06_deeper_network_backprop.py
#
# Goal: step 2's by-hand backprop, generalized from exactly 2 layers (hardcoded
# W1/W2) to an arbitrary stack — a `DeepMLP` class holding a LIST of layers,
# whose forward pass runs the list forward and whose backward pass runs the
# SAME list backward, propagating one layer's error signal into the next.
# This loop, at any depth, is the entire content of "backpropagation" — step
# 2 just wrote it out for the N=2 special case first.
# Step 6: Backprop generalized to a Layer/DeepMLP stack of any depth

import numpy as np

np.random.seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.W = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)  # scaled init — keeps activations from exploding as depth grows
        self.b = np.zeros((1, n_out))
        self.input_cache = None   # this layer's input, saved for the backward pass's grad_W = input.T @ ...
        self.output_cache = None  # this layer's sigmoid output, saved for sigmoid_derivative(output)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        z = x @ self.W + self.b
        self.output_cache = sigmoid(z)
        return self.output_cache

    def backward(self, d_loss_d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """Takes this layer's OWN error signal (gradient of the loss w.r.t.
        its output), updates its own weights, and returns the error signal
        for the layer BEFORE it — the recursive step that makes stacking
        arbitrarily many of these correct."""
        d_output_d_z = self.output_cache * (1 - self.output_cache)
        d_loss_d_z = d_loss_d_output * d_output_d_z

        grad_W = self.input_cache.T @ d_loss_d_z
        grad_b = np.sum(d_loss_d_z, axis=0, keepdims=True)
        d_loss_d_input = d_loss_d_z @ self.W.T  # this layer's contribution to "what should the PREVIOUS layer have output"

        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b
        return d_loss_d_input


class DeepMLP:
    def __init__(self, layer_sizes: list[int]):
        # layer_sizes = [n_in, hidden_1, hidden_2, ..., n_out] — as many
        # Layers as there are consecutive pairs, any depth.
        self.layers = [Layer(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_loss_d_output: np.ndarray, learning_rate: float) -> None:
        # The one line that IS backpropagation: walk the layers in REVERSE,
        # feeding each layer's returned "what the previous layer should have
        # output" gradient straight into the previous layer's backward call.
        for layer in reversed(self.layers):
            d_loss_d_output = layer.backward(d_loss_d_output, learning_rate)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)  # XOR again — same problem as step 2, deeper solution

# 4 layers deep (2 -> 8 -> 8 -> 4 -> 1) instead of step 2's fixed 2 -> 4 -> 1
# — the SAME Layer/DeepMLP code handles this with no changes.
model = DeepMLP([2, 8, 8, 4, 1])
learning_rate = 1.0

for epoch in range(5000):
    prediction = model.forward(X)
    loss = np.mean((prediction - y) ** 2)
    d_loss_d_prediction = 2 * (prediction - y) / len(X)
    model.backward(d_loss_d_prediction, learning_rate)

    if epoch % 1000 == 0:
        print(f"epoch {epoch:>4}  loss {loss:.4f}")

print(f"\nfinal loss: {loss:.4f}  ({len(model.layers)} layers deep)")
print("\npredictions:")
for inputs, label in zip(X, y):
    pred = model.forward(inputs.reshape(1, -1))
    print(f"  {inputs.astype(int)} -> {pred.item():.3f}  (label: {int(label.item())})")
