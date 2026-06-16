# $ venv/bin/python hello-neuralnet/02_manual_backprop_mlp.py
#
# Goal: XOR — the textbook example of why a single neuron (step 1) isn't
# enough. XOR is NOT linearly separable: no single straight line in the 2D
# input plane puts both (0,1)/(1,0) [label 1] on one side and both (0,0)/
# (1,1) [label 0] on the other. A hidden layer fixes this by letting the
# network bend space before the final linear decision — this file derives
# and codes that hidden layer's backward pass by hand, via the chain rule,
# with no autograd at all (step 3 repeats this exact problem WITH autograd,
# to cross-check these gradients are right).
# Step 2: A 2-layer MLP, forward + backward by hand, chain rule through a hidden layer -- learns XOR

import numpy as np

np.random.seed(0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(sigmoid_output: np.ndarray) -> np.ndarray:
    return sigmoid_output * (1 - sigmoid_output)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)  # XOR

n_hidden = 4
W1 = np.random.randn(2, n_hidden) * 0.5  # input -> hidden
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, 1) * 0.5  # hidden -> output
b2 = np.zeros((1, 1))
learning_rate = 1.0

for epoch in range(5000):
    # ---- forward pass ----
    z1 = X @ W1 + b1          # (4, n_hidden)
    a1 = sigmoid(z1)          # hidden layer's activation
    z2 = a1 @ W2 + b2         # (4, 1)
    a2 = sigmoid(z2)          # network's final output

    loss = np.mean((a2 - y) ** 2)

    # ---- backward pass ----
    # Same chain rule as step 1's single neuron, for the OUTPUT layer:
    d_loss_d_a2 = 2 * (a2 - y) / len(X)
    d_a2_d_z2 = sigmoid_derivative(a2)
    d_loss_d_z2 = d_loss_d_a2 * d_a2_d_z2         # (4, 1) — "how wrong was each example's output, pre-activation"

    grad_W2 = a1.T @ d_loss_d_z2                  # (n_hidden, 1)
    grad_b2 = np.sum(d_loss_d_z2, axis=0, keepdims=True)

    # The NEW step vs. step 1: propagate that error BACKWARD through W2 to
    # find "how wrong was each hidden unit's output" — this is literally
    # what "backpropagation" means, the error signal flowing backward
    # through the same weights the forward pass used, layer by layer.
    d_loss_d_a1 = d_loss_d_z2 @ W2.T              # (4, n_hidden)
    d_a1_d_z1 = sigmoid_derivative(a1)
    d_loss_d_z1 = d_loss_d_a1 * d_a1_d_z1

    grad_W1 = X.T @ d_loss_d_z1                   # (2, n_hidden)
    grad_b1 = np.sum(d_loss_d_z1, axis=0, keepdims=True)

    # ---- gradient descent step ----
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1

    if epoch % 1000 == 0:
        print(f"epoch {epoch:>4}  loss {loss:.4f}")

print(f"\nfinal loss: {loss:.4f}")
print("\npredictions after training:")
for inputs, label in zip(X, y):
    a1 = sigmoid(inputs @ W1 + b1)
    a2 = sigmoid(a1 @ W2 + b2)
    print(f"  {inputs.astype(int)} -> {a2.item():.3f}  (label: {int(label.item())})")

# A single neuron (step 1) plateaus around loss=0.25 on this exact problem —
# worth trying: swap this file's two-layer network for step 1's single
# neuron on XOR's data and watch it fail to separate the classes at all.
