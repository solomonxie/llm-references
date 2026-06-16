# $ venv/bin/python hello-neuralnet/04_activation_functions.py
#
# Goal: why ReLU displaced sigmoid/tanh as the default hidden-layer
# activation. Backprop (steps 2-3) multiplies each layer's local gradient
# together, chain-rule style, to get the gradient at any earlier layer — a
# gradient signal that has to pass through N layers gets multiplied by N
# activation derivatives along the way. Sigmoid and tanh's derivatives are
# ALWAYS < 1 (often much less, away from x=0), so that product shrinks
# toward zero as N grows: "vanishing gradients," early layers barely learn.
# ReLU's derivative is exactly 1 for any positive input — multiplying by 1
# repeatedly doesn't shrink anything.
# Step 4: Sigmoid vs. tanh vs. ReLU gradients, and why deep sigmoid/tanh stacks vanish

import numpy as np

x = np.linspace(-5, 5, 11)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


print(f"{'x':>6} {'sigmoid':>10} {'sigmoid_grad':>13} {'tanh_grad':>11} {'relu_grad':>10}")
for xi in x:
    print(f"{xi:>6.1f} {sigmoid(xi):>10.4f} {sigmoid_grad(xi):>13.4f} {tanh_grad(xi):>11.4f} {relu_grad(xi):>10.1f}")

print(f"\nmax possible gradient: sigmoid={sigmoid_grad(0):.4f}  tanh={tanh_grad(0):.4f}  relu=1.0")

# Simulate backprop's gradient product through a chain of N layers, all with
# the SAME local derivative g (a simplification — real gradients vary per
# layer/input — but it isolates exactly the effect described above).
print("\ngradient reaching layer 0, after passing through N layers (each contributing factor g):")
print(f"{'N layers':>10} {'g=0.25 (sigmoid-ish)':>22} {'g=0.5 (tanh-ish)':>18} {'g=1.0 (relu, active)':>22}")
for n_layers in [1, 5, 10, 20, 50]:
    g_sigmoid = 0.25**n_layers   # sigmoid's derivative peaks at 0.25 (at x=0) — this is close to a best case
    g_tanh = 0.5**n_layers       # tanh's derivative peaks at 1.0, but is well below that away from 0 — 0.5 is typical
    g_relu = 1.0**n_layers       # ReLU's derivative is exactly 1 wherever the unit is active (input > 0)
    print(f"{n_layers:>10} {g_sigmoid:>22.2e} {g_tanh:>18.2e} {g_relu:>22.4f}")

# By 20 layers, the sigmoid-chain gradient is already 9 orders of magnitude
# smaller than the ReLU-chain one — for a deep network, that's the
# difference between "this layer's weights update meaningfully" and "this
# layer's weights are effectively frozen." (ReLU trades this for its own
# failure mode — a unit stuck outputting 0 has grad_relu=0 too, permanently
# "dead" — not covered here, but worth knowing exists.)
