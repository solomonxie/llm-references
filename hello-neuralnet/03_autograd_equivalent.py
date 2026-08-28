# $ venv/bin/python 03_autograd_equivalent.py
#
# Goal: prove step 2's hand-derived gradients were actually right, then
# never derive them by hand again. Autograd (`loss.backward()`) computes the
# exact same chain-rule derivatives step 2 worked out on paper — the first
# part of this file checks that claim directly, gradient by gradient, on
# IDENTICAL weights and data; the second part just trains XOR the normal
# way everyone actually does it.
# Step 3: Cross-checks step 2's hand gradients against loss.backward() exactly, then trains the normal way

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

X_np = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_np = np.array([[0], [1], [1], [0]], dtype=float)

n_hidden = 4
W1_init = np.random.randn(2, n_hidden) * 0.5
W2_init = np.random.randn(n_hidden, 1) * 0.5


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---- by-hand forward + backward (identical math to 02_manual_backprop_mlp.py) ----
W1, b1 = W1_init.copy(), np.zeros((1, n_hidden))
W2, b2 = W2_init.copy(), np.zeros((1, 1))

z1 = X_np @ W1 + b1
a1 = sigmoid_np(z1)
z2 = a1 @ W2 + b2
a2 = sigmoid_np(z2)

d_loss_d_z2 = (2 * (a2 - y_np) / len(X_np)) * (a2 * (1 - a2))
grad_W2_by_hand = a1.T @ d_loss_d_z2
d_loss_d_z1 = (d_loss_d_z2 @ W2.T) * (a1 * (1 - a1))
grad_W1_by_hand = X_np.T @ d_loss_d_z1

# ---- same network, same weights, autograd instead ----
X = torch.tensor(X_np, dtype=torch.float64)
y = torch.tensor(y_np, dtype=torch.float64)
W1_t = torch.tensor(W1_init.copy(), dtype=torch.float64, requires_grad=True)
b1_t = torch.zeros(1, n_hidden, dtype=torch.float64, requires_grad=True)
W2_t = torch.tensor(W2_init.copy(), dtype=torch.float64, requires_grad=True)
b2_t = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)

a1_t = torch.sigmoid(X @ W1_t + b1_t)
a2_t = torch.sigmoid(a1_t @ W2_t + b2_t)
loss = torch.mean((a2_t - y) ** 2)
loss.backward()  # computes W1_t.grad, W2_t.grad, etc. — the ENTIRE backward pass, one call

print("gradient comparison (by-hand vs. autograd), same weights, same data:")
print(f"  grad_W2 max abs difference: {np.abs(grad_W2_by_hand - W2_t.grad.numpy()).max():.10f}")
print(f"  grad_W1 max abs difference: {np.abs(grad_W1_by_hand - W1_t.grad.numpy()).max():.10f}")
print("  (should both be ~0 — same math, two different ways of computing it)")

# ---- now train the normal way: autograd + an optimizer, no manual .grad math at all ----
print("\ntraining XOR with autograd + torch.optim (no hand-derived gradients anywhere):")
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 1), torch.nn.Sigmoid())
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
X32, y32 = X.float(), y.float()

for epoch in range(5000):
    prediction = model(X32)
    loss = torch.nn.functional.mse_loss(prediction, y32)

    optimizer.zero_grad()  # gradients accumulate by default — must clear before each backward()
    loss.backward()        # fills every parameter's .grad
    optimizer.step()       # applies param -= lr * param.grad, for every parameter, automatically

    if epoch % 1000 == 0:
        print(f"  epoch {epoch:>4}  loss {loss.item():.4f}")

print("\npredictions:")
with torch.no_grad():
    for inputs, label in zip(X32, y32):
        pred = model(inputs.unsqueeze(0))
        print(f"  {inputs.int().tolist()} -> {pred.item():.3f}  (label: {int(label.item())})")
