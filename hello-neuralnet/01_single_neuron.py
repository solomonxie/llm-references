# $ venv/bin/python hello-neuralnet/01_single_neuron.py
#
# Goal: the smallest possible unit everything else in this folder builds on
# — one neuron. It computes a weighted sum of its inputs plus a bias, then
# squashes that through an activation function (sigmoid here: maps any real
# number to (0, 1), readable as "probability of class 1"). "Training" is
# nothing more than nudging the weights, via gradient descent, so the
# neuron's outputs get closer to the labels it's shown.
# Step 1: One neuron, sigmoid, gradient descent by hand -- learns AND

import numpy as np

np.random.seed(0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(sigmoid_output: np.ndarray) -> np.ndarray:
    # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) — conveniently
    # expressible in terms of the sigmoid's OWN output, not x itself, which
    # is why every neuron below caches its activation instead of recomputing.
    return sigmoid_output * (1 - sigmoid_output)


# Training data: logical AND. Two binary inputs, one binary label — a
# single neuron CAN learn this because AND is linearly separable (one
# straight line in the 2D input plane separates the (1,1) case from the
# other three). XOR (step 2) is the classic case where it can't.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [0], [0], [1]], dtype=float)

weights = np.random.randn(2, 1) * 0.1
bias = np.zeros((1, 1))
learning_rate = 0.5

for epoch in range(2000):
    # Forward pass: one neuron, all 4 examples at once (X is (4, 2), so
    # X @ weights is (4, 1) — one weighted sum per example, batched).
    z = X @ weights + bias
    prediction = sigmoid(z)

    # Mean squared error: average((prediction - label)^2) over the batch.
    error = prediction - y
    loss = np.mean(error**2)

    # Backward pass, by hand, via the chain rule:
    #   dLoss/dPrediction = 2 * error / N              (derivative of MSE)
    #   dPrediction/dz     = sigmoid_derivative(prediction)
    #   dz/dWeights         = X (transposed to line up shapes)
    #   dz/dBias            = 1
    d_loss_d_prediction = 2 * error / len(X)
    d_prediction_d_z = sigmoid_derivative(prediction)
    d_loss_d_z = d_loss_d_prediction * d_prediction_d_z  # (4, 1), one gradient per example

    grad_weights = X.T @ d_loss_d_z  # (2, 1) — chain rule collapses the batch dim via this matmul
    grad_bias = np.sum(d_loss_d_z, axis=0, keepdims=True)

    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias

    if epoch % 500 == 0:
        print(f"epoch {epoch:>4}  loss {loss:.4f}")

print(f"\nfinal weights: {weights.ravel()}, bias: {bias.ravel()}")
print("\npredictions after training:")
for inputs, label in zip(X, y):
    pred = sigmoid(inputs @ weights + bias)
    print(f"  {inputs.astype(int)} -> {pred.item():.3f}  (label: {int(label.item())})")
