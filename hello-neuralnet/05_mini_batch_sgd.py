# $ venv/bin/python 05_mini_batch_sgd.py
#
# Goal: steps 1-3 computed the gradient over the WHOLE dataset every step
# ("batch" gradient descent) — fine for 4 XOR examples, doesn't scale to
# millions. Mini-batch SGD instead: shuffle the data, split it into small
# batches, take one gradient step per batch (using only that batch's
# examples), and call one full pass over all batches an "epoch." This also
# trades a noisier gradient estimate (each batch is a sample, not the true
# gradient) for far more weight updates per unit of data seen.
# Step 5: Shuffled mini-batches vs. full-batch gradient descent, same data

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)


def make_blobs(n_per_class: int, centers: list[tuple[float, float]], std: float) -> tuple[np.ndarray, np.ndarray]:
    """Hand-rolled stand-in for sklearn's make_blobs — sample n_per_class
    Gaussian-scattered points around each center, label them by which
    center they came from."""
    X, y = [], []
    for label, (cx, cy) in enumerate(centers):
        points = np.random.randn(n_per_class, 2) * std + np.array([cx, cy])
        X.append(points)
        y.extend([label] * n_per_class)
    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64)


X, y = make_blobs(n_per_class=150, centers=[(-2, -2), (2, 2), (-2, 2)], std=0.8)
X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)

model = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def train_epoch(batch_size: int) -> tuple[float, int]:
    """One full pass over the (shuffled) dataset, one gradient step per
    batch. Returns the average loss and how many gradient steps that took —
    the whole point being: more, smaller steps per epoch than batch GD's one
    giant step."""
    permutation = torch.randperm(len(X_t))  # shuffling each epoch avoids the same batch composition every time
    total_loss, steps = 0.0, 0
    for start in range(0, len(X_t), batch_size):
        batch_indices = permutation[start : start + batch_size]
        batch_X, batch_y = X_t[batch_indices], y_t[batch_indices]

        logits = model(batch_X)
        loss = torch.nn.functional.cross_entropy(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
    return total_loss / steps, steps


print(f"dataset: {len(X_t)} points, 3 classes")
for epoch in range(10):
    avg_loss, steps_taken = train_epoch(batch_size=32)
    if epoch % 2 == 0:
        acc = accuracy(model(X_t), y_t)
        print(f"epoch {epoch}: {steps_taken} gradient steps, avg loss {avg_loss:.4f}, accuracy {acc:.1%}")

print(f"\nfinal accuracy: {accuracy(model(X_t), y_t):.1%}")

# Same data, ONE giant batch (= all 450 points) per epoch — batch gradient
# descent, step 1-3's style, for comparison. Fewer, "cleaner" (lower-
# variance) updates, but far fewer of them for the same amount of data seen.
print("\nfor comparison — full-batch gradient descent, same data, same epoch count:")
torch.manual_seed(0)
model2 = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
for epoch in range(10):
    logits = model2(X_t)
    loss = torch.nn.functional.cross_entropy(logits, y_t)
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    if epoch % 2 == 0:
        print(f"epoch {epoch}: 1 gradient step, loss {loss.item():.4f}, accuracy {accuracy(logits, y_t):.1%}")
