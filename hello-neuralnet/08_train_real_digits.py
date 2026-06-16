# $ venv/bin/python hello-neuralnet/08_train_real_digits.py
#
# Goal: everything so far (a neuron, backprop, mini-batches, deeper stacks,
# convolution) applied to real data instead of a toy 4-point XOR set — 1,797
# real handwritten digit images (8x8 grayscale, scikit-learn's bundled
# copy of a subset of UCI's digits dataset, no download needed), classified
# 0-9. Same shapes (a Linear -> ReLU -> Linear stack from step 5), same
# training loop (mini-batch SGD, step 5) — just real, messy, human-written
# input instead of a clean synthetic one.
# Step 8: The full stack, trained on real handwritten digits -- 95%+ test accuracy

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

np.random.seed(0)
torch.manual_seed(0)

digits = load_digits()  # .data: (1797, 64) flattened 8x8 images: .target: (1797,) labels 0-9
print(f"dataset: {digits.data.shape[0]} images, {digits.data.shape[1]} pixels each, {len(set(digits.target))} classes")

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)

# Pixel values are 0-16 (not 0-255 — this dataset's own scale) — normalizing
# to roughly [0, 1] helps gradient descent converge; large, unnormalized
# inputs push early logits/gradients to extreme values.
X_train = torch.tensor(X_train / 16.0, dtype=torch.float32)
X_test = torch.tensor(X_test / 16.0, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = torch.nn.Sequential(
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 10),  # 10 output logits — one score per digit class
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def train_epoch(batch_size: int = 32) -> float:
    permutation = torch.randperm(len(X_train))
    total_loss = 0.0
    for start in range(0, len(X_train), batch_size):
        idx = permutation[start : start + batch_size]
        logits = model(X_train[idx])
        loss = torch.nn.functional.cross_entropy(logits, y_train[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (len(X_train) // batch_size + 1)


for epoch in range(1, 21):
    avg_loss = train_epoch()
    if epoch % 4 == 0 or epoch == 1:
        with torch.no_grad():
            train_acc = accuracy(model(X_train), y_train)
            test_acc = accuracy(model(X_test), y_test)
        print(f"epoch {epoch:>2}  loss {avg_loss:.4f}  train acc {train_acc:.1%}  test acc {test_acc:.1%}")

with torch.no_grad():
    final_test_acc = accuracy(model(X_test), y_test)
print(f"\nfinal test accuracy: {final_test_acc:.1%}  (on {len(X_test)} held-out images the model never trained on)")

# A quick look at what it actually gets wrong — the interesting cases,
# unlike the XOR/blob toy problems, which had no ambiguity at all.
with torch.no_grad():
    predictions = model(X_test).argmax(dim=-1)
wrong = (predictions != y_test).nonzero().squeeze(-1)
print(f"\n{len(wrong)} misclassified out of {len(X_test)}:")
for i in wrong[:5].tolist():
    print(f"  predicted {predictions[i].item()}, actually {y_test[i].item()}")
