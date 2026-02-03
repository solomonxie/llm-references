# $ venv/bin/python 07_conv_layer_by_hand.py
#
# Goal: convolution, the operation CNNs are built from — a small learned
# grid of numbers (a "kernel"/"filter") slides across an image, and at each
# position computes ONE number: the elementwise product of the kernel with
# whatever patch of the image it's currently over, summed. Unlike every
# layer so far (every input unit connects to every output unit, "dense"/
# "fully-connected"), a conv layer's kernel is small and REUSED at every
# position — the same edge-detector, say, works the same way whether the
# edge is in the top-left or bottom-right of the image.
# Step 7: 2D convolution (sliding-window dot product) by hand, cross-checked against torch.nn.functional.conv2d

import numpy as np
import torch

image = np.array(
    [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
    ],
    dtype=float,
)  # a 6x6 image: solid black on the left, solid white on the right — one vertical edge, right in the middle

# A vertical-edge-detecting kernel: strongly positive on the right column,
# strongly negative on the left — "high output" means "bright pixels to
# my right, dark pixels to my left," i.e. sitting right on a left-to-right edge.
vertical_edge_kernel = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ],
    dtype=float,
)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """"Valid" convolution — the kernel only slides to positions fully
    inside the image, so the output is smaller than the input by
    (kernel_size - 1) in each dimension. (Technically this is CROSS-
    correlation, not mathematical convolution, which flips the kernel first
    — deep learning frameworks all call this "convolution" regardless, so
    this file does too.)"""
    kh, kw = kernel.shape
    ih, iw = image.shape
    output = np.zeros((ih - kh + 1, iw - kw + 1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            patch = image[i : i + kh, j : j + kw]
            output[i, j] = np.sum(patch * kernel)  # elementwise multiply, then sum — one number per position
    return output


result = convolve2d(image, vertical_edge_kernel)
print("input image (0=black, 1=white):")
print(image)
print("\nvertical-edge kernel:")
print(vertical_edge_kernel)
print("\nconvolution output (peaks exactly where the edge is):")
print(result)

# Cross-check against torch's real conv2d, which needs an explicit
# (batch, channels, height, width) shape even for one grayscale image.
image_t = torch.tensor(image).unsqueeze(0).unsqueeze(0)      # (1, 1, 6, 6)
kernel_t = torch.tensor(vertical_edge_kernel).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
built_in_result = torch.nn.functional.conv2d(image_t, kernel_t).squeeze().numpy()

print(f"\nmatches torch.nn.functional.conv2d: {np.allclose(result, built_in_result)}")

# A real conv LAYER (nn.Conv2d) learns many such kernels at once (one per
# output "channel") via backprop, exactly like every weight in steps 1-6 —
# the kernel values here were hand-picked to detect a specific pattern;
# training would instead let gradient descent discover whatever kernels
# reduce the loss, which for real images turns out to rediscover edge/
# corner/texture detectors like this one on its own, in the first layer.
