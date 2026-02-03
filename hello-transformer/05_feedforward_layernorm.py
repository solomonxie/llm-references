# $ venv/bin/python hello-transformer/05_feedforward_layernorm.py
#
# Goal: the other two ingredients every transformer sublayer needs, besides
# attention itself:
#   - a position-wise feedforward network (FFN) — the same tiny 2-layer MLP
#     applied independently to every token's vector; attention mixes
#     information *across* tokens, the FFN processes each token *on its own*
#   - "Add & Norm" — a residual connection (`x + Sublayer(x)`) plus
#     LayerNorm, wrapped around both attention and the FFN. This is what
#     makes deep stacks (step 6) trainable at all — without the residual,
#     gradients have to flow through every sublayer's transform to reach
#     early layers; with it, there's always a direct path.
# Step 5: Position-wise FFN + residual connection + LayerNorm (Add & Norm)

import torch

torch.manual_seed(0)

seq_len = 5
d_model = 8
d_ff = 32  # FFN's inner width — conventionally 4x d_model


class FeedForward(torch.nn.Module):
    """Linear -> ReLU -> Linear, applied independently to each token's row.
    "Position-wise" just means there's no mixing across the seq_len dimension
    here at all — contrast with attention, which is entirely about mixing
    across it."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(d_ff, d_model),  # back down to d_model so the residual add below type-checks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AddNorm(torch.nn.Module):
    """Wraps a sublayer (attention or FFN) with `LayerNorm(x + sublayer(x))`.
    Passing the sublayer itself in lets step 6 reuse this identically for
    both the attention sublayer and the FFN sublayer."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        #   x ──────────────────────────┐
        #   │                           │ (residual / "skip connection")
        #   ▼                           │
        #   sublayer(x)                 │
        #   │                           │
        #   ▼                           ▼
        #   ────────────── + ───────────
        #                   │
        #                   ▼
        #               LayerNorm
        return self.norm(x + sublayer_output)


x = torch.randn(seq_len, d_model)
ffn = FeedForward(d_model, d_ff)
add_norm = AddNorm(d_model)

ffn_out = ffn(x)
print(f"FFN output shape: {tuple(ffn_out.shape)}  (same as input — FFN never changes seq_len or d_model)")

normed = add_norm(x, ffn_out)
print(f"after Add & Norm:  {tuple(normed.shape)}")
print(f"per-row mean ~0, std ~1 after LayerNorm: mean={normed[0].mean():.4f}, std={normed[0].std():.4f}")

# LayerNorm normalizes *across d_model*, independently per token (contrast
# with BatchNorm, which normalizes across the batch) — appropriate here since
# sequences in a batch can have different lengths and shouldn't affect each
# other's normalization statistics.
