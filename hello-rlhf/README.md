# hello-rlhf

Goal: reward modeling and policy optimization from preference data -- Bradley-Terry reward
models, PPO, and DPO, from scratch, on a toy sequence task with a hidden ground-truth
preference rule (more of the symbol `A` is better) so every step's result is checkable.

Each file is a complete, standalone, runnable script -- later files re-declare code from
earlier ones rather than importing across numbered files.

## Setup

```sh
# from the repo root
python3 -m venv venv && venv/bin/pip install -r hello-rlhf/requirements.txt
venv/bin/python hello-rlhf/01_toy_preference_dataset.py
```

## Notes

- The policy throughout is a table of per-position logits, not an autoregressive network --
  the point of this series is the RLHF loss mechanics (Bradley-Terry, PPO's clip/KL, DPO's
  direct loss), not model architecture; see `hello-transformer` for that side.
- `04`'s KL penalty against the reference policy is computed in closed form (both are simple
  categorical tables) rather than the sampled-estimator most real implementations need.
