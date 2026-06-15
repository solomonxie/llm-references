# hello-llm-training

Goal: pretraining mechanics -- how a language model goes from random weights to generating
plausible text, on a tiny decoder-only transformer and a toy corpus (small enough to train on
CPU in seconds). Follows `hello-transformer` (the architecture) and precedes `hello-finetune`
(which starts from an already-pretrained model); this is the step in between -- how that
starting point gets made in the first place.

Each file is a complete, standalone, runnable script -- later files re-declare code from
earlier ones rather than importing across numbered files.

## Setup

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_toy_corpus_and_tokenizer.py
```

## Steps

| File | Adds |
| --- | --- |
| `01_toy_corpus_and_tokenizer.py` | Toy corpus + char-level tokenizer, encode/decode round trip |
| `02_init_small_transformer.py` | Decoder-only GPT (causal self-attn, learned positional embeddings), random init, untrained (garbage) generation |
| `03_training_loop.py` | Next-char prediction as (input, target) batches from random corpus crops, cross-entropy loss, AdamW |
| `04_loss_curve_and_sampling.py` | LR warmup + cosine decay, loss curve, temperature-sampled generation |

## Notes

- The toy corpus is deliberately tiny and repetitive -- the point is watching the loss actually
  fall and the output actually change, not building a capable model.
- Step 2's decoder-only shape (no encoder, no cross-attention) is what GPT-family models use;
  `hello-transformer` builds the encoder-decoder shape from "Attention Is All You Need" instead.
