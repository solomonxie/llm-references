# hello-transformer

Goal: build the Transformer architecture ("Attention Is All You Need") from raw tensor ops, one
piece at a time — the same "hello world -> full thing" build-up as `hello_webserver` in
`cpp-references`, but for the mechanism under every modern LLM instead of an HTTP server.

Each file is a complete, standalone, runnable script — later files intentionally re-declare
classes from earlier ones (rather than importing across numbered files) so any single file can be
read and run on its own.

| File | Demonstrates |
|---|---|
| `01_tokenize_embed.py` | Char-level tokenizer + `nn.Embedding` lookup |
| `02_positional_encoding.py` | Sinusoidal positional encoding, added to token embeddings |
| `03_scaled_dot_product_attention.py` | `softmax(QK^T / sqrt(d_k))V`, single head, by hand |
| `04_multi_head_attention.py` | Splitting Q/K/V across heads, running attention in parallel, concatenating back |
| `05_feedforward_layernorm.py` | Position-wise FFN + residual connection + LayerNorm ("Add & Norm") |
| `06_encoder_block.py` | One full encoder layer (self-attn + FFN, each Add & Norm'd), stacked N deep |
| `07_decoder_block.py` | Causal self-attention mask + cross-attention to the encoder's output |
| `08_full_transformer.py` | Everything wired end to end: embed -> encoder -> decoder -> vocab logits |
| `09_train_toy_task.py` | Actually training it (teacher forcing, batched) on a toy "reverse this sequence" task, then greedy-decoding for real |

Run any file directly:

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_tokenize_embed.py
# ...
venv/bin/python 09_train_toy_task.py
```

Steps 1-8 print shapes and intermediate values on made-up/random data — there's nothing to
"learn" yet, they're purely about the architecture. Step 9 is the payoff: the same architecture,
actually trained, visibly learning a task from examples.
