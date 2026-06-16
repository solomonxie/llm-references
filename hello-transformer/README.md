# hello-transformer

Goal: build the Transformer architecture ("Attention Is All You Need") from raw tensor ops, one
piece at a time — the same "hello world -> full thing" build-up as `hello_webserver` in
`cpp-references`, but for the mechanism under every modern LLM instead of an HTTP server.

Each file is a complete, standalone, runnable script — later files intentionally re-declare
classes from earlier ones (rather than importing across numbered files) so any single file can be
read and run on its own.

Run any file directly:

```sh
# from the repo root
python3 -m venv venv && venv/bin/pip install -r hello-transformer/requirements.txt
venv/bin/python hello-transformer/01_tokenize_embed.py
# ...
venv/bin/python hello-transformer/09_train_toy_task.py
```

Steps 1-8 print shapes and intermediate values on made-up/random data — there's nothing to
"learn" yet, they're purely about the architecture. Step 9 is the payoff: the same architecture,
actually trained, visibly learning a task from examples.
