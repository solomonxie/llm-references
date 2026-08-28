# hello-rag

Goal: Retrieval-Augmented Generation, past the single-file version in `hello-langchain/07` — each
piece of a real RAG pipeline on its own, then composed, then hardened. Runs against local Ollama
models (`nomic-embed-text` for embeddings, `llama3.2:3b` for generation), no API key or cost.

Each file is a complete, standalone, runnable script.

## Setup

```sh
ollama serve &                        # if not already running
ollama pull llama3.2:3b
ollama pull nomic-embed-text

python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_chunking_strategies.py
```

## Notes

- `01` and `06` are dependency-light on purpose — chunking and keyword scoring are hand-rolled
  rather than reaching for `langchain_text_splitters`/`rank_bm25`, so the mechanism is visible
  before reaching for the library shortcut (the README notes what the "real" library call would be).
- Read in order — `04` reuses `01`'s chunker, `07`/`08` reuse `03`'s retrieval pattern.
