# hello-rag

Goal: Retrieval-Augmented Generation, past the single-file version in `hello-langchain/07` — each
piece of a real RAG pipeline on its own, then composed, then hardened. Runs against local Ollama
models (`nomic-embed-text` for embeddings, `llama3.2:3b` for generation), no API key or cost.

Each file is a complete, standalone, runnable script.

| File | Demonstrates |
|---|---|
| `01_chunking_strategies.py` | Fixed-size vs. sentence-aware vs. overlapping chunking, and why boundaries matter |
| `02_embeddings_similarity.py` | Cosine similarity by hand — the mechanism retrieval ranks by |
| `03_vector_store_retrieval.py` | `InMemoryVectorStore` — the general "k nearest vectors" interface |
| `04_basic_rag_pipeline.py` | Chunk -> embed -> index -> retrieve -> prompt -> generate, end to end |
| `05_metadata_filtering.py` | Filtering the candidate pool by metadata (source, date, permissions) before/alongside similarity |
| `06_hybrid_search.py` | Vector + keyword search combined (reciprocal rank fusion) — catches exact matches embeddings miss |
| `07_reranking.py` | Cheap wide retrieval, then an LLM re-scores/re-sorts the candidates for precision |
| `08_evaluation.py` | A minimal labeled eval set, checking retrieval hit-rate and generation accuracy separately |

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
