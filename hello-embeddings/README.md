# hello-embeddings

Goal: how words become dense vectors that encode meaning -- from one-hot vectors (which
don't) through co-occurrence statistics, word2vec's Skip-gram and CBOW trained from
scratch, vector arithmetic, and a 2D visualization of the learned space.

Each file is a complete, standalone, runnable script -- later files re-declare code from
earlier ones rather than importing across numbered files.

| File | Demonstrates |
|---|---|
| `01_one_hot_and_cooccurrence.py` | One-hot vectors' blindness to meaning, and a co-occurrence matrix |
| `02_cosine_similarity.py` | Cosine similarity, applied to co-occurrence rows |
| `03_skipgram_negative_sampling.py` | word2vec Skip-gram + negative sampling, trained from scratch |
| `04_cbow.py` | CBOW -- the other word2vec objective, averaged context predicts the center word |
| `05_train_on_real_corpus.py` | The same training scaled to a larger, less-curated corpus |
| `06_analogy_arithmetic.py` | Vector arithmetic on trained embeddings (`king - man + woman ~= queen`) |
| `07_visualize_embedding_space.py` | PCA projection to 2D, plotted with cluster labels |

## Setup

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_one_hot_and_cooccurrence.py
```

## Notes

- Steps 3, 4, 6 train on a dozen hand-written sentences -- fast, but too little data for
  reliable analogies; step 6 says so explicitly. Steps 5 and 7 use a larger generated
  corpus with real semantic clusters (royalty/animals/weather/colors) instead.
- `07` writes `embedding_space.png` rather than opening an interactive window.
