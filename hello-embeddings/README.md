# hello-embeddings

Goal: how words become dense vectors that encode meaning -- from one-hot vectors (which
don't) through co-occurrence statistics, word2vec's Skip-gram and CBOW trained from
scratch, vector arithmetic, and a 2D visualization of the learned space.

Each file is a complete, standalone, runnable script -- later files re-declare code from
earlier ones rather than importing across numbered files.

## Setup

```sh
# from the repo root
python3 -m venv venv && venv/bin/pip install -r hello-embeddings/requirements.txt
venv/bin/python hello-embeddings/01_one_hot_and_cooccurrence.py
```

## Notes

- Steps 3, 4, 6 train on a dozen hand-written sentences -- fast, but too little data for
  reliable analogies; step 6 says so explicitly. Steps 5 and 7 use a larger generated
  corpus with real semantic clusters (royalty/animals/weather/colors) instead.
- `07` writes `embedding_space.png` rather than opening an interactive window.
