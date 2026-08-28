# hello-tokenizer

Goal: how text becomes token ids, built up from scratch -- word-level, then char-level,
then the subword schemes (BPE, byte-level BPE, WordPiece) real models actually use, ending
with the real GPT-2/BERT tokenizers side by side with the from-scratch versions.

Each file is a complete, standalone, runnable script -- later files re-declare code from
earlier ones rather than importing across numbered files.

## Setup

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_word_level_tokenizer.py
```

## Notes

- Steps 1-6 use tiny toy corpora on purpose -- the point is watching every merge decision,
  not vocab quality. Step 7's real tokenizers were trained the same way, just on far more text.
