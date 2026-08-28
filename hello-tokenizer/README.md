# hello-tokenizer

Goal: how text becomes token ids, built up from scratch -- word-level, then char-level,
then the subword schemes (BPE, byte-level BPE, WordPiece) real models actually use, ending
with the real GPT-2/BERT tokenizers side by side with the from-scratch versions.

Each file is a complete, standalone, runnable script -- later files re-declare code from
earlier ones rather than importing across numbered files.

| File | Demonstrates |
|---|---|
| `01_word_level_tokenizer.py` | Whitespace/punctuation split + vocab -- and the out-of-vocabulary problem |
| `02_char_level_tokenizer.py` | Character-level tokenizer -- no OOV, but much longer sequences |
| `03_byte_pair_encoding.py` | Training BPE merges from scratch: repeatedly merge the most frequent adjacent pair |
| `04_bpe_encode_decode.py` | Applying learned BPE merges to encode/decode new text, including unseen words |
| `05_byte_level_bpe.py` | Byte-level BPE (GPT-2 style) -- a 256-symbol alphabet covers any input, no `<unk>` ever |
| `06_wordpiece.py` | WordPiece (BERT style): `##` continuation pieces + likelihood-based merge scoring |
| `07_huggingface_tokenizers_compare.py` | Real GPT-2 and BERT tokenizers on the same sentences, next to the from-scratch versions |

## Setup

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python 01_word_level_tokenizer.py
```

## Notes

- Steps 1-6 use tiny toy corpora on purpose -- the point is watching every merge decision,
  not vocab quality. Step 7's real tokenizers were trained the same way, just on far more text.
