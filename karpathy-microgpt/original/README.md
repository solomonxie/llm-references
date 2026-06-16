# original

Verbatim, unmodified vendor copy of Andrej Karpathy's `microgpt.py` -- not written or edited by
this repo, kept here only as a reference to compare the numbered steps in `../` against.

- Source: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
- Blog post: https://karpathy.github.io/2026/02/12/microgpt/
- Vendored commit: `14fb038816c7aae0bb9342c2dbf1a51dd134a5ff` (2026-02-16)
- License: none declared upstream at the time of vendoring -- a gist comment asked for one to be
  added ("any popular open source license") but the file itself carries no license header. Treat
  this copy as "all rights reserved, reproduced for commentary/education" until upstream adds one.

Run as-is (no dependencies beyond the standard library):

```sh
python3 microgpt.py
```

Trains a ~4,200-parameter GPT on ~32k names for 1,000 steps (~90s on a single CPU core, pure
Python, no numpy/torch) and samples 20 new, made-up names from it.
