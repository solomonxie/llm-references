# $ venv/bin/python hello-tokenizer/07_huggingface_tokenizers_compare.py
#
# Goal: put the from-scratch versions (03-06) next to production tokenizers.
# GPT-2's tokenizer is exactly step 5's byte-level BPE, trained on a huge
# corpus with a 50k-merge vocab instead of a toy 20-merge one. BERT's is
# exactly step 6's WordPiece. Same algorithms, just scaled up.
# Step 7: Loading real GPT-2 (byte-level BPE) and BERT (WordPiece) tokenizers

from transformers import AutoTokenizer

gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")

sentences = [
    "the quick brown fox jumps over the lazy dog",
    "tokenization is surprisingly deep",
    "café naïve \U0001f98a",
]

for sentence in sentences:
    print(f"\n{sentence!r}")

    gpt2_ids = gpt2_tok.encode(sentence)
    gpt2_pieces = gpt2_tok.convert_ids_to_tokens(gpt2_ids)
    print(f"  gpt2  (byte-level BPE, vocab={gpt2_tok.vocab_size}): "
          f"{len(gpt2_ids)} tokens -> {gpt2_pieces}")

    bert_ids = bert_tok.encode(sentence, add_special_tokens=False)
    bert_pieces = bert_tok.convert_ids_to_tokens(bert_ids)
    print(f"  bert  (WordPiece,     vocab={bert_tok.vocab_size}): "
          f"{len(bert_ids)} tokens -> {bert_pieces}")

print("\nnotice: GPT-2 never produces [UNK] (byte fallback, step 5) while BERT")
print("can, for characters its WordPiece vocab has no byte-level escape hatch for.")
print("BERT's '##' continuation pieces are exactly step 6's marker.")
