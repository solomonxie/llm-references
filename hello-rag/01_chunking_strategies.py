# $ venv/bin/python 01_chunking_strategies.py
#
# Goal: before anything can be embedded or retrieved, a document has to be
# split into chunks — an embedding model has a token limit, and a whole
# document is usually way more context than one question needs anyway. HOW
# it's split matters a lot: cutting mid-sentence produces chunks that read as
# nonsense in isolation and embed poorly (an embedding of a fragment doesn't
# represent the fragment's actual meaning well).

document = """LangChain is a framework for building applications with large language models. \
It provides abstractions for prompts, chains, and agents. Retrieval-Augmented Generation, or RAG, \
is one of its most common use cases. RAG works by retrieving relevant documents and inserting them \
into the prompt before generation. This grounds the model's answers in real data instead of relying \
purely on what it memorized during training. Chunking strategy has a big effect on RAG quality. \
Chunks that are too large waste context and dilute relevance. Chunks that are too small lose \
surrounding context a reader would need to understand them."""


def fixed_size_chunks(text: str, size: int) -> list[str]:
    """Simplest possible strategy: cut every `size` characters, no regard for
    word or sentence boundaries at all."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def sentence_aware_chunks(text: str, target_size: int) -> list[str]:
    """Split on sentence boundaries first, then GREEDILY pack whole sentences
    into a chunk until adding the next one would exceed target_size — never
    cuts a sentence in half."""
    sentences = [s.strip() + "." for s in text.split(". ") if s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > target_size:
            chunks.append(current.strip())
            current = ""
        current += " " + sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def overlapping_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size, but each chunk repeats the last `overlap` characters of
    the previous one — a fact split across a chunk boundary still appears
    whole in at least one chunk, at the cost of some redundancy."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


print("=== fixed-size (100 chars, no boundary awareness) ===")
for i, chunk in enumerate(fixed_size_chunks(document, 100)):
    print(f"[{i}] {chunk!r}")

print("\n=== sentence-aware (~150 char target, never splits a sentence) ===")
for i, chunk in enumerate(sentence_aware_chunks(document, 150)):
    print(f"[{i}] {chunk!r}")

print("\n=== overlapping (100 chars, 20 char overlap) ===")
for i, chunk in enumerate(overlapping_chunks(document, 100, 20)):
    print(f"[{i}] {chunk!r}")

# Fixed-size cuts mid-word/mid-sentence — visibly worse when read in
# isolation, which is exactly how a retriever will hand a chunk to the
# model: with no surrounding context beyond what the chunk itself contains.
print("\nLangChain ships equivalents of the sentence-aware/overlapping ideas —\n"
      "RecursiveCharacterTextSplitter(chunk_size=..., chunk_overlap=...) from\n"
      "langchain_text_splitters is the standard starting point in practice.")
