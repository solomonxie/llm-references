# $ venv/bin/python 06_hybrid_search.py
#
# Goal: pure embedding similarity is surprisingly bad at exact matches —
# product codes, error codes, people's names, anything where the SPECIFIC
# characters matter more than the general meaning. "SKU-4471" and
# "SKU-4472" embed almost identically (both look like "a product code"
# semantically) even though they're completely different products. Keyword
# search catches exactly this; hybrid search runs both and combines them, so
# neither weakness dominates.

import math
from collections import Counter

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

documents = [
    Document("SKU-4471 is our stainless steel water bottle, 32oz capacity."),
    Document("SKU-4472 is our stainless steel water bottle, 18oz capacity."),
    Document("Our travel mugs keep drinks hot for up to 8 hours."),
    Document("Customer support hours are 9am-5pm Eastern, Monday through Friday."),
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def tokenize(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() else " " for c in text).split()


def keyword_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """A deliberately simple term-overlap score (count of shared tokens,
    weighted by how rare each token is across the corpus) — real keyword
    search uses BM25, which is this same "rare shared terms count more"
    idea with a more careful formula (term frequency saturation, document
    length normalization). The intuition transfers; the formula doesn't need
    to be exact for this to demonstrate the point."""
    doc_counts = Counter(doc_tokens)
    return sum(doc_counts[t] for t in query_tokens if t in doc_counts)


def vector_search(query: str, k: int) -> list[tuple[Document, float]]:
    query_vec = embeddings.embed_query(query)
    scored = [(doc, cosine_similarity(query_vec, embeddings.embed_query(doc.page_content))) for doc in documents]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]


def keyword_search(query: str, k: int) -> list[tuple[Document, float]]:
    query_tokens = tokenize(query)
    scored = [(doc, keyword_score(query_tokens, tokenize(doc.page_content))) for doc in documents]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]


def reciprocal_rank_fusion(*ranked_lists: list[tuple[Document, float]], k: int = 60) -> list[Document]:
    """Combines multiple RANKINGS (not raw scores, which aren't comparable
    across a cosine-similarity list and a keyword-count list) into one: each
    document earns 1/(k + rank) from each list it appears in, summed. A
    document ranked highly in both lists beats one ranked highly in only
    one — the whole point of "hybrid," not just picking one method's winner."""
    fused_scores: dict[str, float] = {}
    doc_by_content = {}
    for ranked_list in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked_list):
            fused_scores[doc.page_content] = fused_scores.get(doc.page_content, 0.0) + 1.0 / (k + rank)
            doc_by_content[doc.page_content] = doc
    ranked_content = sorted(fused_scores, key=lambda content: fused_scores[content], reverse=True)
    return [doc_by_content[content] for content in ranked_content]


query = "what's the capacity of SKU-4471"
print(f"query: {query!r}\n")

print("vector search alone (semantically both water bottles look similar):")
for doc, score in vector_search(query, k=4):
    print(f"  {score:.4f}  {doc.page_content!r}")

print("\nkeyword search alone (the exact SKU code wins decisively):")
for doc, score in keyword_search(query, k=4):
    print(f"  {score:.1f}  {doc.page_content!r}")

print("\nhybrid (reciprocal rank fusion of both):")
for doc in reciprocal_rank_fusion(vector_search(query, k=4), keyword_search(query, k=4)):
    print(f"  {doc.page_content!r}")
