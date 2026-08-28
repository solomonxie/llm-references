# $ venv/bin/python 02_embeddings_similarity.py
#
# Goal: the mechanism retrieval is actually built on. An embedding model
# turns text into a fixed-size vector such that semantically similar text
# lands at similar vectors — "similar" measured by cosine similarity (the
# cosine of the angle between two vectors: 1.0 = pointing the same
# direction, 0.0 = orthogonal/unrelated, -1.0 = opposite). Retrieval is
# nothing more than: embed the query, embed every candidate chunk, rank
# candidates by cosine similarity to the query.

import math

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",  # paraphrase of the first — should score HIGH
    "The stock market fell sharply today.",  # unrelated topic — should score LOW
    "Kittens are small cats.",  # related topic, not a paraphrase — somewhere in between
]

query = "A cat was lying on the carpet."

query_vec = embeddings.embed_query(query)
print(f"embedding dimension: {len(query_vec)}")

print(f"\nquery: {query!r}\n")
scored = [(sentence, cosine_similarity(query_vec, embeddings.embed_query(sentence))) for sentence in sentences]
for sentence, score in sorted(scored, key=lambda pair: pair[1], reverse=True):
    print(f"  {score:.4f}  {sentence!r}")

# embed_documents() embeds a batch in one call instead of one at a time —
# same vectors, just the batched API real pipelines use for a whole corpus.
print("\nembed_documents() (batch) matches embed_query() (single) for the same text:")
single = embeddings.embed_query(sentences[0])
batch = embeddings.embed_documents([sentences[0]])[0]
diff = sum(abs(a - b) for a, b in zip(single, batch))
print(f"  total absolute difference across all dims: {diff:.6f}  (should be ~0)")
