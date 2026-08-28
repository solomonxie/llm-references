# $ venv/bin/python 05_metadata_filtering.py
#
# Goal: pure similarity search can't express "...but only from THIS
# source" or "...but only docs newer than X" — semantically-close chunks
# from the wrong source/tenant/date range still show up. Metadata attached
# to each document at index time (source, category, date, permission level,
# ...) lets a query filter the candidate pool BEFORE (or alongside) ranking
# by similarity, not just after.

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

documents = [
    Document("Employees get 15 days of paid vacation per year.", metadata={"category": "hr", "year": 2024}),
    Document("Vacation days increased to 20 per year starting this year.", metadata={"category": "hr", "year": 2025}),
    Document("The engineering team uses a monorepo for all services.", metadata={"category": "engineering", "year": 2024}),
    Document("On-call rotations are handled through PagerDuty.", metadata={"category": "engineering", "year": 2025}),
]

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents)

query = "How many vacation days do I get?"

# Unfiltered: both HR documents are relevant by similarity, but they
# CONTRADICT each other (15 vs. 20 days) — the older one is stale policy.
print(f"query: {query!r}\n")
print("unfiltered (both years mixed together):")
for doc, score in vector_store.similarity_search_with_score(query, k=4):
    print(f"  {score:.4f}  {doc.page_content!r}  {doc.metadata}")

# InMemoryVectorStore's filter takes a predicate over each Document — real
# vector DBs use their own filter syntax (a dict/query language), but the
# idea is identical: narrow the candidate pool by metadata, THEN rank by
# similarity within what's left.
print("\nfiltered to year=2025 only (the current policy):")
current_year_filter = lambda doc: doc.metadata.get("year") == 2025  # noqa: E731
for doc, score in vector_store.similarity_search_with_score(query, k=4, filter=current_year_filter):
    print(f"  {score:.4f}  {doc.page_content!r}  {doc.metadata}")

# Same mechanism also enforces access control — a document tagged with which
# team/tenant/permission level owns it can be excluded from another team's
# retrieval entirely, not just deprioritized.
print("\nfiltered to category=engineering only:")
eng_filter = lambda doc: doc.metadata.get("category") == "engineering"  # noqa: E731
for doc, score in vector_store.similarity_search_with_score("How do I get paged?", k=4, filter=eng_filter):
    print(f"  {score:.4f}  {doc.page_content!r}  {doc.metadata}")
