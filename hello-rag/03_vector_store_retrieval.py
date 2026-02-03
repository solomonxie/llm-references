# $ venv/bin/python 03_vector_store_retrieval.py
#
# Goal: step 2's cosine-similarity loop, generalized into what a "vector
# store" actually is — a place that holds many (vector, document) pairs and
# answers "give me the k closest to this query vector," without the caller
# re-implementing the similarity math or the sorting by hand each time. At
# small scale (this file) it's a linear scan under the hood — real vector
# databases (Chroma, Pinecone, pgvector, FAISS) add an approximate-nearest-
# neighbor index (e.g. HNSW) so that scan doesn't become the bottleneck at
# millions of vectors, but the *interface* is identical at any scale.

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

documents = [
    Document("RAG stands for Retrieval-Augmented Generation.", metadata={"topic": "rag"}),
    Document("Chunking splits documents into smaller pieces before embedding.", metadata={"topic": "chunking"}),
    Document("Cosine similarity measures the angle between two vectors.", metadata={"topic": "embeddings"}),
    Document("Paris is the capital of France.", metadata={"topic": "geography"}),
    Document("The Eiffel Tower is located in Paris.", metadata={"topic": "geography"}),
]

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents)

query = "How do you measure how similar two embeddings are?"

# similarity_search: just the documents, ranked, no scores.
results = vector_store.similarity_search(query, k=2)
print(f"query: {query!r}")
print("\nsimilarity_search (k=2):")
for doc in results:
    print(f"  {doc.page_content!r}  (topic={doc.metadata['topic']})")

# similarity_search_with_score: same ranking, WITH the actual similarity
# number — useful for thresholding ("only use retrieved chunks above 0.7")
# instead of blindly taking the top k regardless of how weak the match is.
print("\nsimilarity_search_with_score (k=5, all documents, see the score gap):")
for doc, score in vector_store.similarity_search_with_score(query, k=5):
    print(f"  {score:.4f}  {doc.page_content!r}")

# A query about Paris should surface the geography documents instead —
# same store, same code, different question routes to different documents.
geo_query = "What city has the Eiffel Tower?"
print(f"\nquery: {geo_query!r}")
for doc, score in vector_store.similarity_search_with_score(geo_query, k=2):
    print(f"  {score:.4f}  {doc.page_content!r}")
