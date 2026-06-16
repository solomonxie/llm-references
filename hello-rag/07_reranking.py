# $ venv/bin/python hello-rag/07_reranking.py
#
# Goal: embedding similarity is fast (one vector comparison per candidate)
# but imprecise — it only sees "roughly the same topic," not "actually
# answers this specific question." Re-ranking is a two-stage fix used
# throughout real search/RAG systems: cast a wide net cheaply (retrieve
# k=10 by embedding similarity), then have a slower, more accurate judge
# re-score just those 10 and re-sort. The judge here is the LLM itself,
# asked to score relevance directly — a real production system would more
# often use a dedicated cross-encoder model (smaller, cheaper per call than
# a full chat model), but the two-stage SHAPE is identical either way.
# Step 7: Cheap wide retrieval, then an LLM re-scores/re-sorts the candidates for precision

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:3b", temperature=0)

documents = [
    Document("Python was created by Guido van Rossum and first released in 1991."),
    Document("Python is commonly used for data science, web development, and scripting."),
    Document("The Python programming language emphasizes code readability."),
    Document("Snakes in the python family are non-venomous constrictors."),  # same word, wrong sense entirely
    Document("JavaScript, not Python, runs natively in web browsers."),
    Document("Guido van Rossum stepped down as Python's 'benevolent dictator for life' in 2018."),
]

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents)


class RelevanceScore(BaseModel):
    score: int = Field(description="Relevance to the question, 0 (irrelevant) to 10 (directly answers it)")


scorer = llm.with_structured_output(RelevanceScore)


def rerank(query: str, candidates: list[Document]) -> list[tuple[Document, int]]:
    scored = []
    for doc in candidates:
        result = scorer.invoke(f"Question: {query}\nPassage: {doc.page_content}\n\nRate this passage's relevance.")
        scored.append((doc, result.score))
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


query = "Who created Python?"
print(f"query: {query!r}\n")

candidates = vector_store.similarity_search(query, k=6)
print("stage 1 — embedding retrieval order (cheap, approximate):")
for doc in candidates:
    print(f"  {doc.page_content!r}")

print("\nstage 2 — LLM-reranked order (slower, more precise):")
for doc, score in rerank(query, candidates):
    print(f"  {score:>2}  {doc.page_content!r}")

# The "python the snake" document is a good test of whether re-ranking is
# doing real work: it can rank highly by embedding similarity (shares the
# word "python" and general topic proximity) while scoring near 0 once the
# LLM actually reads it against the specific question.
