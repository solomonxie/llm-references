# $ venv/bin/python 07_rag_retrieval.py
#
# Goal: RAG (Retrieval-Augmented Generation) — answer questions using
# documents the model was never trained on, by *retrieving* relevant ones
# and stuffing them into the prompt at call time, instead of fine-tuning.
#
# Two pieces:
#   - embeddings turn text into vectors such that semantically similar text
#     ends up nearby in vector space (a different model from the chat model —
#     here, Ollama's `nomic-embed-text`, not `llama3.2`)
#   - a vector store holds those vectors and answers "which stored vectors
#     are closest to this query vector?" (here, an in-memory one — no
#     external DB — real systems use Chroma/Pinecone/pgvector/etc. for
#     anything beyond toy scale, but the retrieval *interface* is identical)
# Step 7: Embeddings + an in-memory vector store -- grounding answers in your own docs

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:3b", temperature=0)

docs = [
    Document("The Eiffel Tower was completed in 1889 for the World's Fair in Paris."),
    Document("Mount Everest's summit sits at 8,849 meters above sea level."),
    Document("The company's return policy allows refunds within 30 days of purchase, with a receipt."),
    Document("Python's GIL (Global Interpreter Lock) means only one thread executes Python bytecode at a time."),
]

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)

query = "Can I get my money back if I bought something 20 days ago?"

# similarity_search embeds the query with the *same* embedding model, then
# ranks stored documents by vector closeness (cosine similarity here).
retrieved = vector_store.similarity_search(query, k=1)
print(f"query: {query!r}")
print(f"retrieved: {retrieved[0].page_content!r}\n")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using ONLY the given context. If the context doesn't cover it, say so."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

chain = prompt | llm

context = "\n".join(d.page_content for d in retrieved)
response = chain.invoke({"context": context, "question": query})
print(f"answer: {response.content}")

# A question the docs don't cover — the point of "using ONLY the context"
# above is the model should say so instead of falling back to its own
# training knowledge (which it *does* actually know, but that's not what
# RAG is for: grounding answers in *your* documents, not the model's guess).
off_topic = "What's the capital of France?"
retrieved2 = vector_store.similarity_search(off_topic, k=1)
response2 = chain.invoke({"context": retrieved2[0].page_content, "question": off_topic})
print(f"\noff-topic query: {off_topic!r}")
print(f"retrieved (best available, but irrelevant): {retrieved2[0].page_content!r}")
print(f"answer: {response2.content}")
