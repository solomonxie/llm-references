# $ venv/bin/python hello-rag/04_basic_rag_pipeline.py
#
# Goal: chain steps 1-3 into one real, if small, end-to-end pipeline: chunk
# a document -> embed each chunk -> index in a vector store -> given a
# question, retrieve the most relevant chunks -> stuff them into a prompt
# -> generate a grounded answer. Every RAG system, however elaborate, is
# some version of exactly this pipeline.
# Step 4: Chunk -> embed -> index -> retrieve -> prompt -> generate, end to end

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

SOURCE_DOC = """Our return policy allows refunds within 30 days of purchase with a valid receipt. \
Items must be unworn and in original packaging. Sale items are final sale and not eligible for \
refunds. Store credit is offered for returns after 30 days but within 60 days, at the manager's \
discretion. Shipping costs are non-refundable except in cases of a defective or incorrect item. \
To start a return, contact support@example.com with your order number. Refunds are processed \
within 5-7 business days of receiving the returned item. International orders have a 45-day \
return window instead of 30, due to longer shipping times."""


def sentence_aware_chunks(text: str, target_size: int) -> list[str]:
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


embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:3b", temperature=0)

chunks = sentence_aware_chunks(SOURCE_DOC, target_size=150)
print(f"split into {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"  [{i}] {chunk!r}")

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents([Document(chunk) for chunk in chunks])

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using ONLY the given context. Be concise. If the context doesn't say, say so."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def answer(question: str, k: int = 2) -> str:
    retrieved = vector_store.similarity_search(question, k=k)
    context = "\n".join(doc.page_content for doc in retrieved)
    response = (prompt | llm).invoke({"context": context, "question": question})
    return response.content, retrieved


print()
for question in [
    "Can I get a refund on a sale item?",
    "How long do international customers have to return something?",
    "Do you cover return shipping costs?",
]:
    response, retrieved = answer(question)
    print(f"Q: {question}")
    print(f"   retrieved: {[r.page_content for r in retrieved]}")
    print(f"A: {response}\n")
