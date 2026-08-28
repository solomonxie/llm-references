# $ venv/bin/python 05_memory_conversation.py
#
# Goal: multi-turn conversation. A chat model call is stateless — it only
# knows what's in the messages you send it *this call*. "Memory" is nothing
# more than: keep a running list of every message so far, and send the whole
# list back each time. There's no hidden server-side session (Ollama/OpenAI
# don't remember previous calls) — the client owns history entirely.

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b", temperature=0)

history: list = [SystemMessage("You are a concise assistant. Answer in one short sentence.")]


def ask(question: str) -> str:
    history.append(HumanMessage(question))
    response = llm.invoke(history)  # the *entire* history, not just this question
    history.append(AIMessage(response.content))
    return response.content


print(f"Q: My name is Ravi and I collect vintage synthesizers.")
print(f"A: {ask('My name is Ravi and I collect vintage synthesizers.')}")

print(f"\nQ: What's my name?")
print(f"A: {ask('What is my name?')}")  # only answerable because history[1] is still in the list

print(f"\nQ: What do I collect?")
print(f"A: {ask('What do I collect?')}")

print(f"\nfull history ({len(history)} messages):")
for m in history:
    print(f"  [{m.type}] {m.content}")

# Real apps don't keep every message forever — a long-running conversation
# would eventually exceed the model's context window, and older turns matter
# less. Common strategies: keep only the last N turns, periodically
# summarize old turns into one shorter message, or (LangGraph's approach,
# see hello-langgraph) persist history in a checkpointed graph state that can
# be trimmed/summarized as a first-class step in the graph itself.
