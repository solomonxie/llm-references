# $ venv/bin/python 04_llm_node.py
#
# Goal: the first real LLM call, as one node among others — steps 1-3 proved
# the graph mechanics with plain Python; now a node's "computation" happens
# to be a chat model call instead of arithmetic. Nothing about the graph
# itself changes: a node is still a function, state -> partial state update.
#
# Prerequisite: `ollama serve` running with `ollama pull llama3.2:3b`.
# Step 4: A chat-model call as one node among plain-Python ones

from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

llm = ChatOllama(model="llama3.2:3b", temperature=0)


class State(TypedDict):
    topic: str
    draft: str
    word_count: int


def write_draft(state: State) -> dict:
    response = llm.invoke([HumanMessage(f"Write one short sentence about {state['topic']}.")])
    return {"draft": response.content}


def count_words(state: State) -> dict:
    # A plain-Python node right after an LLM node — mixing the two freely is
    # the point: not every step in a pipeline needs to be a model call.
    return {"word_count": len(state["draft"].split())}


builder = StateGraph(State)
builder.add_node("write_draft", write_draft)
builder.add_node("count_words", count_words)
builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "count_words")
builder.add_edge("count_words", END)

graph = builder.compile()

result = graph.invoke({"topic": "the deep sea", "draft": "", "word_count": 0})
print(f"draft:       {result['draft']!r}")
print(f"word count:  {result['word_count']}")

# Different input, same graph — this is still just steps 1-3's pattern; the
# LLM call is a node like any other.
result2 = graph.invoke({"topic": "vintage synthesizers", "draft": "", "word_count": 0})
print(f"\ndraft:       {result2['draft']!r}")
print(f"word count:  {result2['word_count']}")
