# $ venv/bin/python 01_state_graph_basics.py
#
# Goal: LangGraph's core idea, with no LLM involved at all — it's a graph
# library first, an LLM-orchestration library second. A graph has:
#   - a State: a typed dict describing what flows through the graph
#   - nodes: plain functions, each taking the current state and returning a
#     *partial* update to merge into it
#   - edges: which node runs after which
# Compiling turns this into a runnable object; invoking it runs nodes in
# order, threading the (evolving) state through each one.

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    text: str
    word_count: int


def count_words(state: State) -> dict:
    # A node returns only the keys it wants to update — not the whole state.
    # LangGraph merges this dict into the running state after the node runs.
    return {"word_count": len(state["text"].split())}


def shout(state: State) -> dict:
    return {"text": state["text"].upper() + "!"}


builder = StateGraph(State)
builder.add_node("count_words", count_words)
builder.add_node("shout", shout)

#   START ──► count_words ──► shout ──► END
builder.add_edge(START, "count_words")
builder.add_edge("count_words", "shout")
builder.add_edge("shout", END)

graph = builder.compile()

result = graph.invoke({"text": "hello graph world", "word_count": 0})
print(f"final state: {result}")

# .stream() instead of .invoke() surfaces each node's output as it runs,
# instead of only the final state — useful for watching a multi-step graph
# execute, or streaming partial progress to a UI.
print("\nstreamed, node by node:")
for update in graph.stream({"text": "another example run", "word_count": 0}):
    print(f"  {update}")

# The compiled graph is itself just a picture — this dumps it as Mermaid
# syntax, renderable at https://mermaid.live or in any Markdown viewer that
# supports Mermaid.
print("\nmermaid diagram:")
print(graph.get_graph().draw_mermaid())
