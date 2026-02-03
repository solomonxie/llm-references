# $ venv/bin/python 06_persistence_checkpoint.py
#
# Goal: make a graph remember state *between separate .invoke() calls* — the
# hand-rolled memory in hello-langchain/05_memory_conversation.py kept
# history in a Python list that only lived as long as the process did. A
# checkpointer persists the graph's state after every step, keyed by a
# `thread_id`, so a later .invoke() with the same thread_id picks up right
# where the last one left off — even from a different process, if the
# checkpointer backend is persistent (MemorySaver here is in-process only;
# SqliteSaver/PostgresSaver are the same interface, durable to disk).
# Step 6: MemorySaver + thread_id -- state that survives across separate .invoke() calls

from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

llm = ChatOllama(model="llama3.2:3b", temperature=0)


class State(TypedDict):
    # Annotated[..., add_messages] tells LangGraph HOW to merge a node's
    # return value into this key: append to the list, not overwrite it —
    # the same "list of every message so far" idea as
    # hello-langchain/05_memory_conversation.py, but now it's the graph's
    # own persisted state instead of a variable you manage.
    messages: Annotated[list, add_messages]


def chat(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=MemorySaver())

# thread_id is the only thing tying separate .invoke() calls into one
# conversation — every call below uses the SAME config to stay in one thread.
config = {"configurable": {"thread_id": "conversation-1"}}

r1 = graph.invoke({"messages": [HumanMessage("My favorite color is teal.")]}, config)
print(f"turn 1: {r1['messages'][-1].content}")

# Note: only the NEW message is passed in each call — the checkpointer
# already has turn 1's history under this thread_id and prepends it
# automatically. Passing the full history again (like the hand-rolled
# version did) would double it up.
r2 = graph.invoke({"messages": [HumanMessage("What's my favorite color?")]}, config)
print(f"turn 2: {r2['messages'][-1].content}")

# A different thread_id is a completely separate, empty conversation — the
# checkpointer keeps threads fully isolated from each other.
other_config = {"configurable": {"thread_id": "conversation-2"}}
r3 = graph.invoke({"messages": [HumanMessage("What's my favorite color?")]}, other_config)
print(f"\nnew thread: {r3['messages'][-1].content}  (no memory of thread 1 — different thread_id)")

print(f"\nfull persisted history for conversation-1 ({len(r2['messages'])} messages):")
for m in r2["messages"]:
    print(f"  [{m.type}] {m.content}")
