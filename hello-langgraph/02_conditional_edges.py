# $ venv/bin/python 02_conditional_edges.py
#
# Goal: branching. A conditional edge routes to a *different* next node
# depending on the current state, instead of always going to the same one —
# the graph-shaped equivalent of an if/elif/else. This is how a graph
# implements a router: "classify the input, then send it down the matching
# path."

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    amount: float
    category: str
    fee: float


def classify(state: State) -> dict:
    if state["amount"] < 100:
        category = "small"
    elif state["amount"] < 10_000:
        category = "medium"
    else:
        category = "large"
    return {"category": category}


def route_by_category(state: State) -> str:
    # A routing function returns the *name* of the next node — LangGraph
    # calls it after `classify` runs, then dispatches to whichever node name
    # comes back. It reads state but doesn't modify it (no dict update).
    return state["category"]


def small_fee(state: State) -> dict:
    return {"fee": state["amount"] * 0.05}


def medium_fee(state: State) -> dict:
    return {"fee": state["amount"] * 0.02}


def large_fee(state: State) -> dict:
    return {"fee": state["amount"] * 0.005}


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("small", small_fee)
builder.add_node("medium", medium_fee)
builder.add_node("large", large_fee)

builder.add_edge(START, "classify")
#   classify ──► route_by_category(state) ──┬──► small  ──► END
#                                            ├──► medium ──► END
#                                            └──► large  ──► END
# The dict maps route_by_category's return value -> which node to go to;
# without it, the string would have to exactly match a node's own name.
builder.add_conditional_edges("classify", route_by_category, {"small": "small", "medium": "medium", "large": "large"})
builder.add_edge("small", END)
builder.add_edge("medium", END)
builder.add_edge("large", END)

graph = builder.compile()

for amount in [42, 5_000, 250_000]:
    result = graph.invoke({"amount": amount, "category": "", "fee": 0.0})
    print(f"amount={amount:>10} -> category={result['category']:<7} fee={result['fee']:.2f}")
