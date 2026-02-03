# $ venv/bin/python 03_cycles_loop.py
#
# Goal: a cycle — a conditional edge that can route *backward*, so a node
# runs again instead of the graph always moving forward to END. This is what
# separates a graph from a plain DAG pipeline (LCEL chains, step 1-2 here so
# far): a graph can loop until some condition holds, which is exactly the
# shape "retry", "keep refining a draft", or "an agent's think/act loop"
# need — and is not expressible as a straight-line chain at all.
# Step 3: Looping -- a conditional edge that routes backward, not just forward

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    target: int
    guess: int
    attempts: int


def guess(state: State) -> dict:
    # A deliberately dumb "guesser": nudge halfway toward the target each
    # time, standing in for any iterative-refinement step (a real one might
    # ask an LLM to revise a draft based on the last round's critique).
    new_guess = state["guess"] + (state["target"] - state["guess"]) // 2
    if new_guess == state["guess"]:  # integer division can stall one below target
        new_guess += 1
    return {"guess": new_guess, "attempts": state["attempts"] + 1}


def is_close_enough(state: State) -> str:
    if state["guess"] == state["target"]:
        return "done"
    if state["attempts"] >= 10:  # a real loop always needs a hard cap — never trust the "done" condition alone
        return "give_up"
    return "keep_guessing"


builder = StateGraph(State)
builder.add_node("guess", guess)

builder.add_edge(START, "guess")
#            ┌────────────────────────────┐
#            │                            │
#            ▼                            │  ("keep_guessing" routes back to
#   START ──► guess ──► is_close_enough ───┘   the SAME node — that's the cycle)
#                              │
#                              ├──► "done"     ──► END
#                              └──► "give_up"  ──► END
builder.add_conditional_edges("guess", is_close_enough, {"keep_guessing": "guess", "done": END, "give_up": END})

graph = builder.compile()

result = graph.invoke({"target": 73, "guess": 0, "attempts": 0})
print(f"final state: {result}")

print("\nstreamed (watch `guess` fire repeatedly):")
for update in graph.stream({"target": 200, "guess": 1, "attempts": 0}):
    print(f"  {update}")
