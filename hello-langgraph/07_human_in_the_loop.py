# $ venv/bin/python hello-langgraph/07_human_in_the_loop.py
#
# Goal: pause a graph mid-run for a human decision, then resume it — for
# anything too risky to let an LLM do unsupervised (send the email, run the
# migration, spend the money). `interrupt(payload)` called inside a node
# halts the graph right there and surfaces `payload` to the caller; the
# graph's state (including everything computed so far) is frozen, not lost —
# resuming with `Command(resume=value)` makes that `interrupt()` call return
# `value`, as if it had been a normal blocking input() all along.
#
# Requires a checkpointer (step 6) — interrupt/resume works by suspending
# and later restoring persisted state, the same mechanism that makes
# multi-turn memory durable.
# Step 7: interrupt() / Command(resume=...) -- pausing a graph for human approval, then continuing

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    recipient: str
    draft: str
    sent: bool


def write_draft(state: State) -> dict:
    return {"draft": f"Hi {state['recipient']}, following up on our chat — let's sync tomorrow at 2pm."}


def send_email(state: State) -> dict:
    # interrupt(...) freezes the graph HERE and hands `payload` back to
    # whoever called .invoke() — execution of this node does not continue
    # until a later .invoke(Command(resume=...)) call supplies an answer.
    approved = interrupt({"action": "send_email", "to": state["recipient"], "draft": state["draft"]})
    if not approved:
        return {"sent": False}
    print(f"  [actually sending email to {state['recipient']}...]")
    return {"sent": True}


builder = StateGraph(State)
builder.add_node("write_draft", write_draft)
builder.add_node("send_email", send_email)
builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "send_email")
builder.add_edge("send_email", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "approval-1"}}

result = graph.invoke({"recipient": "Priya", "draft": "", "sent": False}, config)

# When a node calls interrupt(), .invoke() returns immediately with an
# `__interrupt__` key instead of running to END — the graph state at this
# point is PAUSED at send_email, not finished.
if "__interrupt__" in result:
    payload = result["__interrupt__"][0].value
    print(f"PAUSED for approval: {payload}")

    # Simulates a human reviewing the draft. Anything could decide this in a
    # real app — a CLI prompt, a Slack approval button, a web form.
    human_says_yes = "sync tomorrow" in payload["draft"]
    print(f"human decision: {'approve' if human_says_yes else 'reject'}")

    # Resuming re-enters send_email with interrupt(...) now returning
    # `human_says_yes` — write_draft does NOT re-run; only the paused node
    # continues, using the state already computed before the pause.
    final = graph.invoke(Command(resume=human_says_yes), config)
    print(f"final state: {final}")
else:
    print(f"final state (no interrupt hit): {result}")
