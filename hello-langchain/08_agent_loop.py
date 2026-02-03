# $ venv/bin/python 08_agent_loop.py
#
# Goal: an "agent" is nothing magical — it's step 6's tool-calling loop, run
# repeatedly instead of once. Ask, let the model request zero or more tool
# calls, execute them, feed results back, repeat until the model responds
# with no more tool calls (its "final answer"). This is the ReAct pattern
# (Reason -> Act -> Observe, looped) written out by hand; hello-langgraph's
# prebuilt agent (05_prebuilt_react_agent.py) is this exact loop, but as a
# library call instead of a while loop you maintain yourself.
# Step 8: The tool-calling loop from 06, run repeatedly until there's a final answer (ReAct)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def get_stock_price(ticker: str) -> float:
    """Look up a stock's current price by ticker symbol."""
    fake_prices = {"AAPL": 227.50, "GOOG": 172.30, "MSFT": 425.10}
    return fake_prices.get(ticker.upper(), -1.0)


@tool
def compare(a: float, b: float) -> str:
    """Compare two numbers, returning which is larger."""
    return "a" if a > b else "b" if b > a else "equal"


tools = [get_stock_price, compare]
tools_by_name = {t.name: t for t in tools}
# A bigger model than steps 1-7's llama3.2:3b — that one occasionally called
# compare() with another tool's *name* as the argument instead of waiting for
# its result (a real, common small-model tool-use failure — the try/except
# below exists because of exactly this). qwen2.5:7b handles the two-tool-call
# question reliably; still worth keeping the guard regardless of model size.
llm = ChatOllama(model="qwen2.5:7b", temperature=0).bind_tools(tools)


def run_agent(question: str, max_steps: int = 5) -> str:
    messages: list = [HumanMessage(question)]

    for step in range(1, max_steps + 1):
        response: AIMessage = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"  [step {step}] no tool calls — final answer")
            return response.content

        print(f"  [step {step}] model requested: {[c['name'] + str(c['args']) for c in response.tool_calls]}")
        for call in response.tool_calls:
            # A small local model occasionally requests a tool before it has
            # the inputs for it (e.g. calling compare() with another tool's
            # *name* as the argument, instead of waiting for that tool's
            # result). Feeding the error back as a ToolMessage — instead of
            # crashing — lets the model see what went wrong and retry on the
            # next step; a production agent loop needs this same guard
            # regardless of model size, since malformed tool args are a
            # normal failure mode, not an edge case.
            try:
                result = tools_by_name[call["name"]].invoke(call["args"])
            except Exception as error:
                result = f"ERROR: {error}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "(gave up after max_steps)"


# Answering this needs looking up BOTH prices before anything can be
# concluded — not something a single tool call can resolve alone. The model
# may satisfy "which is bigger" itself once it has both numbers, or call
# compare() too — either way the loop handles zero, one, or many tool calls
# per step identically.
question = "Which is more expensive right now, AAPL or MSFT?"
print(f"Q: {question}")
answer = run_agent(question)
print(f"A: {answer}")
