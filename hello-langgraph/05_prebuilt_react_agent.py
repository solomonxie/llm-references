# $ venv/bin/python hello-langgraph/05_prebuilt_react_agent.py
#
# Goal: `create_agent` (langgraph.prebuilt's older `create_react_agent` is
# deprecated in favor of this, same idea) — a prebuilt graph that IS steps
# 3+4 combined: an LLM node bound to tools, a conditional edge that loops
# back to the LLM whenever it requested a tool call, and exits once it
# responds with none. Exactly hello-langchain's 08_agent_loop.py's
# hand-written while loop, packaged as one function call instead of code you
# maintain yourself. Worth building the loop by hand once (as that file did)
# so this isn't a black box.
# Step 5: create_agent -- steps 3+4's loop-until-no-tool-calls, prebuilt

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def get_stock_price(ticker: str) -> float:
    """Look up a stock's current price by ticker symbol."""
    fake_prices = {"AAPL": 227.50, "GOOG": 172.30, "MSFT": 425.10}
    return fake_prices.get(ticker.upper(), -1.0)


llm = ChatOllama(model="qwen2.5:7b", temperature=0)  # see hello-langchain/README for why not llama3.2:3b here
agent = create_agent(llm, tools=[get_stock_price])

result = agent.invoke({"messages": [HumanMessage("Which is more expensive, AAPL or MSFT?")]})

print("full message trace (this IS the loop — HumanMessage, AIMessage w/ tool_calls, ToolMessage, ..., final AIMessage):")
for m in result["messages"]:
    label = type(m).__name__
    extra = f"  tool_calls={m.tool_calls}" if getattr(m, "tool_calls", None) else ""
    print(f"  [{label}] {m.content!r}{extra}")

print(f"\nfinal answer: {result['messages'][-1].content}")

# `agent.get_graph().draw_mermaid()` (see hello-langgraph/01) would show
# exactly the same loop-back-to-the-LLM-node shape as 03_cycles_loop.py —
# create_agent isn't a different mechanism, it's the same StateGraph
# primitives, pre-wired.
