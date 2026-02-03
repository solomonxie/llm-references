# $ venv/bin/python 06_tools_function_calling.py
#
# Goal: let the model call real code. `@tool` turns a plain Python function
# into something the model can be told about (name, description, parameter
# schema — all inferred from the function's signature/docstring/type hints).
# `.bind_tools()` makes the model *aware* of the tools; the model can then
# request a call instead of answering directly. The model never executes
# anything itself — it only asks; your code decides whether/how to comply.
# Step 6: @tool + bind_tools() -- the model requests calls, your code executes them

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    # A real tool would call a weather API — hardcoded here to keep this
    # example dependency-free and deterministic.
    fake_data = {"tokyo": "18°C, cloudy", "austin": "31°C, sunny"}
    return fake_data.get(city.lower(), f"No data for {city}")


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


llm = ChatOllama(model="llama3.2:3b", temperature=0)
llm_with_tools = llm.bind_tools([get_weather, add])

response = llm_with_tools.invoke("What's the weather in Tokyo?")
print(f"model wants to call: {response.tool_calls}")

# The model's response has no real answer yet — just a *request* to call
# get_weather(city="Tokyo"). Executing it and feeding the result back is on us.
tools_by_name = {"get_weather": get_weather, "add": add}
messages = [HumanMessage("What's the weather in Tokyo?"), response]
for call in response.tool_calls:
    result = tools_by_name[call["name"]].invoke(call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    print(f"executed {call['name']}({call['args']}) -> {result}")

final = llm_with_tools.invoke(messages)
print(f"\nfinal answer: {final.content}")

# A question needing a *different* tool routes to a different call —
# nothing here is hardcoded to weather; the model picks based on the tools'
# descriptions and the question.
response2 = llm_with_tools.invoke("What is 17 plus 25?")
print(f"\nsecond question's tool call: {response2.tool_calls}")
