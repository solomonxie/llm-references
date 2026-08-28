# $ venv/bin/python 05_multi_tool_agent_loop.py
#
# Goal: step 4 handled exactly one tool call and stopped. A real agent loop
# keeps going -- call a tool, feed the result back, let the model decide if
# it needs another tool or is ready to answer -- until no more tool calls
# come back. This is the same loop as `hello-langchain/08_agent_loop.py`,
# written raw instead of through a framework.
# Step 5: Looping native tool calls until the model gives a final answer

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"  # more reliable multi-step tool reasoning than 3b models


def get_weather(city: str) -> str:
    fake_weather = {"paris": "15C, cloudy", "tokyo": "22C, sunny", "cairo": "34C, clear"}
    return fake_weather.get(city.lower(), f"no data for {city}")


def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))  # toy demo only
    except Exception as e:
        return f"error: {e}"


TOOLS = {"get_weather": get_weather, "calculator": calculator}
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def run_agent(question: str, max_steps: int = 6) -> str:
    messages = [{"role": "user", "content": question}]

    for step in range(max_steps):
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "tools": TOOL_SCHEMAS, "stream": False},
        )
        message = response.json()["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            return message["content"]  # no more tool calls -- the model is done

        for call in tool_calls:
            fn_name = call["function"]["name"]
            args = call["function"]["arguments"]
            print(f"step {step + 1}: calling {fn_name}({args})")
            result = TOOLS[fn_name](**args) if fn_name in TOOLS else f"error: unknown tool {fn_name}"
            print(f"  -> {result}")
            messages.append({"role": "tool", "content": str(result)})

    return "(gave up after max_steps)"


question = "It's currently the temperature it is in Paris. If it were twice as warm, what would that be?"
print(f"question: {question}\n")
print(f"answer: {run_agent(question)}")
