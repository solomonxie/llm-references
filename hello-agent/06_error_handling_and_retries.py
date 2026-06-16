# $ venv/bin/python hello-agent/06_error_handling_and_retries.py
#
# Goal: step 5's loop assumed every tool call is well-formed. Real models
# call tools with missing/wrong-typed args, call a tool that doesn't exist,
# or the tool itself raises. None of that should crash the agent -- it
# should come back as an Observation the model can react to, same as a
# real result, with a retry budget so a persistently broken call doesn't
# loop forever.
# Step 6: Malformed tool calls, exceptions, and bounded retries

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"


def get_weather(city: str) -> str:
    fake_weather = {"paris": "15C, cloudy", "tokyo": "22C, sunny"}
    if city.lower() not in fake_weather:
        raise ValueError(f"no weather data for {city!r}")
    return fake_weather[city.lower()]


def calculator(expression: str) -> str:
    # Deliberately fragile -- no try/except here, so the caller (this
    # file's loop, via safe_call) is responsible for catching it.
    return str(eval(expression, {"__builtins__": {}}))


TOOLS = {"get_weather": get_weather, "calculator": calculator}
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate a math expression",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]

MAX_TOOL_FAILURES = 3


def safe_call(fn_name: str, args: dict) -> tuple[str, bool]:
    """Returns (result_or_error_text, ok)."""
    if fn_name not in TOOLS:
        return f"error: unknown tool {fn_name!r}", False
    try:
        return str(TOOLS[fn_name](**args)), True
    except TypeError as e:
        return f"error: bad arguments {args} for {fn_name}: {e}", False
    except Exception as e:
        return f"error: {fn_name} raised: {e}", False


def run_agent(question: str, max_steps: int = 6) -> str:
    messages = [{"role": "user", "content": question}]
    failures = 0

    for step in range(max_steps):
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "tools": TOOL_SCHEMAS, "stream": False},
        )
        message = response.json()["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            return message["content"]

        for call in tool_calls:
            fn_name = call["function"]["name"]
            args = call["function"]["arguments"]
            result, ok = safe_call(fn_name, args)
            print(f"step {step + 1}: {fn_name}({args}) -> {result} ({'ok' if ok else 'FAILED'})")

            if not ok:
                failures += 1
                if failures >= MAX_TOOL_FAILURES:
                    return f"(giving up: {failures} tool failures in a row)"
                # Feed the error back as the observation -- the model gets a
                # chance to correct its own call (fix args, try a different
                # city, give up on that tool) instead of the loop just dying.
                messages.append({"role": "tool", "content": result})
            else:
                failures = 0
                messages.append({"role": "tool", "content": result})

    return "(gave up after max_steps)"


for question in [
    "What's the weather in Berlin?",  # not in fake_weather -> tool raises
    "What is 2 +* 3?",                 # invalid expression -> calculator raises
]:
    print(f"\nquestion: {question}")
    print(f"answer: {run_agent(question)}")
