# $ venv/bin/python hello-agent/02_function_schema_and_manual_dispatch.py
#
# Goal: a "tool" is just a Python function plus a JSON schema describing its
# name, purpose, and arguments -- the schema is what actually gets shown to
# the model, in the prompt, as text. This step does the whole tool-call
# cycle by hand: describe the tools in the prompt, ask the model to reply
# with JSON when it wants one called, parse that JSON ourselves, run the
# real function, and feed the result back in. Step 4 replaces this hand-
# rolled JSON convention with the model's native tool-calling API.
# Step 2: Function schemas + manually parsing/dispatching a tool call

import json

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"


def get_weather(city: str) -> str:
    fake_weather = {"paris": "15C, cloudy", "tokyo": "22C, sunny", "cairo": "34C, clear"}
    return fake_weather.get(city.lower(), f"no data for {city}")


TOOLS = {
    "get_weather": {
        "fn": get_weather,
        "schema": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"city": "string, e.g. 'Paris'"},
        },
    }
}

tool_descriptions = "\n".join(json.dumps(t["schema"]) for t in TOOLS.values())
system_prompt = f"""You are a helpful assistant with access to these tools:
{tool_descriptions}

To call a tool, reply with ONLY this JSON, nothing else:
{{"tool": "<name>", "args": {{...}}}}

If you don't need a tool, reply normally in plain text."""

messages = [{"role": "system", "content": system_prompt}]


def ask(user_text: str) -> str:
    messages.append({"role": "user", "content": user_text})
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "messages": messages, "stream": False})
    reply = response.json()["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    return reply


def try_parse_tool_call(text: str) -> dict | None:
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "tool" in data and data["tool"] in TOOLS:
            return data
    except json.JSONDecodeError:
        pass
    return None


user_question = "What's the weather like in Tokyo right now?"
print(f"user: {user_question}")
reply = ask(user_question)
print(f"assistant (raw): {reply}")

call = try_parse_tool_call(reply)
if call:
    print(f"-> detected tool call: {call}")
    result = TOOLS[call["tool"]]["fn"](**call["args"])
    print(f"-> tool result: {result}")
    # Feed the observation back as a new user turn -- there's no dedicated
    # "tool" message role here since we're not using a native tool-calling
    # API yet; that distinction shows up in step 4.
    final = ask(f"Tool result: {result}")
    print(f"assistant (final): {final}")
else:
    print("(model answered directly, no tool needed)")
