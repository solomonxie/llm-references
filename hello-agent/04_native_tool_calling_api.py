# $ venv/bin/python 04_native_tool_calling_api.py
#
# Goal: step 2 invented its own JSON-in-the-prompt convention for tool
# calls. Real tool-calling APIs (OpenAI, Anthropic, and Ollama's own
# `/api/chat` `tools` field) standardize this: you pass a JSON Schema per
# tool, the server fine-tunes the model's output to reliably produce a
# structured `tool_calls` field instead of raw text, and there's a
# dedicated "tool" message role for feeding results back.
# Step 4: The same weather tool, via Ollama's native tool-calling API

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"  # must be a tool-calling-capable model


def get_weather(city: str) -> str:
    fake_weather = {"paris": "15C, cloudy", "tokyo": "22C, sunny", "cairo": "34C, clear"}
    return fake_weather.get(city.lower(), f"no data for {city}")


# JSON Schema, not our own made-up format -- this is what actually travels
# over the wire to every major tool-calling API.
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, e.g. 'Paris'"}},
            "required": ["city"],
        },
    },
}

messages = [{"role": "user", "content": "What's the weather like in Cairo right now?"}]

response = requests.post(
    OLLAMA_URL,
    json={"model": MODEL, "messages": messages, "tools": [TOOL_SCHEMA], "stream": False},
)
message = response.json()["message"]
print(f"assistant message: {message}")

tool_calls = message.get("tool_calls", [])
if tool_calls:
    messages.append(message)
    for call in tool_calls:
        fn_name = call["function"]["name"]
        args = call["function"]["arguments"]  # already a parsed dict, not a string to json.loads
        result = get_weather(**args) if fn_name == "get_weather" else f"unknown tool {fn_name}"
        print(f"-> called {fn_name}({args}) = {result}")
        # The dedicated "tool" role is the API-native equivalent of step 2's
        # plain user-turn feedback -- it tells the model exactly which call
        # this observation answers.
        messages.append({"role": "tool", "content": result})

    final = requests.post(OLLAMA_URL, json={"model": MODEL, "messages": messages, "stream": False})
    print(f"final: {final.json()['message']['content']}")
else:
    print(f"(no tool call) {message['content']}")
