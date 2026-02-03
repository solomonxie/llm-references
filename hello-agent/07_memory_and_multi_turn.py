# $ venv/bin/python 07_memory_and_multi_turn.py
#
# Goal: an agent that only remembers the current question forgets tool
# results and decisions the moment it answers. This step keeps the full
# messages list (including every tool call and tool result) alive across
# separate top-level questions, so a later question can refer back to an
# earlier tool result without re-fetching it.
# Step 7: Persisting conversation + tool history across multiple turns

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"


def get_weather(city: str) -> str:
    fake_weather = {"paris": "15C, cloudy", "tokyo": "22C, sunny", "cairo": "34C, clear"}
    return fake_weather.get(city.lower(), f"no data for {city}")


TOOLS = {"get_weather": get_weather}
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]


class Agent:
    """Keeps the full message history (including tool calls/results) across
    separate `.ask()` calls -- that persisted list IS the agent's memory."""

    def __init__(self):
        self.messages = [{"role": "system", "content": "You are a helpful assistant. Be concise."}]

    def ask(self, question: str, max_steps: int = 4) -> str:
        self.messages.append({"role": "user", "content": question})

        for _ in range(max_steps):
            response = requests.post(
                OLLAMA_URL,
                json={"model": MODEL, "messages": self.messages, "tools": TOOL_SCHEMAS, "stream": False},
            )
            message = response.json()["message"]
            self.messages.append(message)

            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                return message["content"]

            for call in tool_calls:
                fn_name = call["function"]["name"]
                args = call["function"]["arguments"]
                result = TOOLS[fn_name](**args) if fn_name in TOOLS else f"unknown tool {fn_name}"
                print(f"  [tool] {fn_name}({args}) -> {result}")
                self.messages.append({"role": "tool", "content": str(result)})

        return "(gave up after max_steps)"


agent = Agent()

print("turn 1: What's the weather in Paris?")
answer1 = agent.ask("What's the weather in Paris?")
print(f"-> {answer1}\n")

print("turn 2: And in Tokyo?")
answer2 = agent.ask("And in Tokyo?")
print(f"-> {answer2}\n")

# No new tool call needed here -- the answer to "which was warmer" is
# derivable from the two tool results already sitting in self.messages.
print("turn 3: Which of those two was warmer?")
answer3 = agent.ask("Which of those two was warmer?")
print(f"-> {answer3}\n")

print(f"total messages retained: {len(agent.messages)}")
