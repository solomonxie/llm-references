# $ venv/bin/python 03_react_prompt_from_scratch.py
#
# Goal: ReAct (Reason + Act) -- instead of a single tool call, the model
# interleaves free-text reasoning with actions, in a loop, until it decides
# it has enough to answer. The whole thing is just a prompting convention
# (a text format the model is asked to follow) plus a loop on our side that
# parses each "Action" line, runs it, and appends the "Observation" back
# into the transcript before asking the model to continue.
# Step 3: The ReAct Thought/Action/Observation loop, via plain-text prompting

import re

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


def get_weather(city: str) -> str:
    fake_weather = {"paris": "15C, cloudy", "tokyo": "22C, sunny", "cairo": "34C, clear"}
    return fake_weather.get(city.lower(), f"no data for {city}")


def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))  # toy demo only -- never eval untrusted input in real code
    except Exception as e:
        return f"error: {e}"


TOOLS = {"get_weather": get_weather, "calculator": calculator}

REACT_PROMPT = """Answer the question by reasoning step by step. You have these tools:
- get_weather(city): current weather for a city
- calculator(expression): evaluate a math expression

Use exactly this format, one step per turn:
Thought: <your reasoning>
Action: <tool_name>(<argument>)

When you have the final answer, instead write:
Thought: <your reasoning>
Final Answer: <answer>

Question: {question}
"""

ACTION_RE = re.compile(r"Action:\s*(\w+)\((.*)\)")
FINAL_RE = re.compile(r"Final Answer:\s*(.+)")


def react_loop(question: str, max_steps: int = 5) -> str:
    transcript = REACT_PROMPT.format(question=question)

    for step in range(max_steps):
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": transcript, "stream": False})
        text = response.json()["response"]
        # Stop at the first Action/Final Answer line -- some models keep
        # rambling and hallucinate their own fake Observation lines otherwise.
        matches = [m for m in (ACTION_RE.search(text), FINAL_RE.search(text)) if m]
        if matches:
            end = min(m.end() for m in matches)
            newline = text.find("\n", end)
            step_text = text[: newline + 1] if newline != -1 else text
        else:
            step_text = text
        print(f"--- step {step + 1} ---\n{step_text.strip()}")
        transcript += step_text

        if final_match := FINAL_RE.search(step_text):
            return final_match.group(1).strip()

        if action_match := ACTION_RE.search(step_text):
            tool_name, arg = action_match.group(1), action_match.group(2).strip().strip("'\"")
            observation = TOOLS[tool_name](arg) if tool_name in TOOLS else f"unknown tool {tool_name}"
            obs_line = f"Observation: {observation}\n"
            print(obs_line.strip())
            transcript += obs_line

    return "(gave up after max_steps without a Final Answer)"


answer = react_loop("If it's above 20C in Tokyo, what's double that temperature?")
print(f"\nfinal answer: {answer}")
