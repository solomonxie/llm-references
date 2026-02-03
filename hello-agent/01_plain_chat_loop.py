# $ venv/bin/python 01_plain_chat_loop.py
#
# Goal: the baseline every agent loop builds on -- send messages, get a
# reply, repeat. No tools yet: the model can only answer from what it
# already knows or was told in the conversation. Talks directly to a local
# Ollama server over its HTTP API (no framework, so nothing hides how a
# "chat" actually works: it's just a growing list of messages resent
# every turn).
# Step 1: A plain multi-turn chat loop, raw HTTP, no tools

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

messages = [{"role": "system", "content": "You are a concise, helpful assistant."}]


def chat(user_text: str) -> str:
    messages.append({"role": "user", "content": user_text})
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "messages": messages, "stream": False})
    response.raise_for_status()
    reply = response.json()["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    return reply


turns = [
    "What's the capital of France?",
    "What's its population, roughly?",  # relies on "its" resolving from prior turn
]

for turn in turns:
    print(f"user: {turn}")
    print(f"assistant: {chat(turn)}\n")

print(f"full message history has {len(messages)} entries -- that's the entire")
print("state of the conversation; there's no other memory mechanism at play.")
