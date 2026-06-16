# $ venv/bin/python hello-langchain/01_llm_call.py
#
# Goal: the simplest possible LangChain call — wrap a running local model
# (via Ollama) in LangChain's ChatModel interface and invoke it. LangChain's
# main value is that every model provider (OpenAI, Anthropic, Ollama, ...)
# implements this same interface, so everything built on top of it in later
# steps (prompts, chains, agents) is provider-agnostic.
#
# Prerequisite: `ollama serve` running, with a model pulled, e.g.:
#   ollama pull llama3.2:3b
# Step 1: ChatOllama.invoke() -- the base chat-model interface every provider implements

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b", temperature=0)

response = llm.invoke("In one short sentence, what is a transformer model?")

print(f"response type: {type(response).__name__}")
print(f"content:       {response.content}")
print(f"tokens used:   {response.usage_metadata}")
