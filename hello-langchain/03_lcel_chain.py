# $ venv/bin/python hello-langchain/03_lcel_chain.py
#
# Goal: LCEL (LangChain Expression Language) — pipe (`|`) runnables together
# into one chain, the same way `|` pipes shell commands. `prompt | llm` isn't
# a metaphor, it's a real composed Runnable: calling `.invoke(x)` on it feeds
# x through prompt.invoke(), pipes that output into llm.invoke(), and so on.
# Every piece from step 2 (prompt templates, chat models) is a Runnable, so
# they compose this way.
# Step 3: LCEL: piping prompt | llm | output_parser into one composed Runnable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [("system", "Answer in exactly one word."), ("human", "{question}")]
)

# Without a chain: three explicit steps, handling the AIMessage wrapper yourself.
messages = prompt.invoke({"question": "What color is the sky on a clear day?"})
raw = llm.invoke(messages)
print(f"unchained: {raw!r}  (an AIMessage, not a plain string)")

# StrOutputParser is a Runnable too — it just extracts `.content` from an
# AIMessage. Piping it on the end means the chain's final output is a plain
# string instead of a wrapped message object.
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "What color is the sky on a clear day?"})
print(f"chained:   {result!r}  (plain string)")

# The same composed chain, reused with different inputs — this is the whole
# payoff: build the pipeline once, invoke it many times.
for question in ["What color is grass?", "What shape is a ball?"]:
    print(f"{question!r} -> {chain.invoke({'question': question})!r}")

# `.batch()` runs multiple inputs — concurrently where the provider supports
# it — instead of a Python loop calling .invoke() one at a time.
answers = chain.batch([{"question": "2+2?"}, {"question": "3+3?"}])
print(f"\nbatched: {answers}")
