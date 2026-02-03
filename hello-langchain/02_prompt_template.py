# $ venv/bin/python 02_prompt_template.py
#
# Goal: stop hand-building prompt strings. ChatPromptTemplate holds a prompt
# with `{placeholders}`, separate from the values that fill them — the
# template is written once, then reused with different inputs. It also
# separates message *roles* (system/human/ai), which matters because chat
# models are trained to treat "the rules" (system) differently from "what the
# user said" (human).

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a terse code reviewer. Answer in one sentence, no preamble."),
        ("human", "What's one risk of this {language} snippet?\n\n{code}"),
    ]
)

# .invoke() on a prompt template fills the placeholders and returns a real
# list of role-tagged messages — nothing is sent to a model yet.
messages = prompt.invoke({"language": "Python", "code": "eval(user_input)"})
print("filled messages:")
for m in messages.to_messages():
    print(f"  [{m.type}] {m.content}")

response = llm.invoke(messages)
print(f"\nmodel response: {response.content}")

# The real point: the same template, reused with different inputs, no string
# formatting at the call site.
for code in ["eval(user_input)", "open(f'/tmp/{user_id}')"]:
    response = llm.invoke(prompt.invoke({"language": "Python", "code": code}))
    print(f"\n{code!r} -> {response.content}")
