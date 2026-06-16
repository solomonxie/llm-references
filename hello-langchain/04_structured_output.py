# $ venv/bin/python hello-langchain/04_structured_output.py
#
# Goal: get typed data back instead of prose. `.with_structured_output()`
# takes a Pydantic model describing the shape you want, converts it to a
# JSON-schema tool definition under the hood, and returns a parsed instance
# of that model — no manual "please respond in JSON" prompt-wrangling or
# regexing a fenced code block out of the response.
# Step 4: with_structured_output(PydanticModel) -- typed data out, not prose

from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b", temperature=0)


class MovieReview(BaseModel):
    title: str = Field(description="The movie's title")
    sentiment: str = Field(description="One of: positive, negative, mixed")
    one_line_summary: str = Field(description="A single sentence summary of the review")


structured_llm = llm.with_structured_output(MovieReview)

review_text = """
Dune: Part Two is a visual spectacle with a thin but effective plot. The
sandworm sequences alone are worth the ticket price, even if the pacing
drags in the middle act.
"""

result = structured_llm.invoke(f"Extract structured info from this review:\n{review_text}")

print(f"type:    {type(result).__name__}")
print(f"result:  {result}")
print(f"title:   {result.title}")
print(f"sentiment: {result.sentiment}")

# Same idea, applied to something with real structure: pull multiple
# typed fields out of unstructured text in one call.
class ExtractedEvent(BaseModel):
    event_name: str
    date: str = Field(description="ISO 8601 date if mentioned, else empty string")
    location: str = Field(description="Empty string if not mentioned")

event_llm = llm.with_structured_output(ExtractedEvent)
event = event_llm.invoke("Reminder: the team offsite is on March 14th in Austin.")
print(f"\nextracted event: {event}")
