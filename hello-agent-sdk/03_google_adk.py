# $ venv/bin/python hello-agent-sdk/03_google_adk.py
#
# Goal: the same weather-tool task through Google's Agent Development Kit.
# Contrast with steps 1-2: ADK separates the Agent (model + instructions +
# tools) from a Runner that drives it against a SessionService -- session
# state is an explicit, swappable dependency rather than hidden inside the
# call, which matters once an agent needs to persist across turns/processes.
# Step 3: A single-tool agent via google.adk Agent + Runner + InMemorySessionService

import asyncio

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME, USER_ID, SESSION_ID = "hello_adk", "user1", "session1"


def get_weather(city: str) -> dict:
    # Fake data -- the point is the plumbing, not a real weather API.
    return {"status": "success", "report": f"Sunny, 22C in {city}."}


root_agent = Agent(
    name="weather_agent",
    model="gemini-2.0-flash",
    instruction="Answer using the get_weather tool when asked about weather.",
    tools=[get_weather],
)


async def main():
    sessions = InMemorySessionService()
    await sessions.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=sessions)

    message = types.Content(role="user", parts=[types.Part(text="What's the weather in Paris?")])
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=message):
        if event.is_final_response():
            print(event.content.parts[0].text)


asyncio.run(main())
