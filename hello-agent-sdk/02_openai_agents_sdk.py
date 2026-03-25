# $ venv/bin/python 02_openai_agents_sdk.py
#
# Goal: the same weather-tool task, this time through OpenAI's Agents SDK.
# Contrast with step 1: tools are plain Python functions decorated with
# @function_tool (schema is inferred from the type hints), and an Agent +
# Runner pair replaces the mcp-server wiring -- a different shape for the
# same underlying job.
# Step 2: A single-tool agent via agents.Agent + Runner.run_sync

from agents import Agent, Runner, function_tool


@function_tool
def get_weather(city: str) -> str:
    # Fake data -- the point is the plumbing, not a real weather API.
    return f"Sunny, 22C in {city}."


agent = Agent(
    name="WeatherAssistant",
    instructions="Answer using the get_weather tool when asked about weather.",
    tools=[get_weather],
)

result = Runner.run_sync(agent, "What's the weather in Paris?")
print(result.final_output)
