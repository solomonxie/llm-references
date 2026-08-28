# $ venv/bin/python hello-agent/11_compare_abstractions.py
#
# Goal: run the same weather-tool task through all three vendor SDKs back
# to back and print what each required: how a tool is declared, what runs
# the loop, and how many lines of plumbing it took. Skips any vendor whose
# API key isn't set rather than failing the whole script.
# Step 11: Side-by-side comparison across steps 8-10

import os
import time

ROWS = [
    ("Claude Agent SDK", "ANTHROPIC_API_KEY", "tool() decorator + create_sdk_mcp_server", "query() async generator"),
    ("OpenAI Agents SDK", "OPENAI_API_KEY", "@function_tool on a plain function", "Runner.run_sync(agent, prompt)"),
    ("Google ADK", "GOOGLE_API_KEY", "plain function, typed dict return", "Runner + InMemorySessionService"),
]


async def run_claude():
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, create_sdk_mcp_server, query, tool

    @tool("get_weather", "Get the current weather for a city", {"city": str})
    async def get_weather(args: dict) -> dict:
        return {"content": [{"type": "text", "text": f"Sunny, 22C in {args['city']}."}]}

    server = create_sdk_mcp_server(name="weather", tools=[get_weather])
    options = ClaudeAgentOptions(mcp_servers={"weather": server}, allowed_tools=["mcp__weather__get_weather"])
    out = []
    async for message in query(prompt="What's the weather in Paris?", options=options):
        if isinstance(message, AssistantMessage):
            out.extend(b.text for b in message.content if isinstance(b, TextBlock))
    return " ".join(out)


def run_openai():
    from agents import Agent, Runner, function_tool

    @function_tool
    def get_weather(city: str) -> str:
        return f"Sunny, 22C in {city}."

    agent = Agent(name="WeatherAssistant", tools=[get_weather])
    return Runner.run_sync(agent, "What's the weather in Paris?").final_output


async def run_adk():
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    def get_weather(city: str) -> dict:
        return {"status": "success", "report": f"Sunny, 22C in {city}."}

    agent = Agent(name="weather_agent", model="gemini-2.0-flash", tools=[get_weather])
    sessions = InMemorySessionService()
    await sessions.create_session(app_name="cmp", user_id="u", session_id="s")
    runner = Runner(agent=agent, app_name="cmp", session_service=sessions)
    message = types.Content(role="user", parts=[types.Part(text="What's the weather in Paris?")])
    async for event in runner.run_async(user_id="u", session_id="s", new_message=message):
        if event.is_final_response():
            return event.content.parts[0].text


RUNNERS = {"Claude Agent SDK": run_claude, "OpenAI Agents SDK": run_openai, "Google ADK": run_adk}

print(f"{'SDK':<20}{'tool declaration':<40}{'loop driver':<32}{'result / status'}")
for name, env_var, tool_decl, driver in ROWS:
    if not os.environ.get(env_var):
        print(f"{name:<20}{tool_decl:<40}{driver:<32}SKIPPED (no {env_var})")
        continue
    start = time.time()
    try:
        import asyncio

        fn = RUNNERS[name]
        result = asyncio.run(fn()) if asyncio.iscoroutinefunction(fn) else fn()
        status = f"{result!r} ({time.time() - start:.1f}s)"
    except Exception as exc:  # keep the comparison running even if one vendor errors
        status = f"ERROR: {exc}"
    print(f"{name:<20}{tool_decl:<40}{driver:<32}{status}")
