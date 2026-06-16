# $ venv/bin/python hello-agent-sdk/01_claude_agent_sdk.py
#
# Goal: the same one-tool task ("what's the weather in Paris?", answered via
# a get_weather tool) through Anthropic's official Claude Agent SDK, to see
# what a vendor SDK buys you over the raw loop in hello-agent: the agent
# loop, tool-call parsing/dispatch, and message plumbing are all handled --
# you declare tools and read back a finished turn.
# Step 1: A single-tool agent via claude_agent_sdk's query() + in-process
# MCP tool server

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)


@tool("get_weather", "Get the current weather for a city", {"city": str})
async def get_weather(args: dict) -> dict:
    # Fake data -- the point is the plumbing, not a real weather API.
    return {"content": [{"type": "text", "text": f"Sunny, 22C in {args['city']}."}]}


weather_server = create_sdk_mcp_server(name="weather", tools=[get_weather])

options = ClaudeAgentOptions(
    mcp_servers={"weather": weather_server},
    allowed_tools=["mcp__weather__get_weather"],
    system_prompt="Answer using the get_weather tool when asked about weather.",
)


async def main():
    async for message in query(prompt="What's the weather in Paris?", options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)


anyio.run(main)
