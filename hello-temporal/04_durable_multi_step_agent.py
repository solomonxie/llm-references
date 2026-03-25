# $ temporal server start-dev &
# $ venv/bin/python 04_durable_multi_step_agent.py
#
# Goal: put it together into a durable multi-step agent -- each tool call
# is its own Activity with its own retry policy, and workflow.query lets
# an outside caller inspect progress without disturbing the run. The
# payoff: kill this script with Ctrl+C mid-loop (after step 1's activity
# has logged) and rerun it. Temporal replays the workflow's event history
# from its server-side log, so completed activities are NOT re-executed --
# the new worker resumes exactly at the next step, not from `run()`'s top.
# Step 4: Each tool call as an Activity, with query-able progress

import asyncio
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import Worker

TASK_QUEUE = "hello-temporal-task-queue"

FAKE_TOOLS = {"search": "found 3 relevant docs", "summarize": "wrote a 2-paragraph summary", "email": "sent to the team"}


@activity.defn
async def run_tool(name: str) -> str:
    activity.logger.info(f"running tool: {name}")
    return FAKE_TOOLS.get(name, f"unknown tool {name}")


@workflow.defn
class AgentWorkflow:
    def __init__(self) -> None:
        self._steps_done: list[str] = []

    @workflow.query
    def steps_done(self) -> list[str]:
        return self._steps_done

    @workflow.run
    async def run(self, plan: list[str]) -> str:
        for step in plan:
            result = await workflow.execute_activity(
                run_tool,
                step,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            self._steps_done.append(f"{step}: {result}")
        return "; ".join(self._steps_done)


async def main():
    client = await Client.connect("localhost:7233")

    async with Worker(client, task_queue=TASK_QUEUE, workflows=[AgentWorkflow], activities=[run_tool]):
        handle = await client.start_workflow(
            AgentWorkflow.run, list(FAKE_TOOLS.keys()), id="agent-wf", task_queue=TASK_QUEUE
        )
        await asyncio.sleep(0.5)
        print("progress so far:", await handle.query(AgentWorkflow.steps_done))
        print("final:", await handle.result())


if __name__ == "__main__":
    asyncio.run(main())
