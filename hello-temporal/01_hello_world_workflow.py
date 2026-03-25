# $ temporal server start-dev &            # local dev server, in another terminal
# $ venv/bin/python 01_hello_world_workflow.py
#
# Goal: the smallest possible Temporal program -- a Workflow (the durable,
# replayable orchestration logic) that calls one Activity (the actual
# side-effecting work). Both a Worker (which executes them) and the Client
# call (which starts one run) live in this single script for a runnable
# demo; in production the worker is normally its own long-lived process.
# Step 1: One workflow, one activity, run end to end

import asyncio
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker

TASK_QUEUE = "hello-temporal-task-queue"


@activity.defn
async def say_hello(name: str) -> str:
    return f"Hello, {name}!"


@workflow.defn
class SayHelloWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        return await workflow.execute_activity(
            say_hello, name, start_to_close_timeout=timedelta(seconds=10)
        )


async def main():
    client = await Client.connect("localhost:7233")

    async with Worker(client, task_queue=TASK_QUEUE, workflows=[SayHelloWorkflow], activities=[say_hello]):
        result = await client.execute_workflow(
            SayHelloWorkflow.run, "World", id="hello-temporal-wf", task_queue=TASK_QUEUE
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
