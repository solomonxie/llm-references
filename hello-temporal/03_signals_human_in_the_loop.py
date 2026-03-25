# $ temporal server start-dev &
# $ venv/bin/python 03_signals_human_in_the_loop.py
#
# Goal: an agent workflow that must pause for a human approval before
# taking an action -- the workflow calls workflow.wait_condition() and
# blocks (durably; it costs nothing while waiting, even for hours) until a
# Signal arrives. Signals are the standard way an external system tells a
# running workflow something happened, without polling.
# Step 3: A workflow that suspends until an `approve` signal arrives

import asyncio
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker

TASK_QUEUE = "hello-temporal-task-queue"


@activity.defn
async def perform_action(action: str) -> str:
    return f"executed: {action}"


@workflow.defn
class ApprovalWorkflow:
    def __init__(self) -> None:
        self._approved: bool | None = None

    @workflow.signal
    def approve(self, approved: bool) -> None:
        self._approved = approved

    @workflow.run
    async def run(self, action: str) -> str:
        await workflow.wait_condition(lambda: self._approved is not None)
        if not self._approved:
            return "rejected"
        return await workflow.execute_activity(perform_action, action, start_to_close_timeout=timedelta(seconds=10))


async def main():
    client = await Client.connect("localhost:7233")

    async with Worker(client, task_queue=TASK_QUEUE, workflows=[ApprovalWorkflow], activities=[perform_action]):
        handle = await client.start_workflow(
            ApprovalWorkflow.run, "delete staging database", id="approval-wf", task_queue=TASK_QUEUE
        )
        print("workflow started, waiting for approval...")
        await asyncio.sleep(2)  # stand-in for however long a human actually takes
        await handle.signal(ApprovalWorkflow.approve, True)
        print(await handle.result())


if __name__ == "__main__":
    asyncio.run(main())
