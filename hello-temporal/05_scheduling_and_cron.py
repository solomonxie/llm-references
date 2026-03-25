# $ temporal server start-dev &
# $ venv/bin/python 05_scheduling_and_cron.py
#
# Goal: run an agent workflow on a recurring cadence -- a nightly report,
# a periodic cleanup -- without a client-side cron job of your own.
# Temporal Schedules live server-side: create one and the server starts a
# new workflow run on the interval, whether or not any client process is
# up to see it happen. Uses a 30s interval here so the demo sees a real
# firing instead of waiting a day.
# Step 5: A Schedule that starts AskLLMWorkflow every 30 seconds

import asyncio
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleIntervalSpec,
    ScheduleSpec,
)
from temporalio.worker import Worker

TASK_QUEUE = "hello-temporal-task-queue"
SCHEDULE_ID = "hello-temporal-daily-report"


@activity.defn
async def write_report() -> str:
    return "report generated"


@workflow.defn
class ReportWorkflow:
    @workflow.run
    async def run(self) -> str:
        return await workflow.execute_activity(write_report, start_to_close_timeout=timedelta(seconds=10))


async def main():
    client = await Client.connect("localhost:7233")

    async with Worker(client, task_queue=TASK_QUEUE, workflows=[ReportWorkflow], activities=[write_report]):
        await client.create_schedule(
            SCHEDULE_ID,
            Schedule(
                action=ScheduleActionStartWorkflow(
                    ReportWorkflow.run, id="report-wf", task_queue=TASK_QUEUE
                ),
                spec=ScheduleSpec(intervals=[ScheduleIntervalSpec(every=timedelta(seconds=30))]),
            ),
        )
        print(f"schedule '{SCHEDULE_ID}' created, waiting for one firing...")
        await asyncio.sleep(35)

        handle = client.get_schedule_handle(SCHEDULE_ID)
        await handle.delete()
        print("schedule deleted")


if __name__ == "__main__":
    asyncio.run(main())
