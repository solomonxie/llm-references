# $ temporal server start-dev &
# $ venv/bin/python hello-temporal/02_llm_activity_with_retries.py
#
# Goal: an Activity that calls a real LLM API -- the kind of call that
# times out or rate-limits in practice. A RetryPolicy on the activity call
# makes Temporal retry it automatically with backoff, entirely outside the
# workflow's own code, and durably: if the *worker process* dies mid-retry,
# a new worker picks the activity back up from Temporal's stored state
# rather than starting the workflow over.
# Step 2: An LLM-calling activity with an explicit RetryPolicy

import asyncio
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import Worker

TASK_QUEUE = "hello-temporal-task-queue"


@activity.defn
async def ask_llm(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(model="claude-opus-5", max_tokens=1024, messages=[{"role": "user", "content": prompt}])
    return next(b.text for b in response.content if b.type == "text")


@workflow.defn
class AskLLMWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return await workflow.execute_activity(
            ask_llm,
            prompt,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=10),
                maximum_attempts=5,
            ),
        )


async def main():
    client = await Client.connect("localhost:7233")

    async with Worker(client, task_queue=TASK_QUEUE, workflows=[AskLLMWorkflow], activities=[ask_llm]):
        result = await client.execute_workflow(
            AskLLMWorkflow.run, "In one sentence, what is durable execution?", id="ask-llm-wf", task_queue=TASK_QUEUE
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
