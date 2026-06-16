# $ venv/bin/python hello-eval/03_llm_as_judge.py
#
# Goal: rubrics (step 2) only catch what you thought to check for in
# advance, and can't judge fluency, coherence, or subtle correctness.
# "LLM-as-judge" asks a model to score another model's output instead --
# more flexible, but now the judge's own reliability matters. The key
# technique: force the judge to return structured JSON (a score + a
# reason) instead of free text, so the result is actually parseable.
# Step 3: Using a local LLM to score answers against a rubric, as JSON

import json

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "qwen2.5:7b"  # a separate, ideally stronger model than what's being judged

JUDGE_PROMPT = """You are grading an AI assistant's answer to a question.

Question: {question}
Reference (a correct answer, for comparison -- the candidate need not match it word for word): {reference}
Candidate answer: {candidate}

Score the candidate from 1 (wrong/unhelpful) to 5 (fully correct and clear).
Reply with ONLY this JSON, nothing else:
{{"score": <1-5 integer>, "reason": "<one sentence>"}}"""


def judge(question: str, reference: str, candidate: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, candidate=candidate)
    response = requests.post(
        OLLAMA_URL,
        json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False},
    )
    raw = response.json()["message"]["content"]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": None, "reason": f"judge returned unparseable output: {raw!r}"}


cases = [
    {
        "question": "Why does ice float on water?",
        "reference": "Ice is less dense than liquid water, so it floats.",
        "candidate": "Ice floats because water expands and becomes less dense when it freezes into a crystal structure.",
    },
    {
        "question": "Why does ice float on water?",
        "reference": "Ice is less dense than liquid water, so it floats.",
        "candidate": "Because ice is cold and cold things float.",  # confidently wrong reasoning
    },
]

for case in cases:
    result = judge(case["question"], case["reference"], case["candidate"])
    print(f"candidate: {case['candidate']!r}")
    print(f"  -> score={result.get('score')}  reason={result.get('reason')}\n")
