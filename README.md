# llm-references

Personal reference notes for working with LLMs — prompt-engineering patterns, API/model behavior,
and lessons learned across projects, kept separate from any one codebase so they're reusable.

## Scope

- **Prompt design**: patterns that held up in practice (schema-in-prompt, fact-sourcing/provenance
  tagging, splitting a large prompt into parallel calls, few-shot grounding via reference tables).
- **Model/API notes**: model IDs, reasoning-effort behavior, context/output limits, pricing,
  caching — things that go stale and are easy to mis-remember.
- **Failure modes**: recurring ways models deviate from an asked-for schema (and the recovery
  patterns that handled them), latency/timeout gotchas, etc.

## Layout

Not fixed yet — add files/folders as material accumulates. Prefer one topic per file over one
giant document.

Each `hello-*/` folder is a numbered, standalone-runnable progressive-learning series — one
script per step, each a full copy of the previous step plus one new concept. See each folder's
own README for what it covers.

**Suggested learning path** across these series: start with how text becomes vectors
(tokenization, then embeddings), then neural network fundamentals and the Transformer
architecture itself, then GPU programming fundamentals if going deeper on performance. From
there, move to inference/decoding mechanics on a real pretrained model, then retrieval-
augmented generation. Once a base model is understood, cover adapting it — fine-tuning (full
vs. LoRA/QLoRA) and quantization — followed by RLHF (reward modeling, PPO, DPO). Agent loops
and tool use come next, with evaluation alongside them to actually grade what those agents
produce. Serving concerns — inference-server mechanics and speculative decoding — are the
last layer, since they assume the modeling fundamentals underneath are already solid.
