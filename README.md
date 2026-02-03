# LLM References

Personal reference notes for working with LLMs — prompt-engineering patterns, API/model behavior,
and lessons learned across projects, kept separate from any one codebase so they're reusable.

## Scope

- **Prompt design**: patterns that held up in practice (schema-in-prompt, fact-sourcing/provenance
  tagging, splitting a large prompt into parallel calls, few-shot grounding via reference tables).
- **Model/API notes**: model IDs, reasoning-effort behavior, context/output limits, pricing,
  caching — things that go stale and are easy to mis-remember.
- **Failure modes**: recurring ways models deviate from an asked-for schema (and the recovery
  patterns that handled them), latency/timeout gotchas, etc.

## Suggested learning path

```
tokenization
embeddings
neural network fundamentals
transformer architecture
GPU fundamentals (optional, performance deep-dive)
inference / decoding mechanics
retrieval-augmented generation
fine-tuning (full fine-tune vs. LoRA/QLoRA)
quantization
RLHF (reward modeling, PPO, DPO)
agent loops (tool use, ReAct)
evaluation
inference-server mechanics
speculative decoding
```
