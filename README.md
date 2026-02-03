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

### Progressive learning series

Each folder below is a numbered, standalone-runnable series — one script per step, each a full
copy of the previous step plus one new concept (see each folder's own README for the step table).

| Folder | Topic |
|---|---|
| `hello-transformer/` | The Transformer architecture, built from raw tensor ops |
| `hello-tokenizer/` | Tokenization from scratch: word/char-level through BPE, byte-level BPE, WordPiece |
| `hello-embeddings/` | Word vectors from scratch: co-occurrence, Skip-gram/CBOW, analogies, visualization |
| `hello-agent/` | Tool-calling/ReAct agent loop, raw HTTP against local Ollama, no framework |
| `hello-eval/` | Grading LLM outputs: string metrics, rubrics, LLM-as-judge, test suites, regression tracking |
| `hello-neuralnet/` | Neural nets from a single neuron through backprop, activations, convolution |
| `hello-inference/` | Decoding strategies and inference mechanics, on a real pretrained model |
| `hello-langchain/` | LangChain's core abstractions, against a local Ollama model |
| `hello-langgraph/` | LangGraph's StateGraph mechanics, agents, persistence, human-in-the-loop |
| `hello-rag/` | Retrieval-Augmented Generation, chunking through hybrid search and evaluation |
| `hello-gpu/` | GPU programming fundamentals (Metal compute kernels) -- separate from ML inference/training |
