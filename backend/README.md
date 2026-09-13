# Backend

## RAG agent

`build_rag_graph` uses `langchain.agents.create_agent` with one retrieval tool.
Invoke it with `{"question": "..."}`. Each successful invocation makes two
`gpt-5-mini` requests: a required retrieval tool call with a model-selected
search query, then an answer with tools disabled. The retriever remains fixed
per graph and returns up to five passages.

The tool returns `Command(update=...)`, storing `retrieved_passages`,
`retrieval_context`, `retrieval_count`, and a matching `ToolMessage` in agent
state. The next model request reads that message; the final state retains
these fields and adds `answer` for evaluation. Empty results still count as
the one retrieval. Each invocation starts with fresh state.

Middleware rejects missing, malformed, multiple, or repeated tool calls;
provider tool-choice settings and prompts are not the only enforcement.
Retrieval errors propagate without retry or final generation. Exactly-once
means one retrieval in a successful invocation, not crash-safe deduplication
across caller retries. Providers must support required tool choice, disabling
parallel calls, and the OpenAI Responses API.

## RAG evaluation

The evaluation is offline by default. The paid harness is gated by
`RUN_RAG_EVAL=1` and writes its report to `backend/.rag-eval/results.json`.
It evaluates the same fixed 20-question validation cohort with each retriever
sequentially (60 cases and five metrics per case).

Before a paid run, verify that the configured provider supports the exact
judge model `gpt-5.4` and its structured-output interface. This explicitly
pins DeepEval 4.2.2's recommended/default judge using its stock `OpenAIModel`,
including native schema parsing; no model capability overrides are applied.
Provider compatibility or availability failures are configuration blockers; the
harness does not silently substitute another model. The existing BM25 index
migration also requires explicit operator authorization. The generator uses
`gpt-5-mini` and the configured OpenAI credentials. Do not put credentials in
reports or command output.

The cohort uses seed 42 and a stable database hash, so it is repeatable when
the eligible database contents are unchanged. This does not make LLM output
deterministic. There is no fresh-cohort, random-seed CLI, or full-dataset
mode. Retrieval searches the global materialized corpus rather than passages
linked to each question.

HotpotQA answers are short gold answers. They are useful expected outputs but
do not necessarily express every multi-hop supporting fact, so contextual
metrics should not be interpreted as a standalone answer-correctness gate.
Each metric may make multiple judge requests; the full evaluation therefore
costs more than 300 model calls even though it has 300 metric measurements.

```bash
# Offline: no opt-in, no paid calls.
cd backend
uv sync --locked
uv run pytest tests/test_rag_graph.py tests/test_rag_eval_helpers.py tests/test_rag_eval_scoring.py tests/test_rag.py -q

# Operator-only, after explicit DB migration authorization.
uv run alembic current
uv run alembic upgrade b2c3d4e5f6a7

# Paid: only after provider compatibility and explicit evaluation authorization.
RUN_RAG_EVAL=1 uv run pytest tests/test_rag.py -q
```
