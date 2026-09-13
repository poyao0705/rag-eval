# Backend

## RAG agent

`build_rag_graph` uses `langchain.agents.create_agent` with one retrieval tool.
Invoke it with `{"question": "..."}`. Each successful invocation makes two
configured answer-model requests: a required retrieval tool call with a
model-selected search query, then an answer with tools disabled. The retriever
remains fixed per graph and returns up to configured `RAG_TOP_K` passages.

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

## RAG configuration

Set RAG and evaluation knobs in `backend/.env`:

```dotenv
RAG_ANSWER_MODEL=gpt-5-mini
RAG_TOP_K=10
RAG_EVAL_JUDGE_MODEL=gpt-5.4
RAG_EVAL_SEED=42
RAG_EVAL_SAMPLE_SIZE=20
RAG_EVAL_METRIC_THRESHOLD=0.5
```

These settings are built once per evaluation and passed to retrieval, answer
model, cohort selection, scoring, and report metadata. `RAG_TOP_K` controls
both the actual retrieval request and `results.json`; it is not report-only.
Positive integers are required for `RAG_TOP_K` and `RAG_EVAL_SAMPLE_SIZE`;
the metric threshold must be in `[0, 1]`.

## RAG evaluation

The evaluation is offline by default. The paid harness is gated by
`RUN_RAG_EVAL=1` and writes its report to `backend/.rag-eval/results.json`.
It evaluates the configured validation cohort with each retriever sequentially
(default: 20 questions, 60 cases, and five metrics per case).

Before a paid run, verify that the configured provider supports
`RAG_EVAL_JUDGE_MODEL` and its structured-output interface. The harness uses
DeepEval's stock `OpenAIModel`, including native schema parsing, and does not
silently substitute another model. Provider compatibility or availability
failures are configuration blockers. The existing BM25 index migration also
requires explicit operator authorization. Do not put credentials in reports or
command output.

The cohort uses `RAG_EVAL_SEED` and a stable database hash, so it is repeatable
when eligible database contents are unchanged. This does not make LLM output
deterministic. Retrieval searches the global materialized corpus rather than
passages linked to each question.

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
