# Backend

## RAG agent

`build_rag_graph` uses `langchain.agents.create_agent` with one retrieval tool.
Invoke it with `{"question": "..."}`. The model may answer directly or request
one retrieval using the invocation's original question unchanged. Answers should
use only retrieved documents; without sufficient evidence, the model should say so.
The retriever
remains fixed per graph and returns up to configured `RAG_TOP_K` passages.

The tool returns `Command(update=...)`, storing `retrieved_passages`,
`retrieval_context`, and a matching `ToolMessage`. The final state also exposes
`answer` for evaluation. Each invocation clears prior messages and evidence.

`ToolCallLimitMiddleware(tool_name="retrieve", run_limit=1,
exit_behavior="continue")` enforces the per-invocation limit. Zero calls are
allowed; empty results still consume the retrieval allowance. A repeated request
receives an error `ToolMessage`, is not executed, and the agent continues. In a
batch, the first retrieval executes and excess retrieval calls are blocked.
Retrieval failures propagate without retry.
The retrieve tool exposes no model-controlled arguments. It reads the original
question from injected runtime state; nonblank question validation occurs before
model execution.
This is not crash-safe deduplication across caller retries. The configured
provider must support tool calling and the OpenAI Responses API.

## RAG configuration

Set RAG and evaluation knobs in `backend/.env`:

```dotenv
RAG_ANSWER_MODEL=gpt-5-mini
RAG_TOP_K=10
RAG_HYBRID_CANDIDATE_TOP_K=50
RAG_EVAL_JUDGE_MODEL=gpt-5.4
RAG_EVAL_SEED=42
RAG_EVAL_SAMPLE_SIZE=20
RAG_EVAL_METRIC_THRESHOLD=0.5
RAG_RERANK_MODEL=rerank-english-v3.0
```

These settings are built once per evaluation and passed to retrieval, answer
model, cohort selection, scoring, and report metadata. `RAG_TOP_K` controls the
final retrieval result count and `results.json`; it is not report-only.
`RAG_HYBRID_CANDIDATE_TOP_K` controls how many candidates each hybrid component
requests before fusion. Hybrid retrieval uses the larger of these two limits.
Positive integers are required for `RAG_TOP_K`, `RAG_HYBRID_CANDIDATE_TOP_K`, and
`RAG_EVAL_SAMPLE_SIZE`; the metric threshold must be in `[0, 1]`.

## Hybrid reranking

Each hybrid requests expanded lexical and vector candidates, fuses their full
deduplicated pool with reciprocal-rank fusion, then sends that pool to the
injected Cohere reranker before applying the final `RAG_TOP_K` limit. Fusion
`scores` retain reciprocal-rank-fusion meaning; `relevance_score` is Cohere's
reranking score, so the two score fields are not interchangeable.

Both hybrid retrievers require the same injected reranker. Reranking failures
propagate to the caller, and empty candidate pools do not call Cohere. The
caller owns the `cohere.AsyncClientV2`, configures it with the secret value, and
manages it using the installed SDK's verified async lifecycle (`async with`);
this scope does not add a client owner or assume a `.close()` method.

```python
reranker = CohereReranker(client, model=settings.rag_config.rerank_model)
bm25_hybrid = HybridBM25Retriever(
    embedder, reranker=reranker,
    candidate_top_k=settings.rag_config.hybrid_candidate_top_k,
)
tsvector_hybrid = HybridTSVectorRetriever(
    embedder, reranker=reranker,
    candidate_top_k=settings.rag_config.hybrid_candidate_top_k,
)
```

The caller-owned client is a configured `cohere.AsyncClientV2` using
`api_key=settings.COHERE_API_KEY.get_secret_value()`.

## RAG evaluation

The evaluation is offline by default. The paid harness is gated by
`RUN_RAG_EVAL=1` and writes its report to `backend/.rag-eval/results.json`.
It evaluates the configured validation cohort sequentially with `bm25`,
`tsvector`, `vector`, `hybrid_bm25`, and `hybrid_tsvector`, in that insertion order.
Adding or removing a graph entry automatically changes execution and reporting;
there is no separate roster to edit. Expected cases are `actual cohort size ×
graph count`: current defaults give 20 questions, 100 cases, and five metrics per
case (500 metric measurements). The live benchmark also requires the existing
`COHERE_API_KEY` for hybrid reranking.
Pipeline names may not be `attempted_case_count`, `completed_case_count`,
`failed_case_count`, or `expected_case_count`: these flat report keys are
reserved and rejected before execution or summary grouping.

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
Metric measurements are not API-call counts: each metric may make multiple
judge requests, and generation, embeddings, and hybrid reranking add calls/cost.

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
