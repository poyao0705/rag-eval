# rag-eval

A rag system with comprehensive eval + observability

## Getting Started

### Backend

1. From the repository root, start PostgreSQL:

   ```bash
   docker compose up -d postgres
   ```

2. Go to the `backend` directory.
3. Get `uv` if you don't have it already.
4. Create a `.env` file with the database URL:

   ```env
   DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/deep_agents_rag
   ```

5. Install dependencies:

   ```bash
   uv sync
   ```

6. Run the Alembic migrations to sync the database schema:

   ```bash
   uv run alembic upgrade head
   ```

7. Ingest the data:

   ```bash
   uv run ingest
   ```

8. Build the deduplicated, question-scoped passage corpus:

   ```bash
   uv run build-passages
   ```

The `hotpot_qa` table remains the raw evaluation data. The passage materializer
stores one unique normalized passage per identity in `source_passage`, while
`hotpot_qa_context` limits retrieval to the original question's context
candidates. Embeddings are a separate next step.

### HotpotQA passage sizing

Measured from the `hotpotqa/hotpot_qa` dataset using the `distractor` configuration across the train and validation splits. A passage is one context document with its sentences joined together; word counts use whitespace splitting.

- Average passage: ~89 words (~550 characters)
- Maximum passage: 1,378 words (7,903 characters)
- Average full question context: ~887 words across about 10 passages
- Maximum full question context: 2,792 words (17,526 characters)

Most passages do not need chunking. Keep context documents separate and chunk only oversized passages when required by the embedding model's input limit; avoid embedding the entire question context as one retrieval unit.
