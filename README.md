# rag-eval

A rag system with comprehensive eval + observability

## Getting Started

### Database (PostgreSQL + ParadeDB)

The repository uses ParadeDB's official PostgreSQL 18 image, which includes
both `pgvector` and the `pg_search` extension. No host-level PostgreSQL
extensions need to be installed.

For a fresh checkout, start PostgreSQL from the repository root:

```bash
docker compose up -d postgres
```

On first initialization, ParadeDB's bootstrap creates the `vector`, `pg_search`,
and related extensions, then the project init script creates `unaccent` in
`deep_agents_rag`. The Compose command keeps ParadeDB's required extensions in
`shared_preload_libraries`.

Verify the installation:

```bash
docker compose exec postgres \
  psql -U postgres -d deep_agents_rag \
  -c "SHOW shared_preload_libraries;"

docker compose exec postgres \
  psql -U postgres -d deep_agents_rag \
  -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_search', 'unaccent') ORDER BY extname;"
```

#### Upgrading from PostgreSQL 17

PostgreSQL 17 data directories cannot be reused by PostgreSQL 18. This project
uses a destructive reset for local development, so back up anything you need
before removing the volume:

```bash
# Destructive: deletes the local PostgreSQL database and all generated data.
docker compose down -v --remove-orphans
docker compose pull postgres
docker compose up -d postgres
```

The fresh PostgreSQL 18 data directory runs ParadeDB's bootstrap and the project
init script automatically. Rerun the backend migrations, ingestion, passage
materialization, and embedding steps below to recreate the application data.
For data that must be preserved, use a PostgreSQL logical dump and restore
instead of reusing the PostgreSQL 17 volume.

#### Query the BM25 index

The Alembic migrations create `source_passage_paradedb_idx` after creating
`source_passage`. The index uses `id` as ParadeDB's unique key and indexes
`title` and `text` for BM25 retrieval.

```bash
docker compose exec postgres psql -U postgres -d deep_agents_rag
```

```sql
SELECT id, title, pdb.score(id) AS score
FROM source_passage
WHERE text ||| 'distributed systems'
ORDER BY pdb.score(id) DESC, id ASC
LIMIT 10;
```

The official image is pinned by its PostgreSQL 18 tag and multi-architecture
digest in `compose.yaml`. Update that image reference deliberately when
upgrading ParadeDB.

### Backend

1. Go to the `backend` directory.
2. Get `uv` if you don't have it already.
3. Create a `.env` file with the database URL and OpenAI API key:

   ```env
   DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/deep_agents_rag
   OPENAI_API_KEY=sk-your-key
   ```

4. Install dependencies:

   ```bash
   uv sync
   ```

5. Run the Alembic migrations to sync the database schema:

   ```bash
   uv run alembic upgrade head
   ```

6. Ingest the data:

   ```bash
   uv run ingest
   ```

7. Build the deduplicated, question-scoped passage corpus:

   ```bash
   uv run build-passages
   ```

8. Embed materialized passages with OpenAI:

   ```bash
   uv run embed-passages
   ```

The `hotpot_qa` table remains the raw evaluation data. The passage materializer
stores one unique normalized passage per identity in `source_passage`, while
`hotpot_qa_context` limits retrieval to the original question's context
candidates. `embed-passages` fills missing embeddings in resumable batches using
`text-embedding-3-small`; rerunning it skips passages that already have vectors.

### HotpotQA passage sizing

Measured from the `hotpotqa/hotpot_qa` dataset using the `distractor` configuration across the train and validation splits. A passage is one context document with its sentences joined together; word counts use whitespace splitting.

- Average passage: ~89 words (~550 characters)
- Maximum passage: 1,378 words (7,903 characters)
- Average full question context: ~887 words across about 10 passages
- Maximum full question context: 2,792 words (17,526 characters)

Most passages do not need chunking. Keep context documents separate and chunk only oversized passages when required by the embedding model's input limit; avoid embedding the entire question context as one retrieval unit.
