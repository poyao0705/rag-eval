# rag-eval

A rag system with comprehensive eval + observability

## Getting Started

### Database (PostgreSQL + ParadeDB)

The repository builds a PostgreSQL 17 image with both `pgvector` and ParadeDB's
`pg_search` extension. No host-level PostgreSQL extensions need to be installed.

For a fresh checkout, build and start PostgreSQL from the repository root:

```bash
docker compose up -d --build postgres
```

On first initialization, Docker automatically creates the `vector`, `pg_search`,
and `unaccent` extensions in `deep_agents_rag`. The image also starts PostgreSQL
with `pg_search` in `shared_preload_libraries`.

Verify the installation:

```bash
docker compose exec postgres \
  psql -U postgres -d deep_agents_rag \
  -c "SHOW shared_preload_libraries;"

docker compose exec postgres \
  psql -U postgres -d deep_agents_rag \
  -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_search', 'unaccent') ORDER BY extname;"
```

#### Existing database volumes

Scripts under `infra/db/init` run only when PostgreSQL initializes an empty data
directory. Developers who created `postgres_data` before ParadeDB was added must
rebuild the container and create the extension once in the existing database:

```bash
docker compose up -d --build --force-recreate postgres
docker compose exec postgres \
  psql -U postgres -d deep_agents_rag \
  -c "CREATE EXTENSION IF NOT EXISTS pg_search CASCADE;"
```

This preserves the existing volume. Do not run `docker compose down -v` unless
you intentionally want to delete all local database data.

#### Create and query a BM25 index

After the Alembic migrations have created `source_passage`, create a ParadeDB
index. A table can have only one ParadeDB index, and its `key_field` must be the
first indexed column and uniquely identify each row.

```bash
docker compose exec postgres psql -U postgres -d deep_agents_rag
```

```sql
CREATE INDEX source_passage_paradedb_idx
ON source_passage
USING paradedb (id, title, text)
WITH (key_field = 'id');

SELECT id, title, pdb.score(id) AS score
FROM source_passage
WHERE text ||| 'distributed systems'
ORDER BY pdb.score(id) DESC, id ASC
LIMIT 10;
```

The ParadeDB package version is pinned in `compose.yaml` and its release checksums
are pinned in `infra/db/Dockerfile`. Update both when upgrading `pg_search`.

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
