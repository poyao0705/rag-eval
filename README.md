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
