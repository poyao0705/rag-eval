# Task 1 Report: Add passage and association models

## Scope

Added the `SourcePassage` and `HotpotQAContext` SQLModel table classes and their focused model tests. The existing `HotpotQA` model was left unchanged.

## Test-first evidence

The focused test suite was written before the production model classes. Its first run failed during test import because `SourcePassage` and `HotpotQAContext` were not yet defined in `backend.db.models`:

```
ImportError: cannot import name 'HotpotQAContext' from 'backend.db.models'
```

After the minimal implementation, the focused suite passed:

```
Ran 2 tests in 0.001s

OK
```

Command:

```
cd backend && UV_CACHE_DIR=/tmp/deep-agents-uv-cache uv run python -m unittest tests.test_models -v
```

## Implementation

- `SourcePassage` uses UUIDv7 identity, stores original and normalized title fields, JSONB sentences, text, and content hash.
- Added the `uq_source_passage_identity` unique constraint over `(normalized_title, content_hash)`.
- Added the content-hash lookup index.
- `HotpotQAContext` uses `(hotpot_qa_id, position)` as its composite primary key and references `source_passage.id`.
- Added the retrieval index over `(hotpot_qa_id, source_passage_id)`.

## Self-review

`git diff --check` passed. Only `backend/src/backend/db/models.py` and `backend/tests/test_models.py` were changed for this task; unrelated pre-existing untracked content under `docs/superpowers/plans/` was not modified.

## Concerns

None.
