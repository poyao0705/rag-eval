"""add ParadeDB BM25 index to source passages

Revision ID: b2c3d4e5f6a7
Revises: a9b8c7d6e5f4
Create Date: 2026-09-13

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a9b8c7d6e5f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS source_passage_paradedb_idx
ON source_passage
USING paradedb (id, title, text)
WITH (key_field = 'id')
""".strip()

_DROP_INDEX = "DROP INDEX IF EXISTS source_passage_paradedb_idx"


def upgrade() -> None:
    """Create the source passage ParadeDB BM25 index."""
    op.execute(_CREATE_INDEX)


def downgrade() -> None:
    """Remove the source passage ParadeDB BM25 index."""
    op.execute(_DROP_INDEX)
