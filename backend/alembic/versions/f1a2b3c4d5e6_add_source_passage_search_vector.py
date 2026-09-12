"""add generated full-text search vector to source passages

Revision ID: f1a2b3c4d5e6
Revises: e7f9a1b3c5d7
Create Date: 2026-09-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "e7f9a1b3c5d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add a stored English full-text search vector and GIN index."""
    op.add_column(
        "source_passage",
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
        ),
    )
    op.create_index(
        "ix_source_passage_search_vector",
        "source_passage",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    """Remove the full-text search index and generated column."""
    op.drop_index(
        "ix_source_passage_search_vector",
        table_name="source_passage",
    )
    op.drop_column("source_passage", "search_vector")
