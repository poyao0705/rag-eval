"""add embeddings to source passages

Revision ID: a9b8c7d6e5f4
Revises: f1a2b3c4d5e6
Create Date: 2026-09-12

"""
from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a9b8c7d6e5f4"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add nullable 1,536-dimensional embeddings and a cosine HNSW index."""
    op.add_column(
        "source_passage",
        sa.Column("embedding", Vector(1536), nullable=True),
    )
    op.create_index(
        "source_passage_embedding_hnsw_idx",
        "source_passage",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Remove the embedding index and column."""
    op.drop_index(
        "source_passage_embedding_hnsw_idx",
        table_name="source_passage",
    )
    op.drop_column("source_passage", "embedding")
