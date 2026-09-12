"""add source passages and HotpotQA context associations

Revision ID: d2e4f6a8b0c1
Revises: c24b04d30087
Create Date: 2026-09-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "d2e4f6a8b0c1"
down_revision: Union[str, Sequence[str], None] = "c24b04d30087"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table("source_passage",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("normalized_title", sa.Text(), nullable=False),
        sa.Column(
            "sentences",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "normalized_title",
            "content_hash",
            name="uq_source_passage_identity",
        ),
    )
    op.create_index(
        "ix_source_passage_content_hash",
        "source_passage",
        ["content_hash"],
    )
    op.create_table("hotpot_qa_context",
        sa.Column("hotpot_qa_id", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column(
            "source_passage_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["hotpot_qa_id"], ["hotpot_qa.id"]),
        sa.ForeignKeyConstraint(["source_passage_id"], ["source_passage.id"]),
        sa.PrimaryKeyConstraint(
            "hotpot_qa_id",
            "position",
            name="hotpot_qa_context_pkey",
        ),
    )
    op.create_index(
        "ix_hotpot_qa_context_retrieval",
        "hotpot_qa_context",
        ["hotpot_qa_id", "source_passage_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_hotpot_qa_context_retrieval",
        table_name="hotpot_qa_context",
    )
    op.drop_table("hotpot_qa_context")
    op.drop_index(
        "ix_source_passage_content_hash",
        table_name="source_passage",
    )
    op.drop_table("source_passage")
