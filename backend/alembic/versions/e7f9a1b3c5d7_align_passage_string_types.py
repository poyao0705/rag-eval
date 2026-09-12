"""align passage string columns with SQLModel's string mapping

Revision ID: e7f9a1b3c5d7
Revises: d2e4f6a8b0c1
Create Date: 2026-09-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e7f9a1b3c5d7"
down_revision: Union[str, Sequence[str], None] = "d2e4f6a8b0c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_STRING_COLUMNS = (
    ("source_passage", "title"),
    ("source_passage", "normalized_title"),
    ("source_passage", "text"),
    ("source_passage", "content_hash"),
    ("hotpot_qa_context", "hotpot_qa_id"),
)


def upgrade() -> None:
    """Align migrated string columns with SQLModel's VARCHAR mapping."""
    for table_name, column_name in _STRING_COLUMNS:
        op.alter_column(
            table_name,
            column_name,
            existing_type=sa.Text(),
            type_=sa.String(),
            existing_nullable=False,
        )


def downgrade() -> None:
    """Restore the original TEXT declarations."""
    for table_name, column_name in _STRING_COLUMNS:
        op.alter_column(
            table_name,
            column_name,
            existing_type=sa.String(),
            type_=sa.Text(),
            existing_nullable=False,
        )
