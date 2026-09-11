"""add split to hotpot qa

Revision ID: c24b04d30087
Revises: 4109dc5ed73a
Create Date: 2026-09-11 20:53:30.081516

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c24b04d30087'
down_revision: Union[str, Sequence[str], None] = '4109dc5ed73a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


split_enum = sa.Enum(
    "TRAIN",
    "VALIDATION",
    "TEST",
    name="hotpotqasplit",
)


def upgrade() -> None:
    """Upgrade schema."""
    split_enum.create(op.get_bind(), checkfirst=True)
    op.add_column("hotpot_qa", sa.Column("split", split_enum, nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("hotpot_qa", "split")
    split_enum.drop(op.get_bind(), checkfirst=True)
