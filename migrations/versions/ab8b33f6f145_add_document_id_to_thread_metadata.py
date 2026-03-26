"""add document_id to thread_metadata

Revision ID: ab8b33f6f145
Revises: c4c4bcf909a4
Create Date: 2026-03-26 13:43:36.076879

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ab8b33f6f145'
down_revision: Union[str, Sequence[str], None] = 'c4c4bcf909a4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('thread_metadata') as batch_op:
        batch_op.add_column(sa.Column('document_id', sa.String(), nullable=True))
        batch_op.create_index('ix_thread_metadata_document_id', ['document_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('thread_metadata') as batch_op:
        batch_op.drop_index('ix_thread_metadata_document_id')
        batch_op.drop_column('document_id')

    # ### end Alembic commands ###
