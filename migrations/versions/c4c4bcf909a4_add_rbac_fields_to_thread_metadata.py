"""add_rbac_fields_to_thread_metadata

Revision ID: c4c4bcf909a4
Revises: eafb8d9484b6   # ← change this to match your actual previous revision ID
Create Date: 2026-03-11 17:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c4c4bcf909a4'
down_revision: Union[str, Sequence[str], None] = 'eafb8d9484b6'  # ← must match your last successful migration
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add RBAC fields to thread_metadata table."""
    with op.batch_alter_table('thread_metadata', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('department', sa.String(), nullable=False, server_default='General'))
        batch_op.add_column(sa.Column('is_global', sa.Boolean(), nullable=False, server_default=sa.text('true')))

        # Foreign key to users
        batch_op.create_foreign_key(
            'fk_thread_metadata_user_id_users',
            'users',
            ['user_id'],
            ['id'],
            ondelete='SET NULL'
        )

        # Indexes
        batch_op.create_index('ix_thread_metadata_user_id', ['user_id'])
        batch_op.create_index('ix_thread_metadata_is_global', ['is_global'])


def downgrade() -> None:
    """Remove RBAC fields from thread_metadata table."""
    with op.batch_alter_table('thread_metadata', schema=None) as batch_op:
        batch_op.drop_index('ix_thread_metadata_is_global')
        batch_op.drop_index('ix_thread_metadata_user_id')
        batch_op.drop_constraint('fk_thread_metadata_user_id_users', type_='foreignkey')
        batch_op.drop_column('is_global')
        batch_op.drop_column('department')
        batch_op.drop_column('user_id')