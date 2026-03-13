"""add_user_rbac_fields

Revision ID: eafb8d9484b6
Revises: 106b989e501f
Create Date: 2026-03-11 17:18:31.770164

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'eafb8d9484b6'
down_revision: Union[str, Sequence[str], None] = '106b989e501f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Create the native PostgreSQL ENUM type first
    userrole_enum = postgresql.ENUM(
        'HR', 'EMPLOYEE', 'EXECUTIVE', 'INTERN',
        name='userrole'
    )
    userrole_enum.create(op.get_bind(), checkfirst=True)

    # 2. Alter users table
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                'role',
                userrole_enum,                    # ← reference the created enum type
                nullable=False,
                server_default='EMPLOYEE'
            )
        )
        batch_op.add_column(sa.Column('department', sa.String(), nullable=False, server_default='General'))
        batch_op.add_column(sa.Column('designation', sa.String(), nullable=True))

        # If you had 'updated_at' column that you want to drop (as in your original)
        batch_op.drop_column('updated_at')

    # Optional: add indexes for performance on frequent filters
    op.create_index(
        'ix_users_role_department',
        'users',
        ['role', 'department'],
        unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_index('ix_users_role_department', if_exists=True)
        batch_op.add_column(sa.Column('updated_at', postgresql.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True))
        batch_op.drop_column('designation')
        batch_op.drop_column('department')
        batch_op.drop_column('role')

    # Drop the enum type (only if no other columns use it)
    userrole_enum = postgresql.ENUM(
        'HR', 'EMPLOYEE', 'EXECUTIVE', 'INTERN',
        name='userrole'
    )
    userrole_enum.drop(op.get_bind(), checkfirst=True)