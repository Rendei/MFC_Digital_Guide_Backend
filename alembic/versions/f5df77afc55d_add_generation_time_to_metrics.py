"""Add generation_time to metrics

Revision ID: f5df77afc55d
Revises: 503cd17d6bfa
Create Date: 2025-03-30 15:54:46.116756

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f5df77afc55d'
down_revision: Union[str, None] = '503cd17d6bfa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('metrics', sa.Column('generation_time_sec', sa.Float(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('metrics', 'generation_time_sec')
    # ### end Alembic commands ###
