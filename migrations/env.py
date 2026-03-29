# from logging.config import fileConfig

# from sqlalchemy import pool, create_engine

# from alembic import context
# import os
# from dotenv import load_dotenv
# from src.config import Config
# from src.database.table_models import Base, User, ThreadMetadata

# load_dotenv()  # if you use .env file

# # this is the Alembic Config object, which provides
# # access to the values within the .ini file in use.
# config = context.config

# # Interpret the config file for Python logging.
# # This line sets up loggers basically.
# if config.config_file_name is not None:
#     fileConfig(config.config_file_name)

# connectable = create_engine(Config.POSTGRES_CONNECTION)

# # add your model's MetaData object here
# # for 'autogenerate' support
# # from myapp import mymodel
# # target_metadata = mymodel.Base.metadata
# target_metadata = Base.metadata

# # other values from the config, defined by the needs of env.py,
# # can be acquired:
# # my_important_option = config.get_main_option("my_important_option")
# # ... etc.


# def include_object(object, name, type_, reflected, compare_to):
#     """
#     Filter: only include objects that are part of our target_metadata.
#     This prevents Alembic from trying to drop LangGraph / pgvector tables.
#     """
#     if type_ == "table":
#         # Only process tables that are in our models (users, future ones)
#         # Ignore everything else (checkpoints, langchain_pg_*, thread_metadata if not in models)
#         if name in target_metadata.tables:
#             return True
#         else:
#             # Skip unmanaged tables — don't drop or alter them
#             return False
    
#     # For other objects (columns, indexes, etc.) — allow if parent is included
#     if hasattr(object, 'table'):
#         if object.table.name in target_metadata.tables:
#             return True
#     return True

# def run_migrations_offline() -> None:
#     """Run migrations in 'offline' mode.

#     This configures the context with just a URL
#     and not an Engine, though an Engine is acceptable
#     here as well.  By skipping the Engine creation
#     we don't even need a DBAPI to be available.

#     Calls to context.execute() here emit the given string to the
#     script output.

#     """
#     url = config.get_main_option("sqlalchemy.url")
#     context.configure(
#         url=url,
#         target_metadata=target_metadata,
#         literal_binds=True,
#         dialect_opts={"paramstyle": "named"},
#     )

#     with context.begin_transaction():
#         context.run_migrations()


# def run_migrations_online():
#     connectable = create_engine(Config.POSTGRES_CONNECTION)

#     with connectable.connect() as connection:
#         context.configure(
#             connection=connection,
#             target_metadata=target_metadata,
            
#             # ── VERY IMPORTANT ──
#             # Tell Alembic to ONLY consider tables that exist in target_metadata
#             # → ignores checkpoints, langchain_pg_*, etc.
#             include_object=include_object,
            
#             # Optional but recommended for safety
#             render_as_batch=True,  # helps with some PostgreSQL operations
#         )

#         with context.begin_transaction():
#             context.run_migrations()


# if context.is_offline_mode():
#     run_migrations_offline()
# else:
#     run_migrations_online()

from logging.config import fileConfig

from sqlalchemy import pool, create_engine
from sqlalchemy.engine import URL

from alembic import context
import os
from dotenv import load_dotenv

from src.config import Config
from src.database.table_models import Base

load_dotenv()

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Use synchronous connection for Alembic (this fixes the MissingGreenlet error)
def get_sync_url():
    """Convert asyncpg URL to psycopg2 for migrations"""
    url = Config.POSTGRES_CONNECTION
    if "postgresql+asyncpg://" in url:
        url = url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    elif "postgresql+aiopg://" in url:
        url = url.replace("postgresql+aiopg://", "postgresql+psycopg2://")
    return url

target_metadata = Base.metadata

# ──────────────────────────────────────────────────────────────
# Custom include_object to protect LangGraph / pgvector tables
# ──────────────────────────────────────────────────────────────
def include_object(object, name, type_, reflected, compare_to):
    """Only manage tables defined in our models. Ignore checkpoints, langchain_pg_embedding, etc."""
    if type_ == "table":
        if name in target_metadata.tables:
            return True
        else:
            return False  # Skip unmanaged tables (don't drop or alter them)

    if hasattr(object, 'table') and object.table.name in target_metadata.tables:
        return True
    return False


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using synchronous engine."""
    connectable = create_engine(
        get_sync_url(),
        poolclass=pool.NullPool,        # Important for Alembic
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            render_as_batch=True,       # Safer for PostgreSQL
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()