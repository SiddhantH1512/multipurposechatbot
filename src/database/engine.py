from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, Engine, text
from src.config import Config
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from src.database.table_models import User

# Derive a sync connection string from the async one.
# POSTGRES_CONNECTION is expected to be postgresql+asyncpg://...
# The sync engine (used by PGVector and thread_service) needs psycopg2.
_async_url: str = Config.POSTGRES_CONNECTION or ""
SYNC_CONNECTION_STRING: str = (
    _async_url
    .replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    .replace("postgresql+aiopg://", "postgresql+psycopg2://")
)

# Global async engine
async_engine: AsyncEngine = create_async_engine(
    Config.POSTGRES_CONNECTION,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_pre_ping=True,
)

# Session factory
async_session_factory = async_sessionmaker(
    bind=async_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

# ──────────────────────────────────────────────────────────────
# 1. Context manager (for manual use if needed)
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# ──────────────────────────────────────────────────────────────
# 2. FastAPI Dependency (THIS IS WHAT YOU MUST USE WITH Depends())
# ──────────────────────────────────────────────────────────────
async def get_async_session_dep() -> AsyncGenerator[AsyncSession, None]:
    """
    This is the correct dependency for FastAPI.
    Use: session: AsyncSession = Depends(get_async_session_dep)
    """
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# Sync engine (for LangGraph checkpointer + PGVector)
# Must use psycopg2 driver, NOT asyncpg
sync_engine: Engine = create_engine(
    SYNC_CONNECTION_STRING,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,
)

@asynccontextmanager
async def rls_context(session: AsyncSession, user: User):
    """
    Sets RLS session variables for row-level security.
    """
    dept = user.department or "General"
    role_val = user.role.value if hasattr(user.role, "value") else str(user.role)

    try:
        await session.execute(
            text(f"SET LOCAL app.current_user_department = '{dept}'")
        )
        await session.execute(
            text(f"SET LOCAL app.current_user_role = '{role_val}'")
        )
        await session.execute(
            text(f"SET LOCAL app.current_user_id = '{user.id}'")
        )
        print(f"[RLS] Set department={dept}, role={role_val}, user_id={user.id}")
        yield session
    finally:
        pass

__all__ = [
    "async_engine", "sync_engine", "SYNC_CONNECTION_STRING",
    "get_async_session", "get_async_session_dep", "rls_context"
]