from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, Engine
from src.config import Config
from contextlib import asynccontextmanager
from typing import AsyncGenerator

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


# Sync engine (for LangGraph checkpointer)
sync_engine: Engine = create_engine(
    Config.POSTGRES_CONNECTION,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,
)

__all__ = ["async_engine", "sync_engine", "get_async_session", "get_async_session_dep"]