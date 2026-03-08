from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import create_engine, Engine
from src.config import Config

# One engine — async version (preferred for FastAPI)
async_engine: AsyncEngine = create_async_engine(
    Config.POSTGRES_CONNECTION,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_pre_ping=True,
)

# Optional: sync fallback engine (for Streamlit / non-async code)
# Only create if really needed — avoid if possible
sync_engine: Engine = create_engine(
    Config.POSTGRES_CONNECTION,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,
)

__all__ = ["async_engine", "sync_engine"]