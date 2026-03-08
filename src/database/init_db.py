# src/database/init_db.py
import asyncio
from sqlalchemy import text
from src.database.engine import async_engine  # your async engine

async def init_all_tables():
    async with async_engine.begin() as conn:
        # ────────────────────────────────────────────────
        # 1. Your application metadata table
        # ────────────────────────────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS thread_metadata (
                thread_id TEXT PRIMARY KEY,
                filename TEXT,
                documents INTEGER DEFAULT 0,
                chunks INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_thread_metadata_created_at
            ON thread_metadata (created_at DESC);
        """))

        # ────────────────────────────────────────────────
        # 2. LangGraph checkpointer tables - CORRECTED SCHEMA
        # ────────────────────────────────────────────────
        
        # Checkpoints table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint JSONB NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );
        """))

        # Checkpoint writes table (with all required columns)
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                blob BYTEA NOT NULL,
                task_path TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            );
        """))

        # Checkpoint blobs table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                channel TEXT NOT NULL,
                version TEXT NOT NULL,
                type TEXT NOT NULL,
                blob BYTEA,
                PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
            );
        """))

        # ────────────────────────────────────────────────
        # Indexes for performance
        # ────────────────────────────────────────────────
        
        # Indexes for checkpoints
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id
            ON checkpoints (thread_id);
        """))

        # Indexes for checkpoint_writes
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread_id
            ON checkpoint_writes (thread_id);
        """))

        # Indexes for checkpoint_blobs
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_checkpoint_blobs_thread_id
            ON checkpoint_blobs (thread_id);
        """))

        # Optional: Composite indexes for common query patterns
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_ns_id
            ON checkpoints (thread_id, checkpoint_ns, checkpoint_id);
        """))

        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread_ns_id
            ON checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id);
        """))

        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_checkpoint_blobs_thread_ns_channel
            ON checkpoint_blobs (thread_id, checkpoint_ns, channel, version);
        """))

    print("✅ All tables initialized successfully:")
    print("  - thread_metadata (custom)")
    print("  - checkpoints (LangGraph)")
    print("  - checkpoint_writes (LangGraph)")
    print("  - checkpoint_blobs (LangGraph)")
    print("  + all required indexes created")

if __name__ == "__main__":
    asyncio.run(init_all_tables())