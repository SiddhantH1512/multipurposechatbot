# src/backend/thread_service.py

from typing import List
from sqlalchemy import text
from src.database.engine import sync_engine


_THREAD_METADATA_TABLE = "thread_metadata"


def retrieve_all_threads():
    from src.database.engine import sync_engine  # assuming you have this
    
    with sync_engine.connect() as conn:
        result = conn.execute(
            text("SELECT DISTINCT thread_id FROM thread_metadata")
        ).fetchall()
    
    return [row[0] for row in result] if result else []


def load_conversation(thread_id: str) -> list:
    """Load messages for a given thread from checkpointer."""
    from src.backend.langgraph_backend import chatbot 
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get('messages', [])


def thread_has_document(thread_id: str) -> bool:
    """
    Check if this thread has an indexed document in thread_metadata.
    Returns: True if a row exists for this thread_id, False otherwise.
    """
    try:
        with sync_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 1 
                FROM thread_metadata 
                WHERE thread_id = :tid
            """), {"tid": str(thread_id)})
            
            # If fetchone() returns a row → exists
            exists = result.fetchone() is not None
            
        return exists
        
    except Exception as e:
        print(f"Error checking document for thread {thread_id}: {e}")
        return False   # safe default: assume no document if DB fails


def thread_document_metadata(thread_id: str) -> dict:
    with sync_engine.connect() as conn:
        result = conn.execute(text("""
            SELECT filename, documents, chunks 
            FROM thread_metadata 
            WHERE thread_id = :tid
        """), {"tid": str(thread_id)})
        
        row = result.fetchone()
        if row:
            return {
                "filename": row[0],
                "documents": row[1],
                "chunks": row[2],
            }
    return {}


def delete_thread(thread_id: str) -> bool:
    from src.database.engine import sync_engine
    from sqlalchemy import text

    try:
        with sync_engine.begin() as conn:
            conn.execute(
                text("DELETE FROM thread_metadata WHERE thread_id = :tid"),
                {"tid": thread_id}
            )
            conn.execute(
                text("DELETE FROM checkpoints WHERE thread_id = :tid"),
                {"tid": thread_id}
            )
            conn.execute(
                text("DELETE FROM checkpoint_writes WHERE thread_id = :tid"),
                {"tid": thread_id}
            )
            conn.execute(
                text("DELETE FROM checkpoint_blobs WHERE thread_id = :tid"),
                {"tid": thread_id}
            )
            conn.execute(
                text("""
                    DELETE FROM langchain_pg_embedding 
                    WHERE cmetadata ->> 'thread_id' = :tid
                """),
                {"tid": thread_id}
            )

        print(f"Thread {thread_id} deleted successfully (metadata + checkpoints + writes + blobs + vectors)")
        return True

    except Exception as e:
        print(f"Failed to delete thread {thread_id}: {str(e)}")
        return False