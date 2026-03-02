# src/backend/thread_service.py

import sqlite3
from typing import Any, Dict, List, Optional
from src.backend.langgraph_backend import chatbot 

_THREAD_METADATA_TABLE = "thread_metadata"


def retrieve_all_threads() -> List[str]:
    """List all thread_ids that exist in the checkpointer."""
    all_threads = set()
    for checkpoint in chatbot.checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def load_conversation(thread_id: str) -> list:
    """Load messages for a given thread from checkpointer."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get('messages', [])


def thread_has_document(thread_id: str) -> bool:
    """Check if this thread has an indexed document (FAISS / vector store)."""
    conn = sqlite3.connect("chatbot.db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT 1 FROM {_THREAD_METADATA_TABLE} WHERE thread_id = ?", (str(thread_id),))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def thread_document_metadata(thread_id: str) -> Dict[str, Any]:
    """Get metadata for the document indexed in this thread."""
    conn = sqlite3.connect("chatbot.db")
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT filename, documents, chunks FROM {_THREAD_METADATA_TABLE} WHERE thread_id = ?",
        (str(thread_id),)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "filename": row[0],
            "documents": row[1],
            "chunks": row[2],
        }
    return {}


def delete_thread(thread_id: str):
    """Delete thread from checkpointer + metadata table + session state."""
    # Delete from checkpointer
    try:
        conn = chatbot.checkpointer.conn
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(thread_id),))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (str(thread_id),))
        conn.commit()
    except Exception as e:
        print(f"Checkpointer delete failed for {thread_id}: {e}")

    # Delete metadata
    try:
        conn = sqlite3.connect("chatbot.db")
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {_THREAD_METADATA_TABLE} WHERE thread_id = ?", (str(thread_id),))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Metadata delete failed: {e}")

    # Clean Streamlit session (if exists)
    import streamlit as st
    threads = st.session_state.get('chat_threads', [])
    if thread_id in threads:
        threads.remove(thread_id)
        st.session_state['chat_threads'] = threads

    titles = st.session_state.get('thread_titles', {})
    titles.pop(thread_id, None)