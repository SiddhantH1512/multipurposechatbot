# backend/thread_service.py

from typing import Any, Dict
import uuid
import streamlit as st


# ── Neutral location for these two mutable globals ──
# (no dependencies → no cycle risk)
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def generate_thread():
    return uuid.uuid4()


def add_threads(thread_id, title="New Chat"):
    if thread_id not in st.session_state.get('chat_threads', []):
        st.session_state['chat_threads'] = st.session_state.get('chat_threads', []) + [thread_id]
        st.session_state['chat_titles'] = st.session_state.get('chat_titles', {})
        st.session_state['chat_titles'][thread_id] = title


def load_conversation(thread_id):
    # Import only when actually needed
    from backend.langgraph_backend import chatbot
    
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get('messages', [])


def generate_chat_title(first_message):
    # Import only when actually needed
    from backend.langgraph_backend import model
    
    prompt = f"Summarize this chat request into a 3-word title: {first_message}"
    title = model.invoke(prompt).content
    return title.strip('"')


def reset_state():
    thread_id = generate_thread()
    st.session_state['thread_id'] = thread_id
    add_threads(thread_id, title="New Chat")
    st.session_state['message_history'] = []


def retrieve_all_threads():
    from backend.langgraph_backend import checkpointer
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


def delete_thread(thread_id):
    """
    Permanently delete all checkpoints for a given thread_id
    and remove it from session state tracking
    """
    try:
        # Import only when actually needed
        from backend.langgraph_backend import checkpointer
        
        conn = checkpointer.conn
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            (str(thread_id),)
        )
        conn.commit()
        
        cursor.execute(
            "DELETE FROM writes WHERE thread_id = ?",
            (str(thread_id),)
        )
        conn.commit()
        
    except Exception as e:
        print(f"Error deleting thread {thread_id}: {e}")

    # Clean session state
    threads = st.session_state.get('chat_threads', [])
    if thread_id in threads:
        threads.remove(thread_id)
        st.session_state['chat_threads'] = threads
    
    st.session_state['chat_titles'] = st.session_state.get('chat_titles', {})
    st.session_state['chat_titles'].pop(thread_id, None)