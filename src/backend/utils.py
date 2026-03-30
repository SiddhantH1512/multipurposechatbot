# src/backend/utils.py

import uuid
import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage
def get_llm_for_title_generation():
    from src.backend.langgraph_backend import llm
    return llm

def generate_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str):
    tid = str(thread_id)
    if tid not in st.session_state.get("chat_threads", []):
        st.session_state.setdefault("chat_threads", []).append(tid)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def get_thread_display_name(thread_id: str) -> str:
    tid = str(thread_id)
    titles = st.session_state.get("thread_titles", {})
    if tid in titles:
        return titles[tid]
    short = tid[:8].upper()
    return f"Chat • {short}"


def set_thread_title_from_first_message(thread_id: str, messages: list[BaseMessage] | None = None):
    tid = str(thread_id)
    if tid in st.session_state.get("thread_titles", {}):
        return

    if messages is None:
        from src.backend.langgraph_backend import chatbot
        try:
            state = chatbot.get_state({"configurable": {"thread_id": tid}})
            messages = state.values.get("messages", [])
        except Exception:
            return

    if not messages:
        return

    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content and msg.content.strip():
            title = msg.content.strip().split('\n')[0][:60].strip()
            if len(title) == 60:
                title += "…"
            title = " ".join(title.split())
            if title:
                st.session_state.setdefault("thread_titles", {})[tid] = title
            return


# Optional: if you want to keep LLM-based title (less common now)
def generate_chat_title(first_message: str) -> str:
    prompt = f"Summarize this chat request into a short 3-6 word title: {first_message}"
    try:
        llm = get_llm_for_title_generation()
        title = llm.invoke(prompt).content.strip('"').strip()
        return title
    except Exception:
        return "New Chat"