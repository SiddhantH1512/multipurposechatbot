from turtle import st
import uuid
import streamlit as st
from backend.langgraph_backend import chatbot, model, checkpointer


def generate_thread():
    thread_id = uuid.uuid4()
    return thread_id

def add_threads(thread_id, title="New Chat"):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        st.session_state['chat_titles'][thread_id] = title


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable":{"thread_id": thread_id}})
    return state.values.get('messages', [])


def generate_chat_title(first_message):
    prompt = f"Summarize this chat request into a 3-word title: {first_message}"
    title = model.invoke(prompt).content
    return title.strip('"')


def reset_state():
    thread_id = generate_thread()
    st.session_state['thread_id'] = thread_id
    add_threads(thread_id, title="New Chat")
    st.session_state['message_history'] = []


def retrieve_thread():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id']) 

    return list(all_threads)