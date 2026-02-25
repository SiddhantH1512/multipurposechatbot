import asyncio
import queue

from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from backend.langgraph_backend import chatbot  
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from backend.thread_service import delete_thread, generate_chat_title, generate_thread, add_threads, reset_state, load_conversation, retrieve_thread



# SESSION SETUP
if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles'] = {}

if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = generate_thread()

if "message_history" not in st.session_state:
    st.session_state['message_history'] = []

if "chat_threads" not in st.session_state:
    st.session_state['chat_threads'] = retrieve_thread()

add_threads(st.session_state['thread_id'])


# STREAMLIT UI - SIDEBAR
# st.sidebar.title('LangGraph Chatbot')

# if st.sidebar.button('New Chat'):
#     reset_state()
#     st.rerun() 

# st.sidebar.header('My Conversations')

# for thread_id in st.session_state['chat_threads'][::-1]:
    
#     logical_name = st.session_state['chat_titles'].get(thread_id, f"Chat {str(thread_id)[:6]}")

#     if st.sidebar.button(logical_name, key=f"btn_{thread_id}", use_container_width=True):
#         st.session_state['thread_id'] = thread_id
        
#         messages = load_conversation(thread_id)

#         temp_messages = []
#         for msg in messages:
#             if isinstance(msg, HumanMessage):
#                 role = 'user'
#             else:
#                 role = 'assistant'
#             temp_messages.append({'role': role, 'content': msg.content})

#         st.session_state['message_history'] = temp_messages
#         st.rerun()
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_state()
    st.rerun()

st.sidebar.header("My Conversations")

for thread_id in list(st.session_state["chat_threads"])[::-1]:  # copy + reverse
    
    title = st.session_state["chat_titles"].get(
        thread_id, 
        f"Chat {str(thread_id)[:8]}…"
    )
    
    # ── Two-column layout: title + delete button ──
    cols = st.sidebar.columns([8, 1])
    
    with cols[0]:
        if st.button(
            title,
            key=f"load_{thread_id}",
            use_container_width=True,
            type="tertiary" if thread_id != st.session_state["thread_id"] else "primary"
        ):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)
            
            # convert to streamlit format
            history = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                history.append({"role": role, "content": msg.content})
                
            st.session_state["message_history"] = history
            st.rerun()
    
    with cols[1]:
        if st.button("🗑", key=f"del_{thread_id}", help="Delete this conversation"):
            delete_thread(thread_id)
            
            # If we just deleted the active thread → create new one
            if thread_id == st.session_state["thread_id"]:
                reset_state()
            
            st.rerun()


# MAIN UI
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.write(user_input) 

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']},
              'metadata': {'thread_id': st.session_state['thread_id']},
              'run_name': 'chat_run'}

    with st.chat_message("assistant"):

        if "async_loop" not in st.session_state:
            st.session_state.async_loop = asyncio.new_event_loop()

        def ai_only_stream():

            async def run():
                async for message_chunk, metadata in chatbot.astream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, AIMessage):
                        yield message_chunk.content

            agen = run()

            while True:
                try:
                    chunk = st.session_state.async_loop.run_until_complete(
                        agen.__anext__()
                    )
                    yield chunk
                except StopAsyncIteration:
                    break

    ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    if len(st.session_state['message_history']) == 2:
        first_user_msg = st.session_state['message_history'][0]['content']
        new_title = generate_chat_title(first_user_msg)
        st.session_state['chat_titles'][st.session_state['thread_id']] = new_title
        st.rerun()


