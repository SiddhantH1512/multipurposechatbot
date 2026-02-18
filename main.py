import streamlit as st
from backend.langgraph_backend import chatbot  
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from utils import generate_chat_title, generate_thread, add_threads, reset_state, load_conversation, retrieve_thread



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
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_state()
    st.rerun() 

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    
    logical_name = st.session_state['chat_titles'].get(thread_id, f"Chat {str(thread_id)[:6]}")

    if st.sidebar.button(logical_name, key=f"btn_{thread_id}", use_container_width=True):
        st.session_state['thread_id'] = thread_id
        
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
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

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    if len(st.session_state['message_history']) == 2:
        first_user_msg = st.session_state['message_history'][0]['content']
        new_title = generate_chat_title(first_user_msg)
        st.session_state['chat_titles'][st.session_state['thread_id']] = new_title
        st.rerun()


