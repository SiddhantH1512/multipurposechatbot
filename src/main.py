import httpx
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from backend.langgraph_backend import chatbot, ingest_pdf
from backend.thread_service import delete_thread, load_conversation, retrieve_all_threads, thread_document_metadata
from src.backend.utils import add_thread, generate_thread_id, get_thread_display_name, reset_chat, set_thread_title_from_first_message

API_BASE_URL = "http://127.0.0.1:8000"


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    raw = retrieve_all_threads() or []
    seen = set()
    cleaned = []
    for t in raw:
        tid = str(t).strip()
        if tid and tid not in seen:
            cleaned.append(tid)
            seen.add(tid)
    st.session_state["chat_threads"] = cleaned

# Extra safety: remove duplicates again (in case of reload / bug)
threads_set = set(str(t).strip() for t in st.session_state["chat_threads"] if t)
st.session_state["chat_threads"] = list(threads_set)

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    thread_key = str(st.session_state["thread_id"])

    # Check if already processed (optional – we can improve this later)
    if uploaded_pdf.name in st.session_state["ingested_docs"].get(thread_key, {}):
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Uploading & Indexing PDF…", expanded=True) as status_box:
            try:
                files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                data = {"thread_id": thread_key}

                response = httpx.post(
                    f"{API_BASE_URL}/ingest",
                    files=files,
                    data=data,
                    timeout=httpx.Timeout(300.0, connect=10.0, read=300.0)
                )

                if response.status_code == 200:
                    result = response.json()
                    summary = result["summary"]
                    
                    # Update local session state (for UI display)
                    st.session_state["ingested_docs"].setdefault(thread_key, {})[uploaded_pdf.name] = summary
                    
                    status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
                    st.sidebar.success(
                        f"Indexed `{summary.get('filename')}` "
                        f"({summary.get('chunks')} chunks, {summary.get('documents')} pages)"
                    )
                else:
                    error_msg = response.json().get("detail", "Unknown error")
                    status_box.update(label=f"❌ Failed: {error_msg}", state="error", expanded=True)
                    st.sidebar.error(f"Ingestion failed: {error_msg}")

            except Exception as e:
                status_box.update(label=f"❌ Error: {str(e)}", state="error", expanded=True)
                st.sidebar.error(f"Upload error: {str(e)}")

st.sidebar.subheader("Past conversations")

if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    # Prevent duplicate keys by using a seen set
    seen_keys = set()

    for thread_id in threads:
        tid = str(thread_id)

        # Auto-set title if missing
        if tid not in st.session_state.get("thread_titles", {}):
            try:
                state = chatbot.get_state({"configurable": {"thread_id": tid}})
                messages = state.values.get("messages", [])
                set_thread_title_from_first_message(tid, messages)
            except Exception:
                pass

        display_name = get_thread_display_name(tid)

        # Make keys unique even if duplicate thread_id exists
        select_key = f"select-{tid}"
        delete_key = f"delete-{tid}"

        if select_key in seen_keys or delete_key in seen_keys:
            # Skip rendering if key would collide (safety)
            continue

        seen_keys.add(select_key)
        seen_keys.add(delete_key)

        col1, col2 = st.sidebar.columns([8, 1])

        if col1.button(display_name, key=select_key, use_container_width=True):
            selected_thread = tid

        if col2.button("🗑", key=delete_key, help="Delete this conversation"):
            delete_thread(tid)


# Handle deferred selection (if you still want to use temp flag)
if "selected_thread_temp" in st.session_state:
    selected_thread = st.session_state["selected_thread_temp"]
    del st.session_state["selected_thread_temp"]

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()           # this will be updated live
            full_response = ""

            try:
                with httpx.stream(
                    "POST",
                    f"{API_BASE_URL}/chat",
                    data={"message": user_input, "thread_id": thread_key},
                    timeout=300.0
                ) as response:
                    response.raise_for_status()  # raise if 4xx/5xx

                    for chunk in response.iter_text():
                        if chunk:
                            full_response += chunk
                            # Show typing cursor effect
                            message_placeholder.markdown(full_response + "▌")

                # Final clean render (remove cursor)
                message_placeholder.markdown(full_response)

                # Save to history
                st.session_state["message_history"].append(
                    {"role": "assistant", "content": full_response}
                )

            except httpx.TimeoutException:
                message_placeholder.error("Request timed out — the model might be slow or busy.")
            except httpx.HTTPStatusError as e:
                message_placeholder.error(f"API error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                message_placeholder.error(f"Unexpected error: {str(e)}")

    # Refresh title and doc info (same as before)
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_key}})  # ← temporary, we'll fix later
    messages_in_graph = state.values.get("messages", [])
    set_thread_title_from_first_message(thread_key, messages_in_graph)

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    thread_key = str(selected_thread)

    messages = load_conversation(selected_thread)

    set_thread_title_from_first_message(selected_thread, messages)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "assistant"
        else:
            continue
        temp_messages.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(thread_key, {})

    st.rerun()