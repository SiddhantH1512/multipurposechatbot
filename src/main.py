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

# Extra safety: remove duplicates again
threads_set = set(str(t).strip() for t in st.session_state["chat_threads"] if t)
st.session_state["chat_threads"] = list(threads_set)

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]  # newest first
selected_thread = None


if "token" not in st.session_state or not st.session_state.get("token"):
    st.title("Login to PolicyIQ")
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if not email or not password:
            st.error("Please enter email and password")
        else:
            try:
                resp = httpx.post(
                    f"{API_BASE_URL}/auth/token",
                    data={"username": email, "password": password},
                    timeout=10.0
                )
                if resp.status_code == 200:
                    tokens = resp.json()
                    st.session_state["token"] = tokens["access_token"]
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error(f"Login failed: {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")
    
    st.stop()



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

# ── HR-only document upload ──
if "is_hr" not in st.session_state:
    st.session_state["is_hr"] = False

# Fetch user role once (runs only if token exists and role not yet loaded)
if st.session_state.get("token") and not st.session_state["is_hr"]:
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        resp = httpx.get(f"{API_BASE_URL}/auth/me", headers=headers, timeout=5.0)
        if resp.status_code == 200:
            user = resp.json()
            st.session_state["user_role"] = user.get("role")
            st.session_state["is_hr"] = user.get("role") == "HR"
            st.session_state["user_department"] = user.get("department")
        else:
            st.session_state["is_hr"] = False
    except Exception as e:
        print(f"Failed to fetch user role: {e}")
        st.session_state["is_hr"] = False

# Now conditionally show the uploader
if st.session_state.get("is_hr", False):
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload HR policy document (global)",
        type=["pdf"],
        help="Only HR users can upload organization-wide documents."
    )

    if uploaded_pdf:
        thread_key = str(st.session_state["thread_id"])

        # Optional: still check if already processed (per-thread UI feedback)
        if uploaded_pdf.name in st.session_state["ingested_docs"].get(thread_key, {}):
            st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
        else:
            with st.sidebar.status("Uploading & Indexing PDF…", expanded=True) as status_box:
                try:
                    files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                    data = {"thread_id": thread_key}

                    headers = {"Authorization": f"Bearer {st.session_state.token}"}

                    response = httpx.post(
                        f"{API_BASE_URL}/ingest",
                        files=files,
                        data=data,
                        headers=headers,
                        timeout=httpx.Timeout(300.0, connect=10.0, read=300.0)
                    )

                    if response.status_code == 200:
                        result = response.json()
                        summary = result["summary"]
                        
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

elif not st.session_state.get("token"):
    st.sidebar.warning("Please log in to access document upload.")
else:
    st.sidebar.info("📄 Document upload is restricted to HR users only.")
    uploaded_pdf = None

st.sidebar.subheader("Past conversations")

if not threads:
    st.sidebar.write("No past conversations yet.")
else:
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

        select_key = f"select-{tid}"
        delete_key = f"delete-{tid}"

        if select_key in seen_keys or delete_key in seen_keys:
            continue

        seen_keys.add(select_key)
        seen_keys.add(delete_key)

        col1, col2 = st.sidebar.columns([8, 1])

        if col1.button(display_name, key=select_key, use_container_width=True):
            st.session_state["selected_thread_temp"] = tid

        if col2.button("🗑", key=delete_key, help="Delete this conversation"):
            success = delete_thread(tid)
            if success:
                # Remove from session state immediately
                if tid in st.session_state["chat_threads"]:
                    st.session_state["chat_threads"].remove(tid)
                
                # Clean up related session state
                thread_key_del = str(tid)
                if thread_key_del in st.session_state["ingested_docs"]:
                    del st.session_state["ingested_docs"][thread_key_del]
                if thread_key_del in st.session_state["thread_titles"]:
                    del st.session_state["thread_titles"][thread_key_del]
                
                st.success(f"Deleted {display_name}")
                st.rerun()  # ← Critical: refresh UI with updated thread list
            else:
                st.error("Failed to delete conversation")

# Handle thread selection
if "selected_thread_temp" in st.session_state:
    selected_thread = st.session_state["selected_thread_temp"]
    del st.session_state["selected_thread_temp"]


if st.sidebar.button("Logout"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Display history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            with httpx.stream("POST", f"{API_BASE_URL}/chat",
                              data={"message": user_input, "thread_id": thread_key},
                              timeout=300.0) as response:
                response.raise_for_status()
                for chunk in response.iter_text():
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state["message_history"].append({"role": "assistant", "content": full_response})
        except Exception as e:
            message_placeholder.error(str(e))

    # Refresh title and doc info
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_key}})
        messages_in_graph = state.values.get("messages", [])
        set_thread_title_from_first_message(thread_key, messages_in_graph)
    except Exception as e:
        print(f"Warning: Could not refresh title: {e}")

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

# Switch to selected thread
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    thread_key = str(selected_thread)

    messages = load_conversation(selected_thread)

    set_thread_title_from_first_message(selected_thread, messages)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content.strip():
            temp_messages.append({"role": "assistant", "content": msg.content})
        # ToolMessages ignored for UI display

    st.session_state["message_history"] = temp_messages

    st.session_state["ingested_docs"].setdefault(thread_key, {})

    st.rerun()