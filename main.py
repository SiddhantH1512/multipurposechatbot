import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.langgraph_backend import chatbot, ingest_pdf
from backend.thread_service import retrieve_all_threads, thread_document_metadata


# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())  # return string directly


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    tid = str(thread_id)
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    return state.values.get("messages", [])


def get_thread_display_name(thread_id):
    tid = str(thread_id)
    titles = st.session_state.get("thread_titles", {})
    if tid in titles:
        return titles[tid]
    short = tid[:8].upper()
    return f"Chat • {short}"


def set_thread_title_from_first_message(thread_id, messages=None):
    tid = str(thread_id)
    if tid in st.session_state.get("thread_titles", {}):
        return

    if messages is None:
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


def delete_thread(thread_id_to_delete):
    tid = str(thread_id_to_delete)
    
    threads = st.session_state.get("chat_threads", [])
    if tid in threads:
        threads.remove(tid)
        st.session_state["chat_threads"] = threads

    st.session_state["thread_titles"].pop(tid, None)
    st.session_state["ingested_docs"].pop(tid, None)

    checkpointer = chatbot.checkpointer
    if checkpointer is not None:
        try:
            checkpointer.delete_thread(tid)
        except Exception as e:
            st.warning(f"DB delete failed for {tid}: {e}")

    if str(st.session_state.get("thread_id", "")) == tid:
        reset_chat()

    st.rerun()


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
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

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
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    state = chatbot.get_state(config={"configurable": {"thread_id": thread_key}})
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