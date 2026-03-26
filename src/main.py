import httpx
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from backend.langgraph_backend import chatbot, ingest_pdf
from backend.thread_service import delete_thread, load_conversation, retrieve_all_threads, thread_document_metadata
from src.backend.utils import add_thread, generate_thread_id, get_thread_display_name, reset_chat, set_thread_title_from_first_message

API_BASE_URL = "http://127.0.0.1:8000"

# ======================= Session Initialization ===================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "authenticated": False,
        "token": None,
        "user_email": None,
        "user_role": None,
        "user_department": None,
        "user_id": None,
        "is_hr": False,
        "message_history": [],
        "thread_id": generate_thread_id(),
        "chat_threads": [],
        "threads_metadata": {},
        "ingested_docs": {},
        "thread_titles": {},
        "docs_list": None,
        "login_counter": 0  # Add counter to force refresh
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
init_session_state()

# ======================= LOGIN SCREEN ===================
def show_login_screen():
    """Display login screen only"""
    st.set_page_config(page_title="PolicyIQ - Login", layout="centered")
    
    st.title("🔐 PolicyIQ")
    st.markdown("### Welcome to PolicyIQ Chatbot")
    st.markdown("Please login to continue")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if not email or not password:
                st.error("Please enter both email and password")
                return
            
            try:
                # Step 1: Get token
                response = httpx.post(
                    f"{API_BASE_URL}/auth/token",
                    data={"username": email, "password": password},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.user_email = email
                    st.session_state.authenticated = True
                    
                    # Step 2: Fetch user profile
                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                    profile_resp = httpx.get(f"{API_BASE_URL}/auth/me", headers=headers, timeout=5.0)
                    if profile_resp.status_code == 200:
                        user = profile_resp.json()
                        st.session_state.user_role = user.get("role")
                        st.session_state.user_department = user.get("department")
                        st.session_state.user_id = user.get("id")
                        st.session_state.is_hr = user.get("role") == "HR"
                        
                        # Debug print to console
                        print(f"Login: {email} - Role: {user.get('role')} - is_hr: {st.session_state.is_hr}")
                    
                    # Step 3: Fetch user's threads
                    try:
                        threads_resp = httpx.get(
                            f"{API_BASE_URL}/threads",
                            headers=headers,
                            timeout=5.0
                        )
                        if threads_resp.status_code == 200:
                            threads_data = threads_resp.json()
                            st.session_state.chat_threads = [t["thread_id"] for t in threads_data["threads"]]
                            # Store metadata for display
                            for t in threads_data["threads"]:
                                if t["metadata"]:
                                    st.session_state.threads_metadata[t["thread_id"]] = t["metadata"]
                    except Exception as e:
                        print(f"Error fetching threads: {e}")
                        st.session_state.chat_threads = []
                    
                    # Step 4: Create initial thread if none exists
                    if not st.session_state.chat_threads:
                        add_thread(st.session_state.thread_id)
                    
                    # Increment login counter to force refresh
                    st.session_state.login_counter += 1
                    
                    st.success(f"✅ Login successful as {email} ({st.session_state.user_role})!")
                    st.rerun()
                else:
                    error_detail = response.json().get("detail", "Invalid credentials")
                    st.error(f"Login failed: {error_detail}")
            except httpx.ConnectError:
                st.error(f"❌ Cannot connect to server at {API_BASE_URL}. Make sure the backend is running.")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")
    
    # Optional: Add a note about demo accounts
    with st.expander("📋 Demo Accounts"):
        st.markdown("""
        **HR Users:**
        - hr@example.com / Test@123456
        - hr2@company.com / Test@123456
        - hr3@company.com / Test@123456
        
        **Engineering:**
        - sarah.eng@company.com / Test@123456
        - mike.eng@company.com / Test@123456
        - alex.eng@company.com / Test@123456
        
        **Sales:**
        - lisa.sales@company.com / Test@123456
        - john.sales@company.com / Test@123456
        
        **Marketing:**
        - jessica.marketing@company.com / Test@123456
        
        **Finance:**
        - robert.finance@company.com / Test@123456
        
        **Executive:**
        - ceo@company.com / Test@123456
        - cto@company.com / Test@123456
        """)

# ======================= MAIN CHAT UI ===================
def show_chat_ui():
    """Display the main chat interface for authenticated users"""
    st.set_page_config(page_title="PolicyIQ Chatbot", layout="wide")
    
    # Debug info in sidebar to verify state
    print(f"Current user: {st.session_state.user_email}, Role: {st.session_state.user_role}, is_hr: {st.session_state.is_hr}")
    
    thread_key = str(st.session_state["thread_id"])
    thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
    
    # ============================ Sidebar ============================
    st.sidebar.title("💬 PolicyIQ Chatbot")
    st.sidebar.markdown(f"**👤 User:** `{st.session_state.user_email}`")
    st.sidebar.markdown(f"**🎭 Role:** `{st.session_state.user_role}`")
    st.sidebar.markdown(f"**🏢 Department:** `{st.session_state.user_department}`")
    st.sidebar.markdown(f"**🔐 HR Status:** `{'✅ HR' if st.session_state.is_hr else '❌ Not HR'}`")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**🔑 Thread ID:** `{thread_key[:8]}...`")
    
    # New Chat button
    if st.sidebar.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        reset_chat()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Document info display
    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        visibility_icon = "🌍" if latest_doc.get('visibility') == 'global' else "🏢"
        st.sidebar.success(
            f"{visibility_icon} Using `{latest_doc.get('filename')}` "
            f"({latest_doc.get('chunks')} chunks, {latest_doc.get('documents')} pages)"
        )
    else:
        st.sidebar.info("📄 No PDF indexed yet.")
    
    st.sidebar.markdown("---")
    
    # ── HR-only document upload ──
    # IMPORTANT: Force evaluation of is_hr for each render
    is_hr = st.session_state.get("is_hr", False)
    
    if is_hr:
        st.sidebar.markdown("### 📤 Document Upload")
        st.sidebar.success("✅ HR privileges detected - Upload enabled")
        
        # Visibility selector for HR
        visibility = st.sidebar.selectbox(
            "Document Visibility",
            ["global", "dept"],
            help="Global: Everyone can see | Department: Only specified department",
            key=f"visibility_selector_{st.session_state.login_counter}"
        )
        
        target_department = None
        if visibility == "dept":
            target_department = st.sidebar.selectbox(
                "Target Department",
                ["Engineering", "Sales", "Marketing", "Finance", "HR", "Executive"],
                help="Which department should have access to this document?",
                key=f"target_dept_selector_{st.session_state.login_counter}"
            )
        
        uploaded_pdf = st.sidebar.file_uploader(
            "Upload HR policy document",
            type=["pdf"],
            help="Only HR users can upload organization-wide documents.",
            key=f"pdf_uploader_{st.session_state.login_counter}"
        )
        
        if uploaded_pdf:
            if uploaded_pdf.name in thread_docs:
                st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
            else:
                with st.sidebar.status("Uploading & Indexing PDF…", expanded=True) as status_box:
                    try:
                        files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                        data = {
                            "thread_id": thread_key,
                            "visibility": visibility
                        }
                        if target_department:
                            data["target_department"] = target_department
                        
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
                            summary["visibility"] = visibility
                            if target_department:
                                summary["target_department"] = target_department
                            
                            st.session_state["ingested_docs"].setdefault(thread_key, {})[uploaded_pdf.name] = summary
                            
                            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
                            st.sidebar.success(
                                f"✅ Indexed `{summary.get('filename')}` "
                                f"({summary.get('chunks')} chunks, {summary.get('documents')} pages) "
                                f"[Visibility: {visibility}{f' → {target_department}' if target_department else ''}]"
                            )
                        else:
                            error_msg = response.json().get("detail", "Unknown error")
                            status_box.update(label=f"❌ Failed: {error_msg}", state="error", expanded=True)
                            st.sidebar.error(f"Ingestion failed: {error_msg}")
                    
                    except Exception as e:
                        status_box.update(label=f"❌ Error: {str(e)}", state="error", expanded=True)
                        st.sidebar.error(f"Upload error: {str(e)}")
    
    elif not st.session_state.get("token"):
        st.sidebar.warning("⚠️ Please log in to access document upload.")
    else:
        st.sidebar.info("📄 Document upload is restricted to HR users only.")
        # Debug: Show why it's restricted
        st.sidebar.caption(f"Debug: User role = {st.session_state.user_role}, is_hr = {is_hr}")
    

    st.sidebar.markdown("---")

    # ── Document Manager (HR) / Document Access (others) ──────────────
    if is_hr:
        st.sidebar.markdown("### 📋 Document Manager")

        if st.sidebar.button("🔄 Refresh Documents", key="refresh_docs_btn", use_container_width=True):
            st.session_state["docs_list"] = None  # force reload

        # Fetch docs list
        if st.session_state.get("docs_list") is None:
            try:
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                r = httpx.get(f"{API_BASE_URL}/documents", headers=headers, timeout=10.0)
                if r.status_code == 200:
                    st.session_state["docs_list"] = r.json().get("documents", [])
                else:
                    st.session_state["docs_list"] = []
                    st.sidebar.error(f"Failed to load documents: {r.status_code}")
            except Exception as e:
                st.session_state["docs_list"] = []
                st.sidebar.error(f"Error: {e}")

        docs_list = st.session_state.get("docs_list") or []

        if not docs_list:
            st.sidebar.info("No documents ingested yet.")
        else:
            VISIBILITY_ICONS = {"global": "🌍", "dept": "🏢", "confidential": "🔒"}
            DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Executive"]

            for idx, doc in enumerate(docs_list):
                doc_id   = doc.get("document_id", "") or f"doc_{idx}"
                filename = doc.get("filename", "unknown")
                vis      = doc.get("visibility", "global")
                dept     = doc.get("department", "")
                chunks   = doc.get("chunk_count", 0)
                icon     = VISIBILITY_ICONS.get(vis, "📄")

                with st.sidebar.expander(f"{icon} {filename}", expanded=False):
                    st.markdown(f"**Visibility:** `{vis}`")
                    st.markdown(f"**Department:** `{dept}`")
                    st.markdown(f"**Chunks:** `{chunks}`")
                    st.markdown(f"**Uploaded by:** `{doc.get('uploaded_by', 'HR')}`")
                    st.markdown("---")
                    st.markdown("**Change visibility:**")

                    new_vis = st.selectbox(
                        "New visibility",
                        ["global", "dept", "confidential"],
                        index=["global", "dept", "confidential"].index(vis) if vis in ["global", "dept", "confidential"] else 0,
                        key=f"vis_select_{idx}"
                    )

                    new_dept = None
                    if new_vis == "dept":
                        current_dept_idx = DEPARTMENTS.index(dept) if dept in DEPARTMENTS else 0
                        new_dept = st.selectbox(
                            "Target department",
                            DEPARTMENTS,
                            index=current_dept_idx,
                            key=f"dept_select_{idx}"
                        )

                    if st.button("✅ Apply", key=f"apply_vis_{idx}", use_container_width=True):
                        try:
                            headers = {"Authorization": f"Bearer {st.session_state.token}"}
                            payload = {"visibility": new_vis}
                            if new_dept:
                                payload["department"] = new_dept
                            resp = httpx.patch(
                                f"{API_BASE_URL}/documents/{doc_id}/visibility",
                                json=payload,
                                headers=headers,
                                timeout=10.0
                            )
                            if resp.status_code == 200:
                                result = resp.json()
                                st.success(f"✅ Updated: {result.get('message')}")
                                st.session_state["docs_list"] = None  # refresh on next render
                                st.rerun()
                            else:
                                st.error(f"Failed: {resp.json().get('detail', resp.status_code)}")
                        except Exception as e:
                            st.error(f"Error: {e}")
    else:
        # Non-HR: read-only view of accessible documents
        st.sidebar.markdown("### 📂 Accessible Documents")

        if st.sidebar.button("🔄 Refresh", key="refresh_docs_nohr_btn", use_container_width=True):
            st.session_state["docs_list"] = None

        if st.session_state.get("docs_list") is None:
            try:
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                r = httpx.get(f"{API_BASE_URL}/documents", headers=headers, timeout=10.0)
                if r.status_code == 200:
                    st.session_state["docs_list"] = r.json().get("documents", [])
                else:
                    st.session_state["docs_list"] = []
            except Exception:
                st.session_state["docs_list"] = []

        docs_list = st.session_state.get("docs_list") or []
        VISIBILITY_ICONS = {"global": "🌍", "dept": "🏢", "confidential": "🔒"}

        if not docs_list:
            st.sidebar.info("No documents available to you yet.")
        else:
            for doc in docs_list:
                vis  = doc.get("visibility", "global")
                icon = VISIBILITY_ICONS.get(vis, "📄")
                st.sidebar.markdown(
                    f"{icon} **{doc.get('filename', 'unknown')}**  \n"
                    f"<small>{vis} · {doc.get('chunk_count', 0)} chunks</small>",
                    unsafe_allow_html=True
                )

    st.sidebar.markdown("---")
    
    # Past conversations
    st.sidebar.subheader("📝 Past conversations")
    threads = st.session_state.get("chat_threads", [])[::-1]  # newest first
    
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
                    if tid in st.session_state["chat_threads"]:
                        st.session_state["chat_threads"].remove(tid)
                    
                    thread_key_del = str(tid)
                    if thread_key_del in st.session_state["ingested_docs"]:
                        del st.session_state["ingested_docs"][thread_key_del]
                    if thread_key_del in st.session_state["thread_titles"]:
                        del st.session_state["thread_titles"][thread_key_del]
                    
                    st.sidebar.success(f"✅ Deleted {display_name}")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to delete conversation")
    
    st.sidebar.markdown("---")
    
    # Logout button
    if st.sidebar.button("🚪 Logout", use_container_width=True, key="logout_btn"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["docs_list"] = None
        st.rerun()
    
    # Handle thread selection
    if "selected_thread_temp" in st.session_state:
        selected_thread = st.session_state["selected_thread_temp"]
        del st.session_state["selected_thread_temp"]
        
        # Load selected thread
        try:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            resp = httpx.get(
                f"{API_BASE_URL}/threads/{selected_thread}",
                headers=headers,
                timeout=10.0
            )
            if resp.status_code == 200:
                data = resp.json()
                messages = data["messages"]
                temp_messages = []
                for msg in messages:
                    if msg["role"] == "user":
                        temp_messages.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        temp_messages.append({"role": "assistant", "content": msg["content"]})
                st.session_state["message_history"] = temp_messages
                st.session_state["thread_id"] = selected_thread
                st.rerun()
            else:
                st.error(f"Failed to load conversation: {resp.status_code}")
        except Exception as e:
            st.error(f"Error loading conversation: {e}")
    
    # ============================ Main Chat Area ========================
    st.title("💬 Multi Utility Chatbot")
    
    # Display chat history
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about your document or use tools")
    
    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                with httpx.stream("POST", f"{API_BASE_URL}/chat",
                                  data={"message": user_input, "thread_id": thread_key},
                                  headers=headers,
                                  timeout=300.0) as response:
                    response.raise_for_status()
                    for chunk in response.iter_text():
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state["message_history"].append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state["message_history"].append({"role": "assistant", "content": error_msg})
        
        # Refresh title and doc info
        try:
            state = chatbot.get_state(config={"configurable": {"thread_id": thread_key}})
            messages_in_graph = state.values.get("messages", [])
            set_thread_title_from_first_message(thread_key, messages_in_graph)
        except Exception as e:
            print(f"Warning: Could not refresh title: {e}")

# ======================= MAIN ENTRY POINT ===================
def main():
    """Main entry point - show login or chat UI based on authentication"""
    if not st.session_state.get("authenticated", False):
        show_login_screen()
    else:
        show_chat_ui()

if __name__ == "__main__":
    main()