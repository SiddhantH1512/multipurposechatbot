import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import httpx
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.backend.thread_service import delete_thread, load_conversation, retrieve_all_threads, thread_document_metadata
from src.backend.utils import add_thread, generate_thread_id, get_thread_display_name, reset_chat, set_thread_title_from_first_message

if os.getenv("DOCKER_ENV") == "true":
    API_BASE_URL = "http://fastapi:8000"
else:
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
        "login_counter": 0,
        "last_follow_up_suggestions": []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
init_session_state()


# ======================= Title refresh via API ===================
def refresh_thread_title_via_api(thread_id: str):
    """
    Fetch conversation history for a thread via the /threads API and
    derive a display title from the first human message.
    Uses the stored JWT token — safe to call after every chat turn.
    """
    token = st.session_state.get("token")
    if not token:
        return

    tid = str(thread_id)
    # Already have a title for this thread — nothing to do
    if tid in st.session_state.get("thread_titles", {}):
        return

    try:
        headers = {"Authorization": f"Bearer {token}"}
        resp = httpx.get(
            f"{API_BASE_URL}/threads/{tid}",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code == 200:
            messages = resp.json().get("messages", [])
            # Convert API response format to LangChain HumanMessage list
            lc_messages = [
                HumanMessage(content=m["content"])
                for m in messages
                if m.get("role") == "user" and m.get("content", "").strip()
            ]
            set_thread_title_from_first_message(tid, lc_messages)
    except Exception as e:
        print(f"[Title refresh] Could not fetch thread {tid}: {e}")


# ======================= LOGIN + REGISTER SCREEN ===================
def show_login_screen():
    """Clean login screen only (no register tab anymore)"""
    st.set_page_config(page_title="PolicyIQ - Login", layout="centered")
    
    st.title("🔐 PolicyIQ")
    st.markdown("### Welcome to PolicyIQ Organizational Assistant")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if not email or not password:
                st.error("Please enter both email and password")
                return
            
            try:
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
                    
                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                    profile_resp = httpx.get(f"{API_BASE_URL}/auth/me", headers=headers, timeout=5.0)
                    
                    if profile_resp.status_code == 200:
                        user = profile_resp.json()
                        st.session_state.user_role = user.get("role")
                        st.session_state.user_department = user.get("department")
                        st.session_state.user_id = user.get("id")
                        st.session_state.is_hr = user.get("role") == "HR"
                        st.session_state.user_tenant_id = user.get("tenant_id", "default")
                    
                    st.success(f"✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

    with st.expander("📋 Demo Accounts"):
        st.markdown("""
        **Default Password:** `Test@123456`
        
        **HR Users (can register & manage users):**
        - hr@company.com
        - hr.specialist@company2.com
        """)


def show_hr_choice_menu():
    """HR Choice Menu after login"""
    st.set_page_config(page_title="PolicyIQ - HR Dashboard", layout="centered")
    
    st.title(f"👋 Welcome back, {st.session_state.user_email.split('@')[0].title()}!")
    st.markdown("### What would you like to do?")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        if st.button("📝 Register New User", use_container_width=True):
            st.session_state.hr_action = "register"
            st.rerun()

    with col2:
        if st.button("🗑️ Manage Users", use_container_width=True):
            st.session_state.hr_action = "manage_users"
            st.rerun()

    with col3:
        if st.button("💬 Go to Chatbot", use_container_width=True, type="primary"):
            st.session_state.hr_action = None
            st.session_state.show_hr_menu = False   # ← THIS WAS MISSING
            st.rerun()

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ======================= MAIN CHAT UI ===================
def show_chat_ui():
    """Display the main chat interface for authenticated users"""
    # Dynamic greeting name
    email = st.session_state.get("user_email", "")
    name_part = "there"
    if email and '@' in email:
        local_part = email.split('@')[0]
        if '.' in local_part:
            name_part = local_part.split('.')[0].title()
        else:
            name_part = local_part.title()

    st.set_page_config(page_title="PolicyIQ Chatbot", layout="wide")
    
    print(f"Current user: {st.session_state.user_email}, Role: {st.session_state.user_role}, is_hr: {st.session_state.is_hr}")
    
    thread_key = str(st.session_state["thread_id"])
    thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
    
    # ============================ Sidebar ============================
    st.sidebar.title("💬 PolicyIQ Chatbot")
    st.sidebar.markdown(f"**👤 User:** `{st.session_state.user_email}`")
    st.sidebar.markdown(f"**🎭 Role:** `{st.session_state.user_role}`")
    st.sidebar.markdown(f"**🏢 Department:** `{st.session_state.user_department}`")
    st.sidebar.markdown(f"**🏷️ Tenant ID:** `{st.session_state.get('user_tenant_id', 'default')}`")
    st.sidebar.markdown(f"**🔐 HR Status:** `{'✅ HR' if st.session_state.is_hr else '❌ Not HR'}`")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**🔑 Thread ID:** `{thread_key[:8]}...`")
    st.sidebar.markdown(f"**👤 Name:** {name_part}")

    # HR Quick Access - Back to Menu
    if st.session_state.get("is_hr"):
        if st.sidebar.button("← Back to HR Menu", use_container_width=True):
            st.session_state.hr_action = None
            st.session_state.show_hr_menu = True
            st.rerun()
    
    st.sidebar.markdown("---")
    
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
    
    is_hr = st.session_state.get("is_hr", False)
    
    if is_hr:
        st.sidebar.markdown("### 📤 Document Upload")
        st.sidebar.success("✅ HR privileges detected - Upload enabled")
        
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

        # === ONE-SHOT UPLOAD (prevents duplicates) ===
        upload_counter = st.session_state.setdefault("upload_counter", 0)
        upload_key = f"pdf_uploader_{st.session_state.login_counter}_{upload_counter}"

        uploaded_pdf = st.sidebar.file_uploader(
            "Upload HR policy document",
            type=["pdf"],
            help="Only HR users can upload organization-wide documents.",
            key=upload_key
        )

        if uploaded_pdf and uploaded_pdf.name not in thread_docs:
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

                        # === CRITICAL: Increment counter to reset uploader ===
                        st.session_state.upload_counter += 1
                        st.rerun()   # Refresh UI with new uploader key

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
        st.sidebar.caption(f"Debug: User role = {st.session_state.user_role}, is_hr = {is_hr}")

    st.sidebar.markdown("---")
    
    # ── Document Manager (HR) / Accessible Documents (others) ──────────────
    if is_hr:
        st.sidebar.markdown("### 📋 Document Manager")

        if st.sidebar.button("🔄 Refresh Documents", key="refresh_docs_btn", use_container_width=True):
            st.session_state["docs_list"] = None

        if st.session_state.get("docs_list") is None:
            try:
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                r = httpx.get(f"{API_BASE_URL}/documents", headers=headers, timeout=10.0)
                if r.status_code == 200:
                    st.session_state["docs_list"] = r.json().get("documents", [])
                else:
                    st.session_state["docs_list"] = []
            except Exception as e:
                st.session_state["docs_list"] = []
                st.sidebar.error(f"Failed to load documents: {e}")

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
                                st.session_state["docs_list"] = None
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
            
            if tid not in st.session_state.get("thread_titles", {}):
                refresh_thread_title_via_api(tid)
            
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
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["docs_list"] = None
        st.rerun()
    
    # Handle thread selection
    if "selected_thread_temp" in st.session_state:
        selected_thread = st.session_state["selected_thread_temp"]
        del st.session_state["selected_thread_temp"]
        
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
    st.title(f"👋 Welcome back, {name_part}! 💬 PolicyIQ")

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

                # Parse follow-up suggestions — always update (including clearing)
                lines = full_response.split("\n")
                suggestions = []
                capture = False
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("**💡 Suggested follow-up questions:**"):
                        capture = True
                        continue
                    if capture and stripped and stripped[0].isdigit() and ". " in stripped:
                        q = stripped.split(". ", 1)[1].strip()
                        if q:
                            suggestions.append(q)

                # Always overwrite — empty list clears stale suggestions from prior turns
                st.session_state["last_follow_up_suggestions"] = suggestions

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state["message_history"].append({"role": "assistant", "content": error_msg})

        # Refresh thread title
        refresh_thread_title_via_api(thread_key)

    # Clickable Follow-up Suggestions
    if st.session_state.get("last_follow_up_suggestions"):
        st.markdown("**💡 Suggested follow-up questions:**")
        cols = st.columns(min(3, len(st.session_state["last_follow_up_suggestions"])))
        
        for idx, question in enumerate(st.session_state["last_follow_up_suggestions"]):
            if idx < len(cols):
                if cols[idx].button(question, key=f"followup_btn_{thread_key}_{idx}", use_container_width=True):
                    st.session_state["message_history"].append({"role": "user", "content": question})
                    st.session_state["last_follow_up_suggestions"] = []
                    st.rerun()

        if st.button("Clear suggestions", key=f"clear_sugg_{thread_key}"):
            st.session_state["last_follow_up_suggestions"] = []
            st.rerun()

# ======================= MAIN ENTRY POINT ===================
def show_register_form():
    """Register new user form for HR with validation"""
    st.title("📝 Register New User")
    st.caption(f"Registering for tenant: **{st.session_state.get('user_tenant_id', 'default')}**")

    with st.form("register_new_user"):
        email = st.text_input("Email Address", placeholder="john.doe@company2.com")
        
        col1, col2 = st.columns(2)
        with col1:
            role = st.selectbox("Role", ["EMPLOYEE", "INTERN", "EXECUTIVE", "HR"])
        with col2:
            department = st.selectbox(
                "Department", 
                ["Engineering", "Sales", "Marketing", "Finance", "HR", "Leadership", "General"]
            )
        
        designation = st.text_input("Designation (Optional)", placeholder="Senior Software Engineer")
        
        password = st.text_input("Password", type="password", value="Test@123456")
        confirm_password = st.text_input("Confirm Password", type="password")

        submitted = st.form_submit_button("✅ Register User", use_container_width=True, type="primary")

        if submitted:
            if not email or not password:
                st.error("Email and password are required")
                return
            
            if password != confirm_password:
                st.error("❌ Passwords do not match")
                return
            
            if len(password.encode('utf-8')) > 72:
                st.error("❌ Password is too long. Maximum 72 bytes (characters) allowed.")
                return
            
            if len(password) < 8:
                st.error("❌ Password must be at least 8 characters long")
                return

            try:
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                payload = {
                    "email": email,
                    "password": password,
                    "role": role,
                    "department": department,
                    "designation": designation if designation else None,
                    "tenant_id": st.session_state.get("user_tenant_id", "default")
                }

                response = httpx.post(
                    f"{API_BASE_URL}/auth/register", 
                    json=payload, 
                    headers=headers, 
                    timeout=10
                )

                if response.status_code == 201:
                    st.success(f"🎉 User **{email}** registered successfully!")
                    st.balloons()
                else:
                    error_msg = response.json().get("detail", "Registration failed")
                    st.error(f"❌ {error_msg}")
            except Exception as e:
                st.error(f"Registration error: {str(e)}")

    # Back button
    if st.button("← Back to HR Menu", use_container_width=True):
        st.session_state.hr_action = None
        st.rerun()


def show_manage_users():
    """Manage / Deactivate users - HR only (Tenant-isolated)"""
    st.set_page_config(page_title="PolicyIQ - Manage Users", layout="wide")
    
    st.title("🗑️ Manage Users")
    st.caption(f"Showing users in your tenant: **{st.session_state.get('user_tenant_id', 'default')}**")

    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        
        # Fetch users from backend (already filtered by tenant in the API)
        resp = httpx.get(f"{API_BASE_URL}/auth/users", headers=headers, timeout=10)
        
        if resp.status_code == 200:
            users = resp.json().get("users", [])
            
            if not users:
                st.info("No users found in your tenant.")
            else:
                st.write(f"**Total users in your tenant:** {len(users)}")
                
                for user in users:
                    with st.expander(f"👤 {user['email']} — {user.get('role', '')} | {user.get('department', '')}"):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**Tenant:** `{user.get('tenant_id', 'default')}`")
                            st.markdown(f"**Designation:** {user.get('designation', '—')}")
                            status = "✅ Active" if user.get('is_active') else "❌ Inactive"
                            st.markdown(f"**Status:** {status}")
                        
                        with col2:
                            if user.get('is_active'):
                                if st.button("Deactivate User", key=f"deact_{user['id']}", type="secondary"):
                                    try:
                                        del_resp = httpx.patch(
                                            f"{API_BASE_URL}/auth/users/{user['id']}/deactivate",
                                            headers=headers,
                                            timeout=8
                                        )
                                        if del_resp.status_code == 200:
                                            st.success(f"✅ {user['email']} has been deactivated")
                                            st.rerun()
                                        else:
                                            st.error(del_resp.json().get("detail", "Failed to deactivate"))
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
        
        else:
            st.error("Failed to load users. Please try again.")
            
    except Exception as e:
        st.error(f"Error loading users: {str(e)}")

    # Back button
    st.markdown("---")
    if st.button("← Back to HR Menu", use_container_width=True):
        st.session_state.hr_action = None
        st.rerun()


def main():
    """Main entry point"""
    if not st.session_state.get("authenticated", False):
        show_login_screen()
    
    # HR users who have not yet chosen an action → show choice menu
    elif st.session_state.get("is_hr") and st.session_state.get("show_hr_menu", True):
        if st.session_state.get("hr_action") == "register":
            show_register_form()
        elif st.session_state.get("hr_action") == "manage_users":
            show_manage_users()
        else:
            show_hr_choice_menu()
    
    else:
        # Normal users OR HR who clicked "Go to Chatbot"
        show_chat_ui()

if __name__ == "__main__":
    main()