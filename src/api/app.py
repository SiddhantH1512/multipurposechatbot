import os

from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select, text

from src.auth.jwt import create_access_token, get_current_user, verify_password
from src.database.engine import async_engine, get_async_session, get_async_session_dep
from src.database.table_models import User
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import APIRouter, Depends, FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from src.backend.langgraph_backend import chatbot, ingest_pdf
from src.backend.thread_service import retrieve_all_threads, thread_document_metadata
from src.backend.thread_service import load_conversation
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from fastapi.responses import StreamingResponse
from fastapi import Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession


app = FastAPI(
    title="LangGraph PDF Chatbot API",
    description="Backend API for multi-threaded PDF-aware chatbot",
    version="0.1.0"
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


router = APIRouter()  # or include in your existing router

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session_dep),
):
    """
    Async endpoint to upload and index a PDF.
    - Only HR can ingest
    - Uses the same transaction/session for all DB writes
    """
    if current_user.role != "HR":
        raise HTTPException(status_code=403, detail="Only HR can ingest documents")

    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    try:
        # Read file contents asynchronously
        contents = await file.read()

        # Call async ingestion function — passes the session
        summary = await ingest_pdf(
            file_bytes=contents,
            thread_id=thread_id,
            filename=file.filename,
            session=session,                      # ← pass the open session
        )

        # Optional: If you want to do extra writes here (not needed anymore),
        # you can do them in the same session — they will be committed together
        # await session.execute(...)

        # No manual commit/close needed — get_async_session handles it

        return JSONResponse(content={
            "status": "success",
            "summary": summary,
            "thread_id": thread_id,
            "uploaded_by": current_user.email,   # optional feedback
        })

    except Exception as e:
        # The session will auto-rollback on exception
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/chat")
async def send_message(
    message: str = Form(...),
    thread_id: str = Form(...)
):
    if not thread_id or not message:
        raise HTTPException(status_code=400, detail="thread_id and message are required")

    def generate():
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

        try:
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content
                elif isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "unknown")
                    if tool_name == "rag_tool":
                        yield "📄 **Searching document...**\n\n"
                    elif tool_name == "get_stock_price":
                        yield "📈 **Fetching latest stock price...**\n\n"
                    elif tool_name == "calculator":
                        yield "🧮 **Calculating...**\n\n"
                    else:
                        yield f"🔧 **Calling {tool_name} tool...**\n\n"
        except Exception as e:
            # ← Catch ALL exceptions and send a clean message to the UI
            error_msg = "Sorry, something went wrong while processing your request. Please try again."
            print(f"Checkpoint/stream error: {str(e)}")  # log the real error server-side
            yield f"\n\n**Error:** {error_msg}"
            # Optionally yield the technical detail only in logs, not UI
            # yield f"\n\n(Technical: {str(e)})"  # remove or comment this line

    return StreamingResponse(generate(), media_type="text/event-stream")
    


@app.get("/threads")
async def list_threads():
    threads = retrieve_all_threads() or []
    result = []
    for tid in threads:
        meta = thread_document_metadata(tid)
        result.append({
            "thread_id": tid,
            "has_document": bool(meta),
            "metadata": meta or None
        })
    return {"threads": result}


@app.get("/thread/{thread_id}")
async def get_conversation(thread_id: str):
    messages = load_conversation(thread_id)
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            history.append({"role": "tool", "content": msg.content, "tool": msg.name})
    return {"thread_id": thread_id, "messages": history}


from fastapi import Form

@app.post("/auth/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_async_session_dep)
):
    result = await session.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/auth/me", response_model=dict)
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role.value if hasattr(current_user.role, "value") else current_user.role,
        "department": current_user.department,
        "designation": current_user.designation,
        "is_active": current_user.is_active
    }


