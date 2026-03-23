from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from src.auth.jwt import get_current_user
from src.backend.thread_service import load_conversation
from src.database.engine import get_async_session_dep
from src.database.table_models import User
 
threads_router = APIRouter(prefix="/threads", tags=["threads"], redirect_slashes=False)
 
@threads_router.get("")
@threads_router.get("/")
async def list_threads(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session_dep)
):
    """List only threads belonging to the current user"""
    result = await session.execute(
        text("""
            SELECT thread_id, filename, documents, chunks, created_at
            FROM thread_metadata
            WHERE user_id = :uid
            ORDER BY created_at DESC
        """),
        {"uid": current_user.id}
    )
    rows = result.fetchall()
    threads = []
    for row in rows:
        thread_data = {
            "thread_id": row[0],
            "metadata": {
                "filename": row[1],
                "documents": row[2],
                "chunks": row[3],
                "created_at": row[4]
            } if row[1] else None
        }
        threads.append(thread_data)
    return {"threads": threads}
 
@threads_router.get("/{thread_id}")
async def get_conversation(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session_dep)
):
    """Get conversation for a specific thread (with ownership check)"""
    # Verify ownership
    result = await session.execute(
        text("SELECT user_id FROM thread_metadata WHERE thread_id = :tid"),
        {"tid": thread_id}
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Thread not found")
    if row[0] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this thread")
    
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