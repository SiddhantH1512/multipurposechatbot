from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from sqlalchemy import text
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from src.auth.jwt import get_current_user
from src.backend.thread_service import load_conversation, thread_document_metadata
from src.config import Config
from src.database.engine import get_async_session_dep
from src.database.table_models import User
import json
import redis.asyncio as aioredis
 
threads_router = APIRouter(prefix="/threads", tags=["threads"], redirect_slashes=False)

redis_client = aioredis.from_url(Config.REDIS_URL)


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


@threads_router.get("")
async def list_threads(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session_dep)
):
    tenant_id = getattr(current_user, "tenant_id", "default")
    cache_key = f"threads:{tenant_id}:{current_user.id}"

    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    result = await session.execute(
        text("""
            SELECT thread_id, filename, documents, chunks, created_at
            FROM thread_metadata
            WHERE user_id = :uid
             AND tenant_id = :tenant_id
            ORDER BY created_at DESC
        """),
        {"uid": current_user.id, "tenant_id": tenant_id}
    )
    rows = result.fetchall()

    threads = []
    for row in rows:
        thread_id = row[0]
        metadata = thread_document_metadata(thread_id)
        
        threads.append({
            "thread_id": thread_id,
            "filename": row[1],
            "documents": row[2],
            "chunks": row[3],
            "created_at": row[4],           # datetime object
            "metadata": metadata
        })

    final = {"threads": threads}

    # Serialize with custom handler for datetime
    await redis_client.set(
        cache_key, 
        json.dumps(final, default=json_serial), 
        ex=180
    )
    
    return final
 
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