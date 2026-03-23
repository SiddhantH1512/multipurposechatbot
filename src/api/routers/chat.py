from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from sqlalchemy import text
from src.auth.jwt import get_current_user
from src.database.engine import get_async_session_dep, rls_context
from src.backend.langgraph_backend import build_chatbot
from src.database.table_models import User
from sqlalchemy.ext.asyncio import AsyncSession
 
chat_router = APIRouter(prefix="/chat", tags=["chat"])
 
@chat_router.post("")
async def send_message(
    message: str = Form(...),
    thread_id: str = Form(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session_dep)
):
    if not thread_id or not message:
        raise HTTPException(status_code=400, detail="thread_id and message are required")
    
    # Ensure thread metadata exists for this user (for conversation isolation)
    result = await session.execute(
        text("SELECT 1 FROM thread_metadata WHERE thread_id = :tid AND user_id = :uid"),
        {"tid": thread_id, "uid": current_user.id}
    )
    if result.scalar_one_or_none() is None:
        await session.execute(
            text("""
                INSERT INTO thread_metadata (thread_id, user_id, department, is_global)
                VALUES (:tid, :uid, :dept, false)
            """),
            {"tid": thread_id, "uid": current_user.id, "dept": current_user.department}
        )
        await session.commit()
    
    async with rls_context(session, current_user):
        # Build a per-request chatbot whose rag_tool is scoped to this user
        user_chatbot = build_chatbot(current_user)
 
        def generate():
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
 
            try:
                for chunk, _ in user_chatbot.stream(
                    {"messages": [HumanMessage(content=message)]},
                    config=config,
                    stream_mode="messages",
                ):
                    if isinstance(chunk, AIMessage) and chunk.content:
                        yield chunk.content
                    elif isinstance(chunk, ToolMessage):
                        tool_name = getattr(chunk, "name", "unknown")
                        if tool_name == "rag_tool":
                            yield "📄 **Searching organisational documents...**\n\n"
                        elif tool_name == "get_stock_price":
                            yield "📈 **Fetching latest stock price...**\n\n"
                        elif tool_name == "calculator":
                            yield "🧮 **Calculating...**\n\n"
                        else:
                            yield f"🔧 **Calling {tool_name} tool...**\n\n"
            except Exception as e:
                error_msg = "Sorry, something went wrong while processing your request. Please try again."
                print(f"Checkpoint/stream error: {str(e)}")
                yield f"\n\n**Error:** {error_msg}"
 
        return StreamingResponse(generate(), media_type="text/event-stream")