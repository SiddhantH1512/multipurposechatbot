# from fastapi import APIRouter, Depends, Form, HTTPException
# from fastapi.responses import StreamingResponse
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# from sqlalchemy import text
# from src.auth.jwt import get_current_user
# from src.backend.audit import log_audit
# from src.backend.rate_limiter import check_rate_limit
# from src.backend.security import sanitize_input
# from src.database.engine import get_async_session_dep, rls_context
# from src.backend.langgraph_backend import build_chatbot
# from src.database.table_models import User
# from sqlalchemy.ext.asyncio import AsyncSession

# chat_router = APIRouter(prefix="/chat", tags=["chat"])


# # ── Helper: emit readable Self-RAG status lines ───────────────────────────
# def _self_rag_status(faithfulness_grade: str, retry_count: int, rewrite_count: int) -> str:
#     lines = []
    
#     if rewrite_count > 0:
#         lines.append(f"✏️ **Query improved** ({rewrite_count} rewrite(s))")
#     if retry_count > 0:
#         lines.append(f"🔄 **Answer refined** (after {retry_count} attempt(s))")

#     if faithfulness_grade == "fully_supported":
#         lines.append("✅ **Document support:** Fully Supported")
#     elif faithfulness_grade == "partially_supported":
#         lines.append("⚠️ **Document support:** Partially Supported")
#     elif faithfulness_grade == "not_supported":
#         lines.append("❌ **Document support:** Not Supported")

#     return "\n".join(lines) + "\n\n" if lines else ""


# @chat_router.post("")
# async def send_message(
#     message: str = Form(...),
#     thread_id: str = Form(...),
#     current_user: User = Depends(check_rate_limit("/chat")),
#     session: AsyncSession = Depends(get_async_session_dep)
# ):
#     if not thread_id or not message:
#         raise HTTPException(status_code=400, detail="thread_id and message are required")

#     # Sanitize input
#     try:
#         sanitized_message, pii_types = sanitize_input(message)
#         if pii_types:
#             print(f"[SECURITY] PII redacted for user {current_user.email}: {pii_types}")
#     except ValueError as e:
#         # Graceful handling for prompt injection / dangerous input
#         def blocked_generate():
#             yield "🚫 **Security Notice**: Your message was blocked for safety reasons.\n\n"
#             yield "It appears to contain potentially harmful instructions (such as prompt injection attempts).\n\n"
#             yield "Please rephrase your question normally."
        
#         return StreamingResponse(blocked_generate(), media_type="text/event-stream")

#     await log_audit(
#         user_id=current_user.id,
#         action="chat_message",
#         resource=thread_id,
#         details=f"length={len(sanitized_message)}",
#         session=session
#     )

#     # Ensure thread metadata exists
#     result = await session.execute(
#         text("SELECT 1 FROM thread_metadata WHERE thread_id = :tid AND user_id = :uid"),
#         {"tid": thread_id, "uid": current_user.id}
#     )
#     if result.scalar_one_or_none() is None:
#         await session.execute(
#             text("""
#                 INSERT INTO thread_metadata (thread_id, user_id, department, is_global)
#                 VALUES (:tid, :uid, :dept, false)
#             """),
#             {"tid": thread_id, "uid": current_user.id, "dept": current_user.department}
#         )
#         await session.commit()

#     async with rls_context(session, current_user):
#         user_chatbot = build_chatbot(current_user)

#         def generate():
#             config = {
#                 "configurable": {"thread_id": thread_id},
#                 "recursion_limit": 60,
#             }

#             try:
#                 final_state = None
#                 rag_status_shown = False

#                 for event in user_chatbot.stream(
#                     {
#                         "messages": [HumanMessage(content=sanitized_message)],
#                         "original_query": sanitized_message,
#                         "current_query": sanitized_message,
#                         "retrieved_context": "",
#                         "relevant_context": "",
#                         "generated_answer": "",
#                         "faithfulness_grade": "",
#                         "unsupported_claims": [],
#                         "retry_count": 0,
#                         "rewrite_count": 0,
#                         "need_retrieval": True,
#                         "skip_retrieval": False,
#                         "answer_useful": False,
#                     },
#                     config=config,
#                     stream_mode="values",
#                 ):
#                     final_state = event

#                     # === Real-time Status Messages (show only once) ===
#                     msgs = event.get("messages", [])
#                     if msgs:
#                         last_msg = msgs[-1]

#                         # Show "Searching documents" only once
#                         if (isinstance(last_msg, ToolMessage) and 
#                             getattr(last_msg, "name", "") == "rag_tool" and 
#                             not rag_status_shown):
                            
#                             if event.get("skip_retrieval", False):
#                                 yield "📭 **No relevant documents found — answering from general knowledge...**\n\n"
#                             else:
#                                 yield "📄 **Searching organisational documents...**\n\n"
#                             rag_status_shown = True

#                         # Show refinement status
#                         retry = event.get("retry_count", 0)
#                         if retry > 0 and isinstance(last_msg, AIMessage) and last_msg.content:
#                             yield f"⚠️ **Refining answer for accuracy (attempt {retry})...**\n\n"

#                 # === Final Output ===
#                 if final_state:
#                     # Self-RAG status (Faithfulness, Rewrites, etc.)
#                     status = _self_rag_status(
#                         faithfulness_grade=final_state.get("faithfulness_grade", ""),
#                         retry_count=final_state.get("retry_count", 0),
#                         rewrite_count=final_state.get("rewrite_count", 0),
#                     )
#                     if status:
#                         yield status

#                     # Extract and stream the final AI answer
#                     answer_found = False
#                     for msg in reversed(final_state.get("messages", [])):
#                         if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
#                             yield msg.content
#                             answer_found = True
#                             break

#                     if not answer_found:
#                         yield "Sorry, I couldn't generate a response. Please try again."

#                 else:
#                     yield "Sorry, I couldn't generate a response."

#             except Exception as e:
#                 print(f"[Self-RAG] Stream error: {type(e).__name__}: {str(e)}")
#                 yield "\n\n**Error:** Sorry, something went wrong while processing your request. Please try again."

#         return StreamingResponse(generate(), media_type="text/event-stream")
from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from sqlalchemy import text
from src.auth.jwt import get_current_user
from src.backend.audit import log_audit
from src.backend.rate_limiter import check_rate_limit
from src.backend.security import sanitize_input
from src.database.engine import get_async_session_dep, rls_context
from src.backend.langgraph_backend import build_chatbot
from src.database.table_models import User
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.self_rag import SelfRAGState  # Make sure this import works

chat_router = APIRouter(prefix="/chat", tags=["chat"])


# ── Helper: emit readable Self-RAG status lines ───────────────────────────
def _self_rag_status(faithfulness_grade: str, retry_count: int, rewrite_count: int) -> str:
    lines = []
    
    if rewrite_count > 0:
        lines.append(f"✏️ **Query improved** ({rewrite_count} rewrite(s))")
    if retry_count > 0:
        lines.append(f"🔄 **Answer refined** (after {retry_count} attempt(s))")

    if faithfulness_grade == "fully_supported":
        lines.append("✅ **Document support:** Fully Supported")
    elif faithfulness_grade == "partially_supported":
        lines.append("⚠️ **Document support:** Partially Supported")
    elif faithfulness_grade == "not_supported":
        lines.append("❌ **Document support:** Not Supported")

    return "\n".join(lines) + "\n\n" if lines else ""


# ── NEW: Format follow-up suggestions nicely ─────────────────────────────
def _format_followups(follow_ups: list[str]) -> str:
    if not follow_ups:
        return ""
    
    lines = ["**💡 Suggested follow-up questions:**"]
    for i, question in enumerate(follow_ups[:3], 1):
        lines.append(f"{i}. {question}")
    return "\n".join(lines) + "\n\n"


@chat_router.post("")
async def send_message(
    message: str = Form(...),
    thread_id: str = Form(...),
    current_user: User = Depends(check_rate_limit("/chat")),
    session: AsyncSession = Depends(get_async_session_dep)
):
    if not thread_id or not message:
        raise HTTPException(status_code=400, detail="thread_id and message are required")

    # Sanitize input
    try:
        sanitized_message, pii_types = sanitize_input(message)
        if pii_types:
            print(f"[SECURITY] PII redacted for user {current_user.email}: {pii_types}")
    except ValueError as e:
        def blocked_generate():
            yield "🚫 **Security Notice**: Your message was blocked for safety reasons.\n\n"
            yield "It appears to contain potentially harmful instructions.\n\n"
            yield "Please rephrase your question normally."
        return StreamingResponse(blocked_generate(), media_type="text/event-stream")

    await log_audit(
        user_id=current_user.id,
        action="chat_message",
        resource=thread_id,
        details=f"length={len(sanitized_message)}",
        session=session
    )

    # Ensure thread metadata exists
    result = await session.execute(
        text("SELECT 1 FROM thread_metadata WHERE thread_id = :tid AND user_id = :uid"),
        {"tid": thread_id, "uid": current_user.id}
    )
    if result.scalar_one_or_none() is None:
        tenant_id = getattr(current_user, "tenant_id", "default")
        await session.execute(
            text("""
                INSERT INTO thread_metadata (thread_id, user_id, department, is_global, tenant_id)
                VALUES (:tid, :uid, :dept, false, :tenant_id)
            """),
            {"tid": thread_id, "uid": current_user.id, "dept": current_user.department, "tenant_id": tenant_id}
        )
        await session.commit()

    async with rls_context(session, current_user):
        user_chatbot = build_chatbot(current_user)

        def generate():
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 60,
            }

            try:
                final_state = None
                status_shown = False

                for event in user_chatbot.stream(
                    {
                        "messages": [HumanMessage(content=sanitized_message)],
                        "original_query": sanitized_message,
                        "current_query": sanitized_message,
                        "retrieved_context": "",
                        "relevant_context": "",
                        "generated_answer": "",
                        "faithfulness_grade": "",
                        "unsupported_claims": [],
                        "retry_count": 0,
                        "rewrite_count": 0,
                        "need_retrieval": True,
                        "skip_retrieval": False,
                        "answer_useful": False,
                    },
                    config=config,
                    stream_mode="values",
                ):
                    final_state = event

                    # === ONLY SHOW MINIMAL USER-FRIENDLY STATUS ===
                    if not status_shown:
                        # Show "Searching documents..." only once
                        msgs = event.get("messages", [])
                        if msgs:
                            last_msg = msgs[-1]
                            if isinstance(last_msg, ToolMessage) and getattr(last_msg, "name", "") == "rag_tool":
                                if event.get("skip_retrieval", False):
                                    yield "📭 No relevant documents found — answering from general knowledge...\n\n"
                                else:
                                    yield "🔍 Searching organisational documents...\n\n"
                                status_shown = True

                # === FINAL OUTPUT ONLY (Clean for user) ===
                if final_state:
                    # Show final answer only
                    answer_found = False
                    for msg in reversed(final_state.get("messages", [])):
                        if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                            yield msg.content.strip() + "\n\n"
                            answer_found = True
                            break

                    # Show follow-up suggestions (clean format)
                    suggestions = final_state.get("follow_up_suggestions", [])
                    if suggestions:
                        yield "**💡 Suggested follow-up questions:**\n"
                        for i, q in enumerate(suggestions[:3], 1):
                            yield f"{i}. {q}\n"

                    if not answer_found:
                        yield "Sorry, I couldn't generate a response. Please try again."

                else:
                    yield "Sorry, I couldn't generate a response."

            except Exception as e:
                print(f"[Self-RAG] Stream error: {type(e).__name__}: {str(e)}")
                yield "\n\n**Error:** Sorry, something went wrong while processing your request."

        return StreamingResponse(generate(), media_type="text/event-stream")