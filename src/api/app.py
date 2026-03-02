# src/api/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import uuid

from src.backend.langgraph_backend import chatbot, ingest_pdf
from src.backend.thread_service import retrieve_all_threads, thread_document_metadata
from src.backend.thread_service import load_conversation
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

app = FastAPI(
    title="LangGraph PDF Chatbot API",
    description="Backend API for multi-threaded PDF-aware chatbot",
    version="0.1.0"
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
):
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    try:
        contents = await file.read()
        summary = ingest_pdf(
            file_bytes=contents,
            thread_id=thread_id,
            filename=file.filename
        )
        return JSONResponse(content={
            "status": "success",
            "summary": summary,
            "thread_id": thread_id
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def send_message(
    message: str = Form(...),
    thread_id: str = Form(...)
):
    if not thread_id or not message:
        raise HTTPException(status_code=400, detail="thread_id and message are required")

    try:
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {"thread_id": thread_id},
        }

        # Stream simulation – for simplicity we collect full response
        messages = []
        for chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, (AIMessage, ToolMessage)):
                messages.append(chunk)

        # Get final AI message (last relevant one)
        ai_content = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                ai_content = msg.content
                break

        return {
            "thread_id": thread_id,
            "user_message": message,
            "assistant_response": ai_content,
            "full_messages": [msg.content for msg in messages if msg.content]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


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