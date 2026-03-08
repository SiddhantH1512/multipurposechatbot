import os

from sqlalchemy import text

from src.database.engine import async_engine
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from src.backend.langgraph_backend import chatbot, ingest_pdf
from src.backend.thread_service import retrieve_all_threads, thread_document_metadata
from src.backend.thread_service import load_conversation
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from fastapi.responses import StreamingResponse
from fastapi import Form, HTTPException


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
    thread_id: str = Form(...)
):
    """
    Async endpoint to upload and index a PDF for a given thread.
    Uses async_engine for any manual DB operations.
    """
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    try:
        # Read file contents (async-friendly)
        contents = await file.read()

        # Call your existing ingestion function
        # (make sure ingest_pdf is compatible — see notes below)
        summary = ingest_pdf(
            file_bytes=contents,
            thread_id=thread_id,
            filename=file.filename
        )

        # Optional: If you still need to do extra manual DB writes here,
        # do it asynchronously like this:
        async with async_engine.connect() as conn:
            await conn.execute(text("""
                INSERT INTO thread_metadata 
                (thread_id, filename, documents, chunks)
                VALUES (:tid, :fn, :docs, :chunks)
                ON CONFLICT (thread_id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    documents = EXCLUDED.documents,
                    chunks = EXCLUDED.chunks,
                    updated_at = CURRENT_TIMESTAMP
            """), {
                "tid": thread_id,
                "fn": file.filename,
                "docs": summary.get("documents", 0),
                "chunks": summary.get("chunks", 0),
            })
            await conn.commit()

        return JSONResponse(content={
            "status": "success",
            "summary": summary,
            "thread_id": thread_id
        })

    except Exception as e:
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




