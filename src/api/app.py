import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
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


from fastapi.responses import StreamingResponse
from fastapi import Form, HTTPException

@app.post("/chat")
async def send_message(
    message: str = Form(...),
    thread_id: str = Form(...)
):
    if not thread_id or not message:
        raise HTTPException(status_code=400, detail="thread_id and message are required")

    def generate():
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,
        }

        try:
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                stream_mode="messages",
            ):
                msg = chunk[0] if isinstance(chunk, tuple) else chunk

                if isinstance(msg, AIMessage):
                    if msg.content.strip():
                        yield msg.content
                    elif msg.tool_calls:
                        yield "→ Retrieving from document...\n"
                    # Do NOT yield anything for empty AIMessage → removes spam

                elif isinstance(msg, ToolMessage):
                    if "Error" not in msg.content:           # optional: hide error messages
                        yield "[Document search complete]\n"
                    else:
                        yield f"[Search issue] {msg.content[:100]}...\n"

        except Exception as e:
            yield f"\n\n**Error:** {str(e)}"

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




