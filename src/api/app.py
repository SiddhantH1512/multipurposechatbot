from fastapi import FastAPI
from src.api.routers.auth import auth_router
from src.api.routers.chat import chat_router
from src.api.routers.threads import threads_router
from src.api.routers.ingest import ingest_router

app = FastAPI(
    title="PolicyIQ API",
    description="Organizational policy RAG backend",
    version="0.1.0"
)

# Include all routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(threads_router)
app.include_router(ingest_router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}





