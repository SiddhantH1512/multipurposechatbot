from __future__ import annotations
import datetime
import os
import uuid

from fastapi import Depends

from src.auth.jwt import get_current_user
from src.config import Config
from src.database.engine import get_async_session
from src.database.table_models import User
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

from psycopg_pool import ConnectionPool
import tempfile
from typing import Annotated, Optional, TypedDict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_huggingface import HuggingFaceEmbeddings
from src.models import ChatOpenAIModel
from src.tools.tool_list import tools, build_rag_tool, email_action_extractor, calculator, get_stock_price, search
from langchain_postgres import PGVector, PGEngine  # alias if needed
from sqlalchemy import text
from src.database.engine import *
from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()
# ────────────────────────────────────────────────
# Global / init (do this once, e.g. at module level or in a setup function)
# ────────────────────────────────────────────────
def get_vector_store():
    return PGVector(
        connection=sync_engine,                   # ← can pass Engine or AsyncEngine
        collection_name="chatbot_documents",
        embeddings=embeddings,
        distance_strategy="cosine",
    )

# Optional: ensure extension is enabled (can run once)
async def ensure_extension():
    async with async_engine.connect() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.commit()
    print("pgvector extension ensured.")

# Collection / table name — choose something stable
COLLECTION_NAME = "chatbot_documents"

# Create / get vector store
# ========================== CONFIG ==========================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True},
)

llm = ChatOpenAIModel()
llm_with_tools = llm.bind_tools(tools)

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "ai_ml_documents"

# Initialize Chroma (persistent)
vector_store = get_vector_store()

# ========================== INGESTION (NEW) ==========================
async def ingest_pdf(
    file_bytes: bytes, 
    thread_id: str, 
    filename: Optional[str] = None, 
    session: AsyncSession = Depends(get_async_session), 
    current_user: User = Depends(get_current_user),
    visibility: str = "global",
    department: Optional[str] = None
) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        temp_path = f.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(docs)

        # Determine which department to use for document access
        if visibility == "global":
            doc_department = "General"  # Not used for filtering
        else:  # visibility == "dept"
            # Use provided department if specified, otherwise fall back to current_user's department
            doc_department = department if department else current_user.department

        # Stable UUID for this document — used to update visibility later
        document_id = str(uuid.uuid4())

        for chunk in chunks:
            metadata = {
                "document_id": document_id,
                "visibility": visibility,
                "department": doc_department,
                "uploaded_by": current_user.id,
                "uploaded_by_email": current_user.email,
                "filename": filename,
                "page": chunk.metadata.get("page", 0) + 1,
                "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            chunk.metadata.update(metadata)

        vector_store.add_documents(chunks)

        # Update thread_metadata - this is for conversation tracking, not document access
        await session.execute(text("""
            INSERT INTO thread_metadata 
            (thread_id, filename, documents, chunks, user_id, department, is_global, document_id)
            VALUES (:thread_id, :filename, :documents, :chunks, :user_id, :dept, :is_global, :document_id)
            ON CONFLICT (thread_id) DO UPDATE SET
                filename = EXCLUDED.filename,
                documents = EXCLUDED.documents,
                chunks = EXCLUDED.chunks,
                user_id = EXCLUDED.user_id,
                department = EXCLUDED.department,
                is_global = EXCLUDED.is_global,
                document_id = EXCLUDED.document_id,
                updated_at = CURRENT_TIMESTAMP
        """), {
            "thread_id": str(thread_id),
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
            "user_id": current_user.id,
            "dept": current_user.department,
            "is_global": visibility == "global",
            "document_id": document_id
        })

        return {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
            "department": doc_department,
            "visibility": visibility,
            "uploaded_by": current_user.email,
            "document_id": document_id,
            "target_department": department if visibility == "dept" else None 
        }
    finally:
        os.unlink(temp_path)

# ========================== STATE ==========================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ========================== CHECKPOINTER ====================
pool = ConnectionPool(Config.POSTGRES_CONNINFO)
checkpointer = PostgresSaver(pool)

# ========================== GRAPH FACTORY ===================
def _build_graph(tool_list):
    """Compile a LangGraph chatbot with the given tool list."""
    _llm_with_tools = llm.bind_tools(tool_list)

    def _chat_node(state: ChatState, config=None):
        system_prompt = """You are a helpful organisational assistant answering questions about HR policies and company documents.

Document-question rules:
1. For any question that may relate to company policies or uploaded documents → ALWAYS call rag_tool first.
2. When you receive the ToolMessage from rag_tool → write the final answer using those excerpts.
   - ALWAYS include citations: [Source 1: filename – page X]
   - ONLY say "No relevant information found" if the context has literally nothing related.
3. Be concise, technical, and factual."""

        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        response = _llm_with_tools.invoke(messages, config=config)
        return {"messages": [response]}

    g = StateGraph(ChatState)
    g.add_node("chat_node", _chat_node)
    g.add_node("tools", ToolNode(tool_list))
    g.add_edge(START, "chat_node")
    g.add_conditional_edges("chat_node", tools_condition)
    g.add_edge("tools", "chat_node")
    return g.compile(checkpointer=checkpointer)


# Default chatbot — HR-level access, used by thread_service / CLI
chatbot = _build_graph(tools)


def build_chatbot(user: "User"):
    """
    Returns a compiled chatbot graph whose rag_tool is scoped to the given user.
    Called per-request in the chat endpoint.
    """
    dept = user.department or "General"
    role = user.role.value if hasattr(user.role, "value") else str(user.role)
    scoped_rag = build_rag_tool(user_department=dept, user_role=role)
    scoped_tools = [search, get_stock_price, calculator, scoped_rag, email_action_extractor]
    return _build_graph(scoped_tools)