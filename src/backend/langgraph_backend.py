from __future__ import annotations
import os

from src.config import Config
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

from psycopg_pool import ConnectionPool
import tempfile
from typing import Annotated, TypedDict
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
from src.tools.tool_list import tools
from langchain_postgres import PGVector, PGEngine  # alias if needed
from sqlalchemy import text
from src.database.engine import *

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
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str = None) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        temp_path = f.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Semantic chunking — best for AI/ML resumes & papers in 2026
        # splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=300,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata.update({
                "thread_id": str(thread_id),
                "filename": filename or "unknown.pdf",
                "page": chunk.metadata.get("page", 0) + 1
            })

        vector_store.add_documents(chunks)

        with sync_engine.connect() as conn:           # ← this is the correct call
            conn.execute(text("""
                INSERT INTO thread_metadata 
                (thread_id, filename, documents, chunks)
                VALUES (:thread_id, :filename, :documents, :chunks)
                ON CONFLICT (thread_id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    documents = EXCLUDED.documents,
                    chunks = EXCLUDED.chunks,
                    updated_at = CURRENT_TIMESTAMP
            """), {
                "thread_id": str(thread_id),
                "filename": filename,
                "documents": len(docs),
                "chunks": len(chunks),
            })
            conn.commit()

        return {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        os.unlink(temp_path)

# ========================== STATE & NODES ==========================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState, config=None):
    thread_id = config.get("configurable", {}).get("thread_id") if config else None

    system_prompt = f"""You are an expert AI/ML assistant answering questions about uploaded documents (research papers, notes, textbooks, etc.).

Document-question rules:
1. For any new question that seems related to uploaded document(s) → ALWAYS call rag_tool first (with thread_id = {thread_id}).
2. When you receive the ToolMessage from rag_tool → this is your signal to NOW write the final answer.
   - Use the excerpts provided — even if the match is indirect or requires inference.
   - If the context mentions layers, N, stacks, encoder/decoder architecture, etc. → use that information.
   - ALWAYS include citations: [Source 1: filename – page X]
   - ONLY say "No relevant information found…" if the context has literally nothing related to the question.
3. Be concise, technical, and factual. Use LaTeX for math.

Never stay silent after receiving context. Always produce an answer."""

    messages = [SystemMessage(content=system_prompt), *state["messages"]]
    print("[LLM DEBUG] Full messages sent to LLM:")
    for msg in messages:
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            print(f"  {msg.type}: {msg.content[:500]}...")
    
    print("[CONTEXT SENT TO MODEL AFTER TOOL]")
    print(messages[-1].content[:1500])
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)

pool = ConnectionPool(Config.POSTGRES_CONNINFO)  # add min_size=4, max_size=20 etc. if needed

checkpointer = PostgresSaver(pool)
from psycopg import Connection

with pool.connection() as conn:
    conn.autocommit = True
    temp_checkpointer = PostgresSaver(conn)
    temp_checkpointer.setup()
    print("Checkpointer schema created successfully")

# Then create the regular checkpointer with the pool
checkpointer = PostgresSaver(pool)

# ========================== GRAPH ==========================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)