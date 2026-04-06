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
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_huggingface import HuggingFaceEmbeddings
from src.models import ChatOpenAIModel
from src.tools.tool_list import tools, build_rag_tool, email_action_extractor, calculator, get_stock_price, search
from langchain_postgres import PGVector
from sqlalchemy import text
from src.database.engine import *
from sqlalchemy.ext.asyncio import AsyncSession


# ── Self-RAG imports ─────────────────────────────────────────
from src.backend.self_rag import (
    SelfRAGState,
    build_graders,
    make_followup_node,
    make_retrieval_gate_node,
    make_relevance_filter_node,
    make_generate_node,
    make_faithfulness_node,
    make_usefulness_node,
    make_query_rewrite_node,
    increment_retry,
    route_after_gate,
    route_after_faithfulness,
    route_after_usefulness,
    MAX_RETRIES,
    MAX_QUERY_REWRITES,
)

# ────────────────────────────────────────────────────────────────────────
# Embeddings — BAAI/bge-large-en-v1.5
# ────────────────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ── LLM Semantic Cache with Redis ────────────────────────────────────────
from langchain_redis import RedisSemanticCache
from langchain_core.globals import set_llm_cache
import logging

logger = logging.getLogger(__name__)
_semantic_cache = None

def init_semantic_cache():
    global _semantic_cache
    if _semantic_cache is not None:
        return _semantic_cache
    try:
        _semantic_cache = RedisSemanticCache(
            redis_url=Config.REDIS_URL,
            embeddings=embeddings,
            score_threshold=0.25,
            ttl=3600,
        )
        set_llm_cache(_semantic_cache)
        logger.info("✅ LLM Semantic Cache initialized")
        return _semantic_cache
    except Exception as e:
        logger.warning(f"⚠️ Semantic cache unavailable: {e}")
        return None

def get_semantic_cache():
    return init_semantic_cache()

load_dotenv()

llm = ChatOpenAIModel()
llm_with_tools = llm.bind_tools(tools)

# ────────────────────────────────────────────────────────────────────────
# Vector store
# ────────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "chatbot_documents"

def get_vector_store():
    return PGVector(
        connection=sync_engine,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        distance_strategy="cosine",
    )

async def ensure_extension():
    async with async_engine.connect() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.commit()
    print("pgvector extension ensured.")

vector_store = get_vector_store()

# ────────────────────────────────────────────────────────────────────────
# Checkpointer
# ────────────────────────────────────────────────────────────────────────
pool = ConnectionPool(Config.POSTGRES_CONNINFO)
checkpointer = PostgresSaver(pool)


# ────────────────────────────────────────────────────────────────────────
# Chunking — flat 800-char section-aware splits
#
# Why flat (not hierarchical parent-child):
#   Parent-child retrieval was tested and caused Context Precision to drop
#   from 0.63 → 0.56 and Context Recall from 0.54 → 0.45. The 400-char
#   child chunks lack enough semantic context for bge-large to score them
#   reliably, and the parent lookup added failure modes (Redis cold start,
#   deduplication reducing coverage from k=8 to ~4-6 unique sections).
#
# Why 800 chars (not the original 1200):
#   1200-char chunks split policy sections so the heading lands in one
#   chunk and the value in the next. At 800 chars, a single numbered
#   policy clause (eligibility + exception + condition) typically fits
#   in one chunk, keeping the general rule and its restrictions together.
#
# Section-aware separators split at numbered headings and section markers
# before falling back to paragraph/line breaks — preserving clause
# boundaries without any embedding calls at ingest time.
# ────────────────────────────────────────────────────────────────────────

SECTION_AWARE_SEPARATORS = [
    r"\n(?=\d+\.\d+\s)",          # "3.2 Sub-section"
    r"\n(?=\d+\.\s+[A-Z])",       # "3. Section Title"
    r"\n(?=Section\s+\d+)",        # "Section 4"
    r"\n(?=[A-Z][a-z]+\s+\d+:)",  # "Stage 4:"
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=SECTION_AWARE_SEPARATORS,
    is_separator_regex=True,
    length_function=len,
)


# ────────────────────────────────────────────────────────────────────────
# Ingestion — flat chunking
# ────────────────────────────────────────────────────────────────────────
async def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user),
    visibility: str = "global",
    department: Optional[str] = None,
) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        temp_path = f.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        doc_department = "General" if visibility == "global" else (department or current_user.department)
        tenant_id      = getattr(current_user, "tenant_id", "default")
        document_id    = str(uuid.uuid4())

        chunks = _splitter.split_documents(docs)
        print(f"[Ingest] {filename}: {len(docs)} pages → {len(chunks)} chunks")

        # Deduplication
        await session.execute(text("""
            DELETE FROM langchain_pg_embedding
            WHERE cmetadata->>'filename' = :filename
              AND cmetadata->>'tenant_id' = :tenant_id
        """), {"filename": filename, "tenant_id": tenant_id})

        await session.execute(text("""
            DELETE FROM thread_metadata
            WHERE filename = :filename
              AND tenant_id = :tenant_id
        """), {"filename": filename, "tenant_id": tenant_id})

        await session.commit()

        for chunk in chunks:
            chunk.metadata.update({
                "document_id":       document_id,
                "tenant_id":         tenant_id,
                "visibility":        visibility,
                "department":        doc_department,
                "uploaded_by":       current_user.id,
                "uploaded_by_email": current_user.email,
                "filename":          filename,
                "page":              chunk.metadata.get("page", 0) + 1,
                "uploaded_at":       datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })

        vector_store.add_documents(chunks)

        await session.execute(text("""
            INSERT INTO thread_metadata
            (thread_id, filename, documents, chunks, user_id, department,
             is_global, document_id, tenant_id)
            VALUES
            (:thread_id, :filename, :documents, :chunks, :user_id, :dept,
             :is_global, :document_id, :tenant_id)
            ON CONFLICT (thread_id) DO UPDATE SET
                filename    = EXCLUDED.filename,
                documents   = EXCLUDED.documents,
                chunks      = EXCLUDED.chunks,
                user_id     = EXCLUDED.user_id,
                department  = EXCLUDED.department,
                is_global   = EXCLUDED.is_global,
                document_id = EXCLUDED.document_id,
                tenant_id   = EXCLUDED.tenant_id,
                updated_at  = CURRENT_TIMESTAMP
        """), {
            "thread_id":   str(thread_id),
            "filename":    filename,
            "documents":   len(docs),
            "chunks":      len(chunks),
            "user_id":     current_user.id,
            "dept":        current_user.department,
            "is_global":   visibility == "global",
            "document_id": document_id,
            "tenant_id":   tenant_id,
        })

        await session.commit()

        return {
            "filename":          filename,
            "documents":         len(docs),
            "chunks":            len(chunks),
            "department":        doc_department,
            "visibility":        visibility,
            "uploaded_by":       current_user.email,
            "document_id":       document_id,
            "target_department": department if visibility == "dept" else None,
            "tenant_id":         tenant_id,
        }

    finally:
        os.unlink(temp_path)


# ────────────────────────────────────────────────────────────────────────
# Self-RAG graph factory
# ────────────────────────────────────────────────────────────────────────

def _build_self_rag_graph(tool_list: list, user_llm=None):
    _llm = user_llm or llm
    _llm_with_tools = _llm.bind_tools(tool_list)

    graders = build_graders(_llm)
    tool_node = ToolNode(tool_list)

    retrieval_gate_node   = make_retrieval_gate_node(graders)
    relevance_filter_node = make_relevance_filter_node(graders)
    generate_node         = make_generate_node(_llm_with_tools, _llm)
    faithfulness_node     = make_faithfulness_node(graders)
    usefulness_node       = make_usefulness_node(graders)
    query_rewrite_node    = make_query_rewrite_node(graders)

    def direct_chat_node(state: SelfRAGState):
        system = "You are a helpful organisational assistant."
        messages = [SystemMessage(content=system)] + state["messages"]
        response: AIMessage = _llm.invoke(messages)
        return {
            "generated_answer":     response.content,
            "messages":             [response],
            "faithfulness_grade":   "fully_supported",
            "answer_useful":        True,
            "is_conflict_question": False,
        }

    g = StateGraph(SelfRAGState)

    g.add_node("retrieval_gate",     retrieval_gate_node)
    g.add_node("agent",              lambda state: {"messages": [_llm_with_tools.invoke(state["messages"])]})
    g.add_node("call_rag",           tool_node)
    g.add_node("relevance_filter",   relevance_filter_node)
    g.add_node("generate",           generate_node)
    g.add_node("direct_chat",        direct_chat_node)
    g.add_node("faithfulness",       faithfulness_node)
    g.add_node("increment_retry",    increment_retry)
    g.add_node("usefulness",         usefulness_node)
    g.add_node("query_rewrite",      query_rewrite_node)
    g.add_node("generate_followups", make_followup_node(graders))

    g.add_edge(START, "retrieval_gate")
    g.add_conditional_edges("retrieval_gate", route_after_gate,
                            {"call_rag": "agent", "generate": "direct_chat"})
    g.add_conditional_edges("agent", tools_condition,
                            {"tools": "call_rag", END: "generate"})
    g.add_edge("call_rag",         "relevance_filter")
    g.add_edge("relevance_filter", "generate")
    g.add_edge("generate",         "faithfulness")
    g.add_conditional_edges("faithfulness", route_after_faithfulness,
                            {"retry_generate": "increment_retry", "check_usefulness": "usefulness"})
    g.add_edge("increment_retry",  "generate")
    g.add_conditional_edges("usefulness", route_after_usefulness,
                            {"rewrite_query": "query_rewrite", "finish": "generate_followups"})
    g.add_edge("query_rewrite",      "agent")
    g.add_edge("generate_followups", END)
    g.add_edge("direct_chat",        END)

    return g.compile(checkpointer=checkpointer)


# ────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────

chatbot = _build_self_rag_graph(tools)


def build_chatbot(user: "User"):
    dept      = user.department or "General"
    role      = user.role.value if hasattr(user.role, "value") else str(user.role)
    tenant_id = getattr(user, "tenant_id", "default")

    scoped_rag = build_rag_tool(
        user_department=dept,
        user_role=role,
        tenant_id=tenant_id,
    )
    scoped_tools = [search, get_stock_price, calculator, scoped_rag, email_action_extractor]
    return _build_self_rag_graph(scoped_tools)