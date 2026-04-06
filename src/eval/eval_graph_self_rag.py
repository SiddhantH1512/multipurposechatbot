"""
Builds a Self-RAG graph for evaluation using the REAL production
_build_self_rag_graph factory from langgraph_backend.

Key differences from the old eval_graph.py:
  - Uses HR-level privileged rag_tool (sees all documents in the default tenant)
  - Overrides Postgres + Redis to localhost before any lazy import
  - Exposes the compiled graph via get_self_rag_eval_chatbot()

Do NOT import this at module level from production code — it exists
purely for offline RAGAS evaluation runs.
"""

import os

# ── Force localhost BEFORE any src.config / src.database imports ────────────
os.environ["POSTGRES_CONNECTION"] = (
    "postgresql+asyncpg://postgres:Siddhant1512!@localhost:5433/chatbot_db"
)
os.environ["REDIS_URL"] = "redis://localhost:6379"

# Also patch Config at runtime (in case it was already imported)
from src.config import Config
Config.POSTGRES_CONNECTION = os.environ["POSTGRES_CONNECTION"]
Config.REDIS_URL            = os.environ["REDIS_URL"]
Config.POSTGRES_CONNINFO    = (
    "host=localhost port=5433 dbname=chatbot_db "
    "user=postgres password=Siddhant1512!"
)

print("🔧 eval_graph_self_rag: localhost Postgres (5433) + Redis (6379)")

# ── Now safe to import the backend ──────────────────────────────────────────
from src.backend.langgraph_backend import _build_self_rag_graph
from src.tools.tool_list import (
    search,
    get_stock_price,
    calculator,
    build_rag_tool,
    email_action_extractor,
)

from langchain_openai import ChatOpenAI
_eval_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")   # make sure this is set
)
# Build a privileged (HR-level) rag_tool so it can see ALL documents
# in the default tenant — same access level as the ingest user.
_eval_rag_tool = build_rag_tool(
    user_department="ALL",
    user_role="HR",          # HR = privileged, sees all docs
    tenant_id="company2",
)

_eval_tools = [search, get_stock_price, calculator, _eval_rag_tool, email_action_extractor]

# Compile once and reuse across all eval questions
_eval_chatbot = _build_self_rag_graph(_eval_tools, user_llm=_eval_llm)


def get_self_rag_eval_chatbot():
    """Return the compiled Self-RAG evaluation chatbot."""
    return _eval_chatbot