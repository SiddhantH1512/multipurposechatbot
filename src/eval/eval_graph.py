# src/eval/eval_graph.py
import os
from langchain_core.messages import HumanMessage
from src.backend.langgraph_backend import _build_self_rag_graph, tools
from src.config import Config

# Force localhost for evaluation
os.environ["POSTGRES_CONNECTION"] = "postgresql+psycopg://postgres:Siddhant1512!@localhost:5433/chatbot_db"

# Override the connection info used by checkpointer
Config.POSTGRES_CONNINFO = (
    "host=localhost "
    "port=5433 "
    "dbname=chatbot_db "
    "user=postgres "
    "password=Siddhant1512!"
)

print("🔧 Eval Graph: Using localhost Postgres connection")

# Create a fresh graph for evaluation (this avoids the pre-initialized global one)
eval_chatbot = _build_self_rag_graph(tools)

def get_eval_chatbot():
    return eval_chatbot