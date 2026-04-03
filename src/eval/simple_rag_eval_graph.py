# src/eval/minimal_rag_eval.py
import os
from langchain_core.messages import HumanMessage
from src.backend.langgraph_backend import tools
from src.config import Config

# Force localhost
os.environ["POSTGRES_CONNECTION"] = "postgresql+psycopg://postgres:Siddhant1512!@localhost:5433/chatbot_db"
Config.POSTGRES_CONNINFO = (
    "host=localhost port=5433 dbname=chatbot_db user=postgres password=Siddhant1512!"
)

print("🔧 Minimal RAG Eval: Creating basic graph with only rag_tool")

# Create a very minimal graph for evaluation (only rag_tool + basic LLM)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate
from src.models import ChatOpenAIModel

llm = ChatOpenAIModel()

def minimal_rag_node(state):
    """Simple node that forces RAG tool usage for evaluation"""
    return {"messages": state["messages"]}

graph = StateGraph(dict)
graph.add_node("agent", minimal_rag_node)
graph.add_node("tools", ToolNode([t for t in tools if t.name == "rag_tool"]))  # only rag_tool

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", END)

minimal_eval_chatbot = graph.compile()

def get_minimal_eval_chatbot():
    return minimal_eval_chatbot