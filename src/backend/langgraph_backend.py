from __future__ import annotations
import os
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

import tempfile
import sqlite3
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_huggingface import HuggingFaceEmbeddings
from src.models import ChatOpenAIModel
from src.tools.tool_list import tools


load_dotenv()

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
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

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

        conn = sqlite3.connect("chatbot.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO thread_metadata 
            (thread_id, filename, documents, chunks, index_path)
            VALUES (?, ?, ?, ?, ?)
        """, (str(thread_id), filename, len(docs), len(chunks), CHROMA_PATH))
        conn.commit()
        conn.close()

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

# Checkpointer (SQLite stays)
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ========================== GRAPH ==========================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)