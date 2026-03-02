import os
import sqlite3
from typing import Any, Dict, Optional

from langchain_core.retrievers import BaseRetriever
import requests
from src.config import Config
from src.schemas.email_schema import EmailExtraction
from src.models import ChatGrokModel, ChatGeminiModel
from src.prompts.email_prompts import email_prompt_template
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import json

model = ChatGrokModel()

search = DuckDuckGoSearchRun(region="us-en")
@tool
def email_action_extractor(email_test: str) -> dict:
    '''
    Extracts actionable tasks from an email.
    '''
    model = ChatGrokModel()
    structured_model = model.with_structured_output(EmailExtraction)
    prompt = email_prompt_template.format_messages(
        email_text=email_test
    )
    result: EmailExtraction = structured_model.invoke(prompt)

    return result.model_dump()


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


def _get_retriever(thread_id: Optional[str]) -> Optional[BaseRetriever]:
    """Load FAISS retriever from disk for the given thread, or return None."""
    if not thread_id:
        return None

    # Lazy imports — only executed when function is called
    from src.backend.langgraph_backend import embeddings
    from langchain_community.vectorstores import FAISS

    try:
        conn = sqlite3.connect("chatbot.db", timeout=5)  # small timeout to avoid hanging
        cursor = conn.cursor()
        cursor.execute(
            "SELECT index_path FROM thread_metadata WHERE thread_id = ?",
            (str(thread_id),)
        )
        row = cursor.fetchone()
    except sqlite3.Error as e:
        print(f"DB error while loading retriever for {thread_id}: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

    if not row or not row[0]:
        return None

    index_path = row[0]

    if not os.path.isdir(index_path):  # FAISS save_local creates a directory
        print(f"Index path does not exist or is not a directory: {index_path}")
        return None

    try:
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
    except Exception as e:
        print(f"Failed to load FAISS index for thread {thread_id}: {e}")
        return None

    

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    # ── This is the corrected part ──
    from src.backend.thread_service import thread_document_metadata   # lazy import here too

    doc_meta = thread_document_metadata(thread_id) if thread_id else {}
    source_file = doc_meta.get("filename", "unknown")

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": source_file,
    }

tools = [search, get_stock_price, calculator, rag_tool, email_action_extractor]





if __name__ == "__main__":
    test_email = "Hi Siddhant,Once your account is full, syncing and other features will be paused:Syncing across devicesSaving shared filesCreating new documentsAdding new uploadsBacking up photosEven adding a single new file could pause syncing across your devices and limit your ability to access your files when you need them."
    print(email_action_extractor(test_email))