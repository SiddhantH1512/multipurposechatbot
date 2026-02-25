from typing import Any, Dict, Optional

import requests
from backend.thread_service import _THREAD_METADATA, _THREAD_RETRIEVERS
from config import Config
from schemas.email_schema import EmailExtraction
from backend.models import ChatGrokModel, ChatGeminiModel, ChatOllamaModel
from prompts.email_prompts import email_prompt_template
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


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
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

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search, get_stock_price, calculator, rag_tool, email_action_extractor]





if __name__ == "__main__":
    test_email = "Hi Siddhant,Once your account is full, syncing and other features will be paused:Syncing across devicesSaving shared filesCreating new documentsAdding new uploadsBacking up photosEven adding a single new file could pause syncing across your devices and limit your ability to access your files when you need them."
    print(email_action_extractor(test_email))