from typing import Optional
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import requests
from src.schemas.email_schema import EmailExtraction
from src.models import ChatGrokModel
from src.prompts.email_prompts import email_prompt_template
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import json

model = ChatGrokModel()
_reranker = FlashrankRerank()

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
    ONLY use this for simple math when explicitly asked to calculate numbers.
    Do NOT use for anything else, especially not for processing text or context.
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



@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> str:
    """
    AI/ML domain RAG tool. Retrieves relevant passages using hybrid (vector + BM25)
    search and reranks them with Flashrank. Returns formatted context for the LLM.
    """
    from src.backend.langgraph_backend import vector_store

    if not thread_id:
        return "Error: thread_id is required for document-specific search."

    print(f"[RAG] Query: {query}")
    print(f"[RAG] Thread filter: {thread_id}")

    # ───────────────────────────────────────────────────────────────
    # Fetch all documents for this thread using similarity_search with filter
    # (PGVector supports metadata filtering via where clause)
    # ───────────────────────────────────────────────────────────────
    try:
        # Use retriever with filter to get documents (this is the correct way)
        filtered_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 50,  # fetch more candidates since we filter + rerank
                "filter": {"thread_id": str(thread_id)}
            }
        )

        # For BM25 fallback, we still need all docs → fetch them separately
        # PGVector .similarity_search() with k=0 or large k can approximate "get all"
        all_docs = vector_store.similarity_search(
            query="",  # empty query = fetch all (with filter)
            k=1000,    # large enough to get everything in thread
            filter={"thread_id": str(thread_id)}
        )

        print(f"[RAG] Found {len(all_docs)} raw documents for thread {thread_id}")

    except Exception as e:
        print(f"[RAG] PGVector retrieval failed: {e}")
        return f"Retrieval error: {str(e)}"

    if not all_docs:
        return "No documents have been uploaded or indexed in this conversation yet."

    # ───────────────────────────────────────────────────────────────
    # Continue with BM25 + ensemble + reranking (same as before)
    # ───────────────────────────────────────────────────────────────
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6,
            "filter": {"thread_id": str(thread_id)}
        }
    )

    try:
        bm25_retriever = BM25Retriever.from_documents(
            all_docs,  # use the full docs we fetched
            k=6
        )
    except Exception as e:
        print(f"[RAG] BM25 initialization failed: {e}")
        bm25_retriever = None

    if bm25_retriever:
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        print("[RAG] Using hybrid vector + BM25 ensemble")
    else:
        ensemble = vector_retriever
        print("[RAG] Falling back to vector-only")

    try:
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=_reranker,
            base_retriever=ensemble
        )
        docs = compression_retriever.invoke(query)
        print(f"[RAG] After hybrid retrieval + Flashrank reranking: {len(docs)} docs")
    except Exception as e:
        print(f"[RAG] Reranking failed: {e} → falling back to ensemble")
        docs = ensemble.invoke(query)
        print(f"[RAG] Fallback ensemble retrieval: {len(docs)} docs")

    if not docs:
        return "No relevant passages found in the uploaded document(s)."

    # Format output (same as before)
    context_blocks = []
    sources = []

    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "N/A")
        filename = doc.metadata.get("filename", "document.pdf")

        context_blocks.append(
            f"--- Excerpt {i} (from {filename}, page {page}) ---\n"
            f"{doc.page_content.strip()}\n"
        )
        sources.append(f"[{i}] {filename} – page {page}")

    readable_output = (
        "Relevant excerpts from uploaded documents (hybrid search + reranked):\n\n"
        + "\n".join(context_blocks)
        + "\n\nSources:\n"
        + "\n".join(sources)
    )

    return readable_output



tools = [search, get_stock_price, calculator, rag_tool, email_action_extractor]





if __name__ == "__main__":
    test_email = "Hi Siddhant,Once your account is full, syncing and other features will be paused:Syncing across devicesSaving shared filesCreating new documentsAdding new uploadsBacking up photosEven adding a single new file could pause syncing across your devices and limit your ability to access your files when you need them."
    print(email_action_extractor(test_email))