from typing import Optional
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import requests
from src.schemas.email_schema import EmailExtraction
from src.models import ChatGrokModel
from src.prompts.email_prompts import email_prompt_template
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

model = ChatGrokModel()

def get_reranker():
    """Lazy-load Flashrank reranker only when first used"""
    global _reranker
    if '_reranker' not in globals():
        from langchain_community.document_compressors import FlashrankRerank
        globals()['_reranker'] = FlashrankRerank()
    return globals()['_reranker']

search = DuckDuckGoSearchRun(region="us-en")

@tool
def email_action_extractor(email_test: str) -> dict:
    '''Extracts actionable tasks from an email.'''
    model = ChatGrokModel()
    structured_model = model.with_structured_output(EmailExtraction)
    prompt = email_prompt_template.format_messages(email_text=email_test)
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
    using Alpha Vantage.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


def build_rag_tool(user_department: str, user_role: str):
    """
    Returns a rag_tool instance scoped to the calling user.

    Access rules:
      - global docs  → everyone can read
      - dept docs    → only users whose department matches OR executives/HR
    """
    privileged_roles = {"HR", "EXECUTIVE"}
    is_privileged = user_role in privileged_roles

    @tool
    def rag_tool(query: str) -> str:
        """
        Retrieves relevant passages from organizational documents the current
        user is authorised to access, using hybrid (vector + BM25) search and
        Flashrank reranking.
        """
        from src.backend.langgraph_backend import vector_store

        print(f"[RAG] Query: '{query}' | user_dept={user_department} role={user_role} privileged={is_privileged}")

        # ── Build the PGVector metadata filter ──────────────────────────────
        # Fetch (1) all global docs + (2) dept-specific docs the user can see.
        # PGVector filter syntax uses $or / $and operators over cmetadata keys.
        if is_privileged:
            # HR and Executives see everything
            pgvector_filter = None
            print("[RAG] Privileged user — no filter applied")
        else:
            pgvector_filter = {
                "$or": [
                    {"visibility": {"$eq": "global"}},
                    {
                        "$and": [
                            {"visibility": {"$eq": "dept"}},
                            {"department": {"$eq": user_department}},
                        ]
                    },
                ]
            }
            print(f"[RAG] Filter: global OR (dept AND department={user_department})")

        # ── Fetch candidates for BM25 ────────────────────────────────────────
        try:
            fetch_kwargs = {"k": 500}
            if pgvector_filter:
                fetch_kwargs["filter"] = pgvector_filter

            all_docs = vector_store.similarity_search("", **fetch_kwargs)
            print(f"[RAG] Authorised document pool: {len(all_docs)} chunks")
        except Exception as e:
            print(f"[RAG] PGVector fetch failed: {e}")
            return f"Retrieval error: {str(e)}"

        if not all_docs:
            if is_privileged:
                return "No organisational documents have been uploaded yet. Please contact HR."
            return (
                "No documents are available for your department. "
                "If you believe this is an error, please contact HR."
            )

        # ── Vector retriever (with filter) ───────────────────────────────────
        retriever_kwargs: dict = {"k": 6}
        if pgvector_filter:
            retriever_kwargs["filter"] = pgvector_filter

        vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=retriever_kwargs,
        )

        # ── BM25 retriever (over pre-filtered pool) ──────────────────────────
        try:
            bm25_retriever = BM25Retriever.from_documents(all_docs, k=6)
        except Exception as e:
            print(f"[RAG] BM25 init failed: {e}")
            bm25_retriever = None

        if bm25_retriever:
            ensemble = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.7, 0.3],
            )
            print("[RAG] Using hybrid vector + BM25 ensemble")
        else:
            ensemble = vector_retriever
            print("[RAG] Falling back to vector-only")

        # ── Rerank ───────────────────────────────────────────────────────────
        try:
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=get_reranker(),
                base_retriever=ensemble,
            )
            docs = compression_retriever.invoke(query)
            print(f"[RAG] After reranking: {len(docs)} docs")
        except Exception as e:
            print(f"[RAG] Reranking failed: {e} — falling back to ensemble")
            docs = ensemble.invoke(query)
            print(f"[RAG] Fallback retrieval: {len(docs)} docs")

        if not docs:
            return "No relevant passages found in the documents you have access to."

        # ── Format output ────────────────────────────────────────────────────
        context_blocks, sources = [], []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "N/A")
            filename = doc.metadata.get("filename", "document.pdf")
            uploaded_by = doc.metadata.get("uploaded_by_email", "HR")
            context_blocks.append(
                f"--- Excerpt {i} (from {filename}, page {page}) ---\n"
                f"{doc.page_content.strip()}\n"
            )
            sources.append(f"[{i}] {filename} – page {page} (uploaded by: {uploaded_by})")

        return (
            "Relevant excerpts from organisational documents (hybrid search + reranked):\n\n"
            + "\n".join(context_blocks)
            + "\n\nSources:\n"
            + "\n".join(sources)
        )

    return rag_tool


# ── Static tool list (used as default / for non-chat paths) ─────────────────
# rag_tool here has NO user filter — only use this list in non-authenticated
# contexts (e.g. CLI testing). The chat endpoint always calls build_rag_tool().
_default_rag = build_rag_tool(user_department="General", user_role="HR")  # HR sees all
tools = [search, get_stock_price, calculator, _default_rag, email_action_extractor]


if __name__ == "__main__":
    test_email = (
        "Hi Siddhant, Once your account is full, syncing and other features will be paused..."
    )
    print(email_action_extractor(test_email))