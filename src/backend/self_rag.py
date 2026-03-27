"""
Self-RAG implementation for PolicyIQ.

Implements a 4-stage self-reflective retrieval pipeline:
  1. Retrieval gate       — should we even call the RAG tool?
  2. Relevance grading    — are the retrieved docs actually relevant?
  3. Faithfulness check   — is the generated answer supported by the docs?
  4. Usefulness check     — does the answer resolve the user's query?

If faithfulness is Partial/Not-Supported, the graph re-generates up to
MAX_RETRIES times, tightening the system prompt each loop.
If usefulness fails, the query is rewritten and the retrieval is retried.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
MAX_RETRIES = 3          # max faithfulness re-generation attempts
MAX_QUERY_REWRITES = 2   # max query rewrite + re-retrieval cycles


# ──────────────────────────────────────────────────────────────
# Pydantic schemas for structured LLM outputs
# ──────────────────────────────────────────────────────────────

class RetrievalDecision(BaseModel):
    """Should the RAG tool be called for this query?"""
    need_retrieval: bool = Field(
        description="True if the query requires document retrieval, False for general / conversational queries."
    )
    reasoning: str = Field(description="Brief one-sentence justification.")


class RelevanceGrade(BaseModel):
    """Is a retrieved document relevant to the user query?"""
    relevant: bool = Field(
        description="True if the document chunk is relevant to answering the query."
    )
    reasoning: str = Field(description="Brief one-sentence justification.")


class FaithfulnessGrade(BaseModel):
    """Is the answer grounded in the provided documents?"""
    grade: Literal["fully_supported", "partially_supported", "not_supported"] = Field(
        description=(
            "fully_supported   — every claim in the answer is backed by the documents.\n"
            "partially_supported — some claims are backed, others are hallucinated.\n"
            "not_supported      — answer is not grounded in the provided documents."
        )
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="List the specific claims NOT supported by the documents (empty if fully_supported).",
    )
    reasoning: str = Field(description="One-sentence overall justification.")


class UsefulnessGrade(BaseModel):
    """Does the answer actually resolve the user's query?"""
    useful: bool = Field(
        description="True if the answer fully or sufficiently resolves the user's query."
    )
    missing: str = Field(
        default="",
        description="What the answer is missing (empty string if useful=True).",
    )


class RewrittenQuery(BaseModel):
    """A rewritten, clearer version of the original query."""
    query: str = Field(description="The improved search query.")
    reasoning: str = Field(description="Why this rewrite will yield better results.")


# ──────────────────────────────────────────────────────────────
# Self-RAG state (extends the base ChatState)
# ──────────────────────────────────────────────────────────────

class SelfRAGState(TypedDict):
    # Core conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Self-RAG tracking fields
    original_query: str              # the unmodified user question
    current_query: str               # may be rewritten
    retrieved_context: str           # raw context from rag_tool
    relevant_context: str            # context after relevance filtering
    generated_answer: str            # latest draft answer
    faithfulness_grade: str          # fully_supported | partially_supported | not_supported
    unsupported_claims: list[str]    # claims flagged as hallucinations
    retry_count: int                 # faithfulness retry counter
    rewrite_count: int               # query-rewrite counter
    need_retrieval: bool             # retrieval gate decision
    skip_retrieval: bool             # set True when no relevant docs found
    answer_useful: bool              # usefulness gate result


# ──────────────────────────────────────────────────────────────
# Helper: build grader chains for a given LLM
# ──────────────────────────────────────────────────────────────

def build_graders(llm):
    """
    Returns a dict of grader chains, each backed by the provided LLM
    with structured output.  Call once per chatbot instance.
    """

    # ── 1. Retrieval gate ─────────────────────────────────────
    retrieval_gate_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at deciding whether a user query requires searching "
         "organisational documents (policies, HR guides, company info) or can be "
         "answered directly from general knowledge.\n\n"
         "Return need_retrieval=True for: policy questions, HR procedure questions, "
         "document-specific questions, or anything that references 'our', 'the company', "
         "'policy', 'procedure', 'handbook', 'guidelines'.\n"
         "Return need_retrieval=False for: general knowledge, math, greetings, "
         "stock prices, weather, or any clearly non-document question."),
        ("human", "User query: {query}"),
    ])
    retrieval_gate = retrieval_gate_prompt | llm.with_structured_output(RetrievalDecision)

    # ── 2. Relevance grader ───────────────────────────────────
    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a relevance grader. Given a document excerpt and a user query, "
         "decide if the excerpt contains information that could help answer the query.\n"
         "Be permissive — mark relevant=True even if the excerpt only partially addresses "
         "the query.  Only mark relevant=False if the excerpt is completely unrelated."),
        ("human",
         "Query: {query}\n\n"
         "Document excerpt:\n{document}"),
    ])
    relevance_grader = relevance_prompt | llm.with_structured_output(RelevanceGrade)

    # ── 3. Faithfulness grader ────────────────────────────────
    faithfulness_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict faithfulness auditor.  Given an AI-generated answer and the "
         "source documents it was supposed to use, determine whether every factual claim "
         "in the answer is supported by the documents.\n\n"
         "Grading rubric:\n"
         "  fully_supported     — ALL claims are directly supported.\n"
         "  partially_supported — SOME claims are supported, but at least one is not.\n"
         "  not_supported       — the answer is not grounded in the documents at all.\n\n"
         "List every unsupported claim explicitly."),
        ("human",
         "Source documents:\n{context}\n\n"
         "AI-generated answer:\n{answer}"),
    ])
    faithfulness_grader = faithfulness_prompt | llm.with_structured_output(FaithfulnessGrade)

    # ── 4. Usefulness grader ──────────────────────────────────
    usefulness_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict answer quality evaluator for a company document RAG system.\n\n"
         "Your job is to decide if the AI's answer properly resolves the user's query "
         "based on the retrieved company documents.\n\n"
         "Evaluation Rules:\n"
         "- useful=True ONLY if the answer is directly supported by the documents and fully addresses the query.\n"
         "- useful=True is also acceptable if the answer clearly and honestly states that the information "
         "is NOT mentioned in the uploaded documents.\n"
         "- useful=False if the answer gives general advice, industry standards, speculation, or 'it varies by company' "
         "when the user asked about THIS company's document.\n"
         "- useful=False if the answer is vague, evasive, or hallucinates information not present in the documents."),
        ("human",
         "User query: {query}\n\n"
         "AI answer:\n{answer}"),
    ])
    usefulness_grader = usefulness_prompt | llm.with_structured_output(UsefulnessGrade)

    # ── 5. Query rewriter ─────────────────────────────────────
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a search query optimiser. Given the user's original question and "
         "the reason the previous retrieval was not useful, rewrite the query to be "
         "more specific, use different terminology, or decompose it into a simpler "
         "sub-question that is more likely to match documents in an organisational "
         "knowledge base."),
        ("human",
         "Original query: {query}\n"
         "Problem with previous retrieval: {problem}"),
    ])
    query_rewriter = rewrite_prompt | llm.with_structured_output(RewrittenQuery)

    return {
        "retrieval_gate": retrieval_gate,
        "relevance_grader": relevance_grader,
        "faithfulness_grader": faithfulness_grader,
        "usefulness_grader": usefulness_grader,
        "query_rewriter": query_rewriter,
    }


# ──────────────────────────────────────────────────────────────
# Node functions (pure functions — stateless, accept state dict)
# ──────────────────────────────────────────────────────────────

def make_retrieval_gate_node(graders: dict):
    """Returns a node that decides whether retrieval is needed."""

    def retrieval_gate_node(state: SelfRAGState) -> dict:
        # Extract the latest human message
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        print(f"[Self-RAG] Retrieval gate → query='{query[:80]}...'")
        decision: RetrievalDecision = graders["retrieval_gate"].invoke({"query": query})
        print(f"[Self-RAG] need_retrieval={decision.need_retrieval} | {decision.reasoning}")

        return {
            "original_query": query,
            "current_query": query,
            "need_retrieval": decision.need_retrieval,
            "skip_retrieval": False,
            "retry_count": 0,
            "rewrite_count": 0,
            "retrieved_context": "",
            "relevant_context": "",
            "generated_answer": "",
            "faithfulness_grade": "",
            "unsupported_claims": [],
            "answer_useful": False,
        }

    return retrieval_gate_node


def make_relevance_filter_node(graders: dict):
    """
    After the rag_tool has run, filters retrieved chunks to only the relevant ones.
    Expects the last ToolMessage in state to contain the raw RAG output.
    """

    def relevance_filter_node(state: SelfRAGState) -> dict:
        query = state.get("current_query", state.get("original_query", ""))

        # Find latest ToolMessage
        raw_context = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "rag_tool":
                raw_context = msg.content
                break

        if not raw_context:
            print("[Self-RAG] No rag_tool message found — skipping relevance filter")
            return {"retrieved_context": "", "relevant_context": "", "skip_retrieval": True}

        # Split into individual excerpt blocks
        excerpts = re.split(r"(?=--- Excerpt \d+)", raw_context)
        excerpts = [e.strip() for e in excerpts if e.strip() and "--- Excerpt" in e]

        if not excerpts:
            # Context may not be in the expected format (e.g. error message)
            print(f"[Self-RAG] Context not in excerpt format — using raw: {raw_context[:100]}")
            return {
                "retrieved_context": raw_context,
                "relevant_context": raw_context,
                "skip_retrieval": False,
            }

        print(f"[Self-RAG] Grading {len(excerpts)} excerpts for relevance …")
        relevant_excerpts = []
        for i, excerpt in enumerate(excerpts):
            try:
                grade: RelevanceGrade = graders["relevance_grader"].invoke(
                    {"query": query, "document": excerpt}
                )
                if grade.relevant:
                    relevant_excerpts.append(excerpt)
                    print(f"[Self-RAG]   Excerpt {i+1}: ✅ relevant")
                else:
                    print(f"[Self-RAG]   Excerpt {i+1}: ❌ irrelevant — {grade.reasoning}")
            except Exception as e:
                print(f"[Self-RAG]   Excerpt {i+1}: grading error ({e}) — keeping")
                relevant_excerpts.append(excerpt)  # keep on error

        if not relevant_excerpts:
            print("[Self-RAG] No relevant excerpts found — will skip retrieval answer")
            return {
                "retrieved_context": raw_context,
                "relevant_context": "",
                "skip_retrieval": True,
            }

        filtered = "\n\n".join(relevant_excerpts)
        print(f"[Self-RAG] {len(relevant_excerpts)}/{len(excerpts)} excerpts passed relevance filter")
        return {
            "retrieved_context": raw_context,
            "relevant_context": filtered,
            "skip_retrieval": False,
        }

    return relevance_filter_node


def make_generate_node(llm_with_tools, llm):
    """
    Generate an answer.
    - If skip_retrieval=True  → answer from general knowledge.
    - On retry              → tighten the system prompt to remove hallucinations.
    """

    def generate_node(state: SelfRAGState) -> dict:
        retry_count = state.get("retry_count", 0)
        skip = state.get("skip_retrieval", False)
        context = state.get("relevant_context", "")
        unsupported = state.get("unsupported_claims", [])

        if skip or not context:
            system = (
                "You are a helpful organisational assistant. "
                "No relevant information was found in the uploaded company documents for this query.\n\n"
                "Rules:\n"
                "- Clearly state that the requested information is NOT available in the documents.\n"
                "- Do NOT give general advice or industry standards.\n"
                "- Do NOT speculate or say 'it can vary'.\n"
                "- Suggest the user check the official HR policy or employee handbook if appropriate.\n"
                "- Keep the response short and direct."
            )
        elif retry_count == 0:
            system = (
                "You are a helpful organisational assistant. "
                "Answer ONLY using the provided document excerpts. "
                "Include citations in the format [Source N: filename – page X]. "
                "Do NOT add information not present in the excerpts."
                "\n\nDocument excerpts:\n" + context
            )
        else:
            # Tightened prompt on retry
            unsupported_str = "\n".join(f"  - {c}" for c in unsupported) if unsupported else "  (see previous response)"
            system = (
                "You are a strict organisational assistant. "
                "Your previous answer contained unsupported claims. "
                "You MUST remove ALL claims not directly found in the documents below.\n\n"
                f"Unsupported claims to REMOVE:\n{unsupported_str}\n\n"
                "Answer ONLY from the document excerpts below. "
                "Include citations [Source N: filename – page X].\n\n"
                "Document excerpts:\n" + context
            )

        messages = [SystemMessage(content=system)] + state["messages"]
        response: AIMessage = llm.invoke(messages)

        print(f"[Self-RAG] Generated answer (retry={retry_count}): {response.content[:120]}…")
        return {
            "generated_answer": response.content,
            "messages": [response],
        }

    return generate_node


def make_faithfulness_node(graders: dict):
    """Grade whether the generated answer is faithful to the retrieved context."""

    def faithfulness_node(state: SelfRAGState) -> dict:
        context = state.get("relevant_context", "")
        answer = state.get("generated_answer", "")
        skip = state.get("skip_retrieval", False)

        # If we skipped retrieval, faithfulness is N/A — mark as supported
        if skip or not context:
            return {
                "faithfulness_grade": "fully_supported",
                "unsupported_claims": [],
            }

        try:
            grade: FaithfulnessGrade = graders["faithfulness_grader"].invoke(
                {"context": context, "answer": answer}
            )
            print(f"[Self-RAG] Faithfulness: {grade.grade} | {grade.reasoning}")
            if grade.unsupported_claims:
                print(f"[Self-RAG] Unsupported claims: {grade.unsupported_claims}")
            return {
                "faithfulness_grade": grade.grade,
                "unsupported_claims": grade.unsupported_claims,
            }
        except Exception as e:
            print(f"[Self-RAG] Faithfulness grading error: {e} — assuming fully_supported")
            return {"faithfulness_grade": "fully_supported", "unsupported_claims": []}

    return faithfulness_node


def make_usefulness_node(graders: dict):
    """Grade whether the answer is actually useful to the user."""

    def usefulness_node(state: SelfRAGState) -> dict:
        query = state.get("original_query", "")
        answer = state.get("generated_answer", "")

        try:
            grade: UsefulnessGrade = graders["usefulness_grader"].invoke(
                {"query": query, "answer": answer}
            )
            print(f"[Self-RAG] Usefulness: useful={grade.useful} | missing='{grade.missing}'")
            return {
                "answer_useful": grade.useful,
                # Store missing info in unsupported_claims temporarily for rewrite
                "unsupported_claims": [grade.missing] if not grade.useful and grade.missing else state.get("unsupported_claims", []),
            }
        except Exception as e:
            print(f"[Self-RAG] Usefulness grading error: {e} — assuming useful")
            return {"answer_useful": True}

    return usefulness_node


def make_query_rewrite_node(graders: dict):
    """Rewrite the query when the answer is not useful."""

    def query_rewrite_node(state: SelfRAGState) -> dict:
        query = state.get("current_query", state.get("original_query", ""))
        problem = state.get("unsupported_claims", ["answer was not useful"])
        problem_str = problem[0] if problem else "answer was not useful or incomplete"
        rewrite_count = state.get("rewrite_count", 0) + 1

        try:
            rewrite: RewrittenQuery = graders["query_rewriter"].invoke(
                {"query": query, "problem": problem_str}
            )
            new_query = rewrite.query
            print(f"[Self-RAG] Query rewrite #{rewrite_count}: '{new_query}' | {rewrite.reasoning}")
        except Exception as e:
            print(f"[Self-RAG] Query rewrite error: {e} — keeping original")
            new_query = query

        # Inject the rewritten query as a new HumanMessage so the RAG tool sees it
        return {
            "current_query": new_query,
            "rewrite_count": rewrite_count,
            "skip_retrieval": False,
            "retrieved_context": "",
            "relevant_context": "",
            "messages": [HumanMessage(content=new_query)],
        }

    return query_rewrite_node


# ──────────────────────────────────────────────────────────────
# Routing / conditional edge functions
# ──────────────────────────────────────────────────────────────

def route_after_gate(state: SelfRAGState) -> str:
    """After retrieval gate: call rag_tool or go straight to generate."""
    if state.get("need_retrieval", True):
        return "call_rag"
    return "generate"


def route_after_faithfulness(state: SelfRAGState) -> str:
    """After faithfulness check: retry, or move to usefulness check."""
    grade = state.get("faithfulness_grade", "fully_supported")
    retry_count = state.get("retry_count", 0)

    if grade in ("partially_supported", "not_supported") and retry_count < MAX_RETRIES:
        print(f"[Self-RAG] Faithfulness={grade} → re-generating (attempt {retry_count + 1}/{MAX_RETRIES})")
        return "retry_generate"
    return "check_usefulness"


def route_after_usefulness(state: SelfRAGState) -> str:
    """After usefulness check: rewrite query or finish."""
    useful = state.get("answer_useful", True)
    rewrite_count = state.get("rewrite_count", 0)

    if not useful and rewrite_count < MAX_QUERY_REWRITES:
        print(f"[Self-RAG] Answer not useful → rewriting query (rewrite {rewrite_count + 1}/{MAX_QUERY_REWRITES})")
        return "rewrite_query"
    return "finish"


def increment_retry(state: SelfRAGState) -> dict:
    """Helper node: increment retry_count before re-generating."""
    return {"retry_count": state.get("retry_count", 0) + 1}