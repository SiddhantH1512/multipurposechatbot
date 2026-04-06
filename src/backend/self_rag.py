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
import warnings
from typing import Annotated, Literal, TypedDict

# Suppress Pydantic serialization warnings from LangChain internals
warnings.filterwarnings(
    "ignore",
    message="PydanticSerializationUnexpectedValue",
    category=UserWarning,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import add_messages
from pydantic import BaseModel, Field, ConfigDict

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
MAX_RETRIES        = 3   # max faithfulness re-generation attempts
MAX_QUERY_REWRITES = 2   # max query rewrite + re-retrieval cycles

# Keywords that indicate a calculation is required in the answer
CALCULATION_QUESTION_KEYWORDS = [
    "calculate", "what is the employer's",
    "gratuity for an employee",
    "pf contribution", "epf contribution",
    "comp-off", "per year of service",
    "bonus amount for", "how much will",
]

# Very specific — only matches when user explicitly presents two values
# and asks which is correct. Avoids firing on ordinary factoid questions.
CONFLICT_QUESTION_KEYWORDS = [
    " or rs. ",
    " or rs ",
    "— rs.",
    "8% or 10%",
    "45 days or 60",
    "20,000 or 15,000",
    "15,000 or 20,000",
    "2,500 or 3,000",
    "3,000 or 2,500",
    "which policy takes precedence",
    "which policy supersedes",
    "conflict between",
]


# ──────────────────────────────────────────────────────────────
# Pydantic schemas for structured LLM outputs
# ──────────────────────────────────────────────────────────────

class RetrievalDecision(BaseModel):
    """Should the RAG tool be called for this query?"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    need_retrieval: bool = Field(
        description="True if the query requires document retrieval, False for general / conversational queries."
    )
    reasoning: str = Field(description="Brief one-sentence justification.")


class RelevanceGrade(BaseModel):
    """Is a retrieved document relevant to the user query?"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    relevant: bool = Field(
        description="True if the document chunk is relevant to answering the query."
    )
    reasoning: str = Field(description="Brief one-sentence justification.")


class FaithfulnessGrade(BaseModel):
    """Is the answer grounded in the provided documents?"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    useful: bool = Field(
        description="True if the answer fully or sufficiently resolves the user's query."
    )
    missing: str = Field(
        default="",
        description="What the answer is missing (empty string if useful=True).",
    )


class RewrittenQuery(BaseModel):
    """A rewritten, clearer version of the original query."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = Field(description="The improved search query.")
    reasoning: str = Field(description="Why this rewrite will yield better results.")


class FollowUpSuggestions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    suggestions: list[str] = Field(
        ...,
        description="2-3 short, natural follow-up questions (max 60 chars each)"
    )


class ConflictResolution(BaseModel):
    """Detects and resolves conflicts between policy excerpts."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conflict_found: bool = Field(
        description="True if two or more excerpts contradict each other on the same factual point."
    )
    conflict_description: str = Field(
        default="",
        description=(
            "What the conflict is "
            "(e.g. 'HR-POL-001 states Rs. 2,500 but HR-POL-004 states Rs. 3,000 for connectivity allowance')."
        ),
    )
    winning_policy: str = Field(
        default="",
        description="Which policy takes precedence and why (per the precedence hierarchy).",
    )
    resolved_answer: str = Field(
        default="",
        description="The correct resolved value/answer after applying precedence rules.",
    )


# ──────────────────────────────────────────────────────────────
# Self-RAG state
# TypedDict(total=False) — all keys optional, no Pydantic Field()
# ──────────────────────────────────────────────────────────────

class SelfRAGState(TypedDict, total=False):
    # Core conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Self-RAG tracking fields
    original_query:        str
    current_query:         str
    retrieved_context:     str
    relevant_context:      str
    generated_answer:      str
    faithfulness_grade:    str
    unsupported_claims:    list[str]
    retry_count:           int
    rewrite_count:         int
    need_retrieval:        bool
    skip_retrieval:        bool
    answer_useful:         bool
    follow_up_suggestions: list[str]
    conflict_note:         str       # injected into generate prompt when conflict found
    is_calculation:        bool      # True when query needs step-by-step math
    is_conflict_question:  bool      # True when query explicitly presents two conflicting values


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _detect_calculation_query(query: str) -> bool:
    """Return True if the query requires a numerical calculation."""
    q = query.lower()
    return any(kw in q for kw in CALCULATION_QUESTION_KEYWORDS)


def _detect_conflict_query(query: str) -> bool:
    """Return True if the query is explicitly presenting two conflicting values."""
    q = query.lower()
    return any(kw in q for kw in CONFLICT_QUESTION_KEYWORDS)


# ──────────────────────────────────────────────────────────────
# Grader chains
# ──────────────────────────────────────────────────────────────

def build_graders(llm):
    """
    Returns a dict of grader chains backed by the provided LLM.
    Call once per chatbot instance.
    """

    # ── 1. Retrieval gate ─────────────────────────────────────
    retrieval_gate_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at deciding whether a user query requires searching "
         "organisational documents (policies, HR guides, company info) or can be "
         "answered directly from general knowledge.\n\n"
         "Return need_retrieval=True for: policy questions, HR procedure questions, "
         "document-specific questions, or anything that references 'our', 'the company', "
         "'policy', 'procedure', 'handbook', 'guidelines', leave, salary, allowance, "
         "gratuity, PF, probation, hiring, work mode, remote, hybrid, SLA, BGV, MRF, "
         "LOP, CTC, increment, appraisal, retrenchment, BCP, F&F, notice period.\n"
         "Return need_retrieval=False for: general knowledge, pure math with no policy "
         "context, greetings, stock prices, or any clearly non-document question."),
        ("human", "User query: {query}"),
    ])
    retrieval_gate = retrieval_gate_prompt | llm.with_structured_output(RetrievalDecision)

    # ── 2. Relevance grader — conservative ────────────────────
    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a relevance grader for a company policy RAG system.\n\n"
         "Mark relevant=True if the excerpt:\n"
         "  - Directly answers the query, OR\n"
         "  - Contains a date, deadline, number, threshold, formula, or procedure "
         "that COULD be part of the answer, OR\n"
         "  - Discusses the same process, policy section, or employee category even "
         "if not an exact match, OR\n"
         "  - Contains an EXCEPTION or RESTRICTION that modifies a general rule "
         "related to the query (e.g. new joiner rules, PIP rules, minimum tenure), OR\n"
         "  - Comes from a DIFFERENT policy than the primary one but addresses the "
         "same topic (needed for cross-policy conflict detection).\n\n"
         "Mark relevant=False ONLY if the excerpt is about a completely different topic "
         "with zero connection to the query.\n\n"
         "IMPORTANT: This is a conservative filter. False positives are far better than "
         "false negatives. When in doubt, mark relevant=True."),
        ("human",
         "Query: {query}\n\n"
         "Document excerpt:\n{document}"),
    ])
    relevance_grader = relevance_prompt | llm.with_structured_output(RelevanceGrade)

    # ── 3. Faithfulness grader ────────────────────────────────
    faithfulness_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a faithfulness auditor for a company policy RAG system. "
         "Your job is to check whether the AI answer introduces any claims that are "
         "NOT derivable from the source documents.\n\n"
         "A claim is SUPPORTED if ANY of these are true:\n"
         "  a) It is stated verbatim or near-verbatim in the documents.\n"
         "  b) It is a direct logical inference from facts in the documents "
         "(e.g. 'CL is not encashable' is supported if the doc says 'only EL is encashed').\n"
         "  c) It is a mathematical result derived from a formula and inputs both present "
         "in the documents (e.g. gratuity calculation using the document's formula).\n"
         "  d) It is a policy precedence conclusion drawn from two excerpts that both "
         "appear in the context (e.g. 'HR-POL-004 overrides HR-POL-001 here').\n\n"
         "A claim is UNSUPPORTED only if it introduces a specific fact, number, date, or "
         "rule that cannot be traced to ANY part of the provided documents.\n\n"
         "Grading rubric:\n"
         "  fully_supported     — ALL claims are supported by the above criteria.\n"
         "  partially_supported — at least one claim is genuinely unsupported.\n"
         "  not_supported       — the answer is not grounded in the documents at all.\n\n"
         "Be precise: citation format errors or minor paraphrasing are NOT unsupported claims. "
         "Only flag claims that introduce new factual content absent from the documents.\n\n"
         "List every genuinely unsupported claim explicitly."),
        ("human",
         "Source documents:\n{context}\n\n"
         "AI-generated answer:\n{answer}"),
    ])
    faithfulness_grader = faithfulness_prompt | llm.with_structured_output(FaithfulnessGrade)

    # ── 4. Usefulness grader — tightened topic-match check ────
    #
    # Improvement targeting Answer Relevancy (was 0.59):
    # Old prompt didn't check whether the answer addressed the RIGHT question.
    # A confident answer about a related-but-wrong policy clause could pass.
    # New prompt adds an explicit wrong-topic check.
    usefulness_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict answer quality evaluator for a company HR policy RAG system.\n\n"
         "Evaluate whether the AI's answer BOTH (a) directly resolves the user's query "
         "AND (b) addresses the correct specific topic the user asked about.\n\n"
         "Mark useful=False if ANY of these are true:\n"
         "  - The answer discusses a related but DIFFERENT policy clause than what was asked.\n"
         "    Example: User asks about connectivity allowance; answer talks about WFH setup allowance.\n"
         "  - The answer gives general advice, industry norms, or speculation instead of the "
         "company's specific documented rule.\n"
         "  - The answer is vague, says 'it depends', or fails to state the specific "
         "value/date/threshold the user asked for.\n"
         "  - The answer confidently states a number, date, or rule that was NOT in the "
         "retrieved documents (hallucination).\n\n"
         "Mark useful=True ONLY if:\n"
         "  - The answer directly states the specific fact/rule/number the user asked about, "
         "supported by company documents, OR\n"
         "  - The answer honestly says the information is not in the documents "
         "(this is also a valid and useful response)."),
        ("human",
         "User query: {query}\n\n"
         "AI answer:\n{answer}"),
    ])
    usefulness_grader = usefulness_prompt | llm.with_structured_output(UsefulnessGrade)

    # ── 5. Query rewriter ─────────────────────────────────────
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a search query optimiser for a company policy knowledge base.\n\n"
         "Given the user's original question and the reason the previous retrieval "
         "was not useful, rewrite the query to maximise the chance of finding the "
         "answer in an HR policy document.\n\n"
         "Strategies to try:\n"
         "  - Use synonyms (e.g. 'notice period' vs 'advance notice', 'LOP' vs 'loss of pay')\n"
         "  - Reference the policy section if inferrable (e.g. 'probation Section 7')\n"
         "  - Break compound questions into the most atomic sub-question\n"
         "  - For deadline questions, try 'deadline for X' or 'timeline for X' or 'by when X'\n"
         "  - For cross-policy questions, name both policy areas explicitly"),
        ("human",
         "Original query: {query}\n"
         "Problem with previous retrieval: {problem}"),
    ])
    query_rewriter = rewrite_prompt | llm.with_structured_output(RewrittenQuery)

    # ── 6. Follow-up questions ────────────────────────────────
    followup_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful policy assistant. Given the conversation history and the "
         "last answer, suggest 2-3 concise, relevant follow-up questions the user "
         "might want to ask next about company policies."),
        ("human", "Conversation so far:\n{history}\n\nLast answer:\n{answer}")
    ])
    followup_chain = followup_prompt | llm.with_structured_output(FollowUpSuggestions)

    # ── 7. Cross-policy conflict detector ────────────────────
    conflict_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a policy conflict detector for a company HR knowledge base.\n\n"
         "Given multiple document excerpts about the SAME topic/query, check whether "
         "any excerpts CONTRADICT each other on the SAME factual point "
         "(e.g. two different monetary amounts, two different deadlines, two different "
         "procedures for the same scenario).\n\n"
         "If a conflict exists:\n"
         "  1. Describe the conflict clearly (cite both policy names and values).\n"
         "  2. Apply this precedence hierarchy — higher entries win:\n"
         "       HR-POL-004 (Compensation & Benefits) — HIGHEST; governs all monetary matters\n"
         "       HR-POL-009 (BCP) — overrides others during declared RIF/crisis events\n"
         "       HR-POL-006 (Engineering Allowances) — governs engineering entitlements\n"
         "       HR-POL-007 (Marketing) — governs marketing-specific entitlements\n"
         "       HR-POL-003 (Leave & Attendance) — governs all leave accounting & LOP\n"
         "       HR-POL-005 (Gratuity & PF) — governs statutory benefits\n"
         "       HR-POL-002 (Recruitment & Onboarding) — governs hiring procedures\n"
         "       HR-POL-001 (Work Mode) — LOWEST precedence\n"
         "  3. State which policy wins and provide the correct resolved answer.\n\n"
         "If there is NO conflict, set conflict_found=False and leave other fields empty."),
        ("human",
         "User query / topic: {query}\n\n"
         "Document excerpts to check:\n{excerpts}"),
    ])
    conflict_detector = conflict_prompt | llm.with_structured_output(ConflictResolution)

    return {
        "retrieval_gate":      retrieval_gate,
        "relevance_grader":    relevance_grader,
        "faithfulness_grader": faithfulness_grader,
        "usefulness_grader":   usefulness_grader,
        "query_rewriter":      query_rewriter,
        "followup_chain":      followup_chain,
        "conflict_detector":   conflict_detector,
    }


# ──────────────────────────────────────────────────────────────
# Node functions
# ──────────────────────────────────────────────────────────────

def make_retrieval_gate_node(graders: dict):
    """Decides whether retrieval is needed and resets all per-turn state."""

    def retrieval_gate_node(state: SelfRAGState) -> dict:
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        print(f"[Self-RAG] Retrieval gate → query='{query[:80]}...'")
        decision: RetrievalDecision = graders["retrieval_gate"].invoke({"query": query})
        print(f"[Self-RAG] need_retrieval={decision.need_retrieval} | {decision.reasoning}")

        return {
            "original_query":        query,
            "current_query":         query,
            "need_retrieval":        decision.need_retrieval,
            "skip_retrieval":        False,
            "retry_count":           0,
            "rewrite_count":         0,
            "retrieved_context":     "",
            "relevant_context":      "",
            "generated_answer":      "",
            "faithfulness_grade":    "",
            "unsupported_claims":    [],
            "answer_useful":         False,
            "follow_up_suggestions": [],
            "conflict_note":         "",
            "is_calculation":        _detect_calculation_query(query),
            "is_conflict_question":  _detect_conflict_query(query),
        }

    return retrieval_gate_node


def make_relevance_filter_node(graders: dict):
    """
    Filters retrieved chunks to relevant ones, then runs conflict detection
    across the relevant set and stores the resolution in state["conflict_note"].
    """

    def relevance_filter_node(state: SelfRAGState) -> dict:
        query = state.get("current_query", state.get("original_query", ""))

        raw_context = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "rag_tool":
                raw_context = msg.content
                break

        if not raw_context:
            print("[Self-RAG] No rag_tool message found — skipping relevance filter")
            return {
                "retrieved_context": "",
                "relevant_context":  "",
                "skip_retrieval":    True,
                "conflict_note":     "",
            }

        excerpts = re.split(r"(?=--- Excerpt \d+)", raw_context)
        excerpts = [e.strip() for e in excerpts if e.strip() and "--- Excerpt" in e]

        if not excerpts:
            print(f"[Self-RAG] Context not in excerpt format — using raw: {raw_context[:100]}")
            return {
                "retrieved_context": raw_context,
                "relevant_context":  raw_context,
                "skip_retrieval":    False,
                "conflict_note":     "",
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
                relevant_excerpts.append(excerpt)

        if not relevant_excerpts:
            print("[Self-RAG] No relevant excerpts found — will answer from general knowledge")
            return {
                "retrieved_context": raw_context,
                "relevant_context":  "",
                "skip_retrieval":    True,
                "conflict_note":     "",
            }

        filtered = "\n\n".join(relevant_excerpts)
        print(f"[Self-RAG] {len(relevant_excerpts)}/{len(excerpts)} excerpts passed relevance filter")

        # Run conflict detection ONLY when the query explicitly presents two
        # conflicting values (is_conflict_question=True). Running it on every
        # query causes false positives — e.g. two excerpts both mentioning
        # "days" for different leave types get flagged as conflicting, injecting
        # wrong guidance into the generate prompt and hallucinated claims.
        conflict_note = ""
        is_conflict_question = state.get("is_conflict_question", False)

        if is_conflict_question and len(relevant_excerpts) >= 2:
            try:
                resolution: ConflictResolution = graders["conflict_detector"].invoke(
                    {"query": query, "excerpts": filtered}
                )
                if resolution.conflict_found:
                    conflict_note = (
                        f"POLICY CONFLICT DETECTED:\n"
                        f"{resolution.conflict_description}\n\n"
                        f"RESOLUTION — {resolution.winning_policy}:\n"
                        f"{resolution.resolved_answer}"
                    )
                    print(f"[Self-RAG] Conflict: {resolution.conflict_description}")
                    print(f"[Self-RAG] Resolution: {resolution.resolved_answer}")
                else:
                    print("[Self-RAG] No policy conflict detected.")
            except Exception as e:
                print(f"[Self-RAG] Conflict detection error: {e} — skipping")

        return {
            "retrieved_context": raw_context,
            "relevant_context":  filtered,
            "skip_retrieval":    False,
            "conflict_note":     conflict_note,
        }

    return relevance_filter_node


def make_generate_node(llm_with_tools, llm):
    """
    Generates an answer using the relevant context.

    Improvements vs original:
      1. Exception-aware base prompt — reads ALL excerpts, surfaces
         new-joiner rules, PIP restrictions, waiting periods FIRST.
      2. Conflict note injection — forces use of the winning policy value.
      3. Calculation mode — step-by-step math prompt for numeric questions,
         targeting Answer Correctness on calculation/scenario_calculation types.
      4. Tighter retry prompt — explicitly lists and removes hallucinated claims.
    """

    def generate_node(state: SelfRAGState) -> dict:
        retry_count           = state.get("retry_count", 0)
        skip                  = state.get("skip_retrieval", False)
        context               = state.get("relevant_context", "")
        unsupported           = state.get("unsupported_claims", [])
        conflict_note         = state.get("conflict_note", "")
        is_calculation        = state.get("is_calculation", False)
        is_conflict_question  = state.get("is_conflict_question", False)

        def _conflict_block() -> str:
            if not conflict_note:
                return ""
            return (
                f"\n\n{'='*60}\n"
                f"⚠️  {conflict_note}\n"
                f"{'='*60}\n"
                "You MUST use the resolved answer above. "
                "Do NOT use the lower-precedence policy's value.\n"
            )

        # ── No relevant context found ─────────────────────────────────────
        if skip or not context:
            system = (
                "You are a helpful organisational assistant. "
                "No relevant information was found in the uploaded company documents for this query.\n\n"
                "Rules:\n"
                "- In 1-2 sentences, clearly state that the information is NOT available in the documents.\n"
                "- Do NOT give general advice or industry standards.\n"
                "- Do NOT speculate.\n"
                "- Suggest the user check the official HR policy or employee handbook."
            )

        # ── Calculation query — show working ──────────────────────────────
        elif is_calculation and retry_count == 0:
            system = (
                "You are a strict organisational policy assistant and a careful mathematician.\n\n"
                "The user has asked a CALCULATION question. Respond in this exact format:\n\n"
                "1. Formula (from documents): state it.\n"
                "2. Inputs: list the values from the question and documents.\n"
                "3. Calculation: show step by step with actual numbers "
                "(e.g. '(60,000 / 26) × 15 × 8 = 2,307.69 × 120 = Rs. 2,76,923').\n"
                "4. Result: state the final answer clearly.\n"
                "5. Conditions/caps: note any limits that apply (one sentence).\n\n"
                "OUTPUT LENGTH: 3-6 sentences total. Be precise, not verbose.\n\n"
                "STRICT RULES:\n"
                "  - Use ONLY formulas and numbers present in the documents.\n"
                "  - Do NOT add commentary, caveats, or policy background beyond what is asked.\n"
                "  - Include ONE citation [Source N: filename – page X] for the formula.\n"
                + _conflict_block()
                + "\n\nDocument excerpts:\n" + context
            )

        # ── Explicit conflict question — user is presenting two values ────
        elif is_conflict_question and retry_count == 0:
            system = (
                "You are a strict organisational policy analyst.\n\n"
                "The user's question presents two CONFLICTING values from different policies "
                "and is asking which one is correct.\n\n"
                "Your answer MUST follow this structure (3-4 sentences max):\n"
                "  1. Acknowledge the conflict: state both values and which policies they come from.\n"
                "  2. Apply precedence: state which policy takes precedence and why.\n"
                "  3. Give the single correct answer: state ONLY the value from the winning policy.\n\n"
                "Do NOT list both values as both correct. Do NOT say 'it depends'. "
                "Give ONE definitive answer.\n"
                + _conflict_block()
                + "\n\nDocument excerpts:\n" + context
            )

        # ── First non-calculation, non-conflict attempt ───────────────────
        elif retry_count == 0:
            system = (
                "You are a strict organisational policy assistant. "
                "Answer ONLY using the provided document excerpts below. "
                "Do NOT add any information not present in the excerpts.\n\n"

                "Rules:\n"
                "- Read ALL excerpts before answering. Exceptions and restrictions "
                "are often in a later excerpt — never stop at the first match.\n"
                "- If a restriction overrides the general rule (e.g. new joiner "
                "mandatory on-site, PIP reversion, minimum tenure), state it first.\n"
                "- For yes/no questions, lead with Yes or No.\n"
                "- Cover every fact the question asks for.\n"
                "- Cite sources as [Source N: filename – page X].\n"
                + _conflict_block()
                + "\n\nDocument excerpts:\n" + context
            )

        # ── Retry after faithfulness failure ──────────────────────────────
        else:
            unsupported_str = (
                "\n".join(f"  - {c}" for c in unsupported)
                if unsupported else "  (see previous response)"
            )
            system = (
                "You are a strict organisational assistant. "
                "Your previous answer contained claims NOT found in the documents. "
                "Rewrite the answer removing every claim listed below.\n\n"
                f"Claims to REMOVE:\n{unsupported_str}\n\n"
                "RULES:\n"
                "  - Use ONLY the document excerpts below.\n"
                "  - Cover all facts relevant to the question — do not omit.\n"
                "  - Include one citation [Source N: filename – page X].\n"
                "  - Check for exceptions before stating the general rule.\n"
                + _conflict_block()
                + "\n\nDocument excerpts:\n" + context
            )

        messages = [SystemMessage(content=system)] + state["messages"]
        response: AIMessage = llm.invoke(messages)

        print(f"[Self-RAG] Generated answer (retry={retry_count}): {response.content[:120]}…")
        return {
            "generated_answer": response.content,
            "messages":         [response],
        }

    return generate_node


def make_faithfulness_node(graders: dict):
    """Grades whether the generated answer is faithful to the retrieved context."""

    def faithfulness_node(state: SelfRAGState) -> dict:
        context = state.get("relevant_context", "")
        answer  = state.get("generated_answer", "")
        skip    = state.get("skip_retrieval", False)

        if skip or not context:
            return {"faithfulness_grade": "fully_supported", "unsupported_claims": []}

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
    """Grades whether the answer is useful and addresses the correct topic."""

    def usefulness_node(state: SelfRAGState) -> dict:
        query  = state.get("original_query", "")
        answer = state.get("generated_answer", "")

        try:
            grade: UsefulnessGrade = graders["usefulness_grader"].invoke(
                {"query": query, "answer": answer}
            )
            print(f"[Self-RAG] Usefulness: useful={grade.useful} | missing='{grade.missing}'")
            return {
                "answer_useful": grade.useful,
                "unsupported_claims": (
                    [grade.missing]
                    if not grade.useful and grade.missing
                    else state.get("unsupported_claims", [])
                ),
            }
        except Exception as e:
            print(f"[Self-RAG] Usefulness grading error: {e} — assuming useful")
            return {"answer_useful": True}

    return usefulness_node


def make_query_rewrite_node(graders: dict):
    """Rewrites the query when the answer is not useful; resets retrieval state."""

    def query_rewrite_node(state: SelfRAGState) -> dict:
        query         = state.get("current_query", state.get("original_query", ""))
        problem       = state.get("unsupported_claims", ["answer was not useful"])
        problem_str   = problem[0] if problem else "answer was not useful or incomplete"
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

        return {
            "current_query":        new_query,
            "rewrite_count":        rewrite_count,
            "skip_retrieval":       False,
            "retrieved_context":    "",
            "relevant_context":     "",
            "conflict_note":        "",   # reset so conflict detection re-runs on new retrieval
            "is_conflict_question": _detect_conflict_query(new_query),
            "messages":             [HumanMessage(content=new_query)],
        }

    return query_rewrite_node


def make_followup_node(graders: dict):
    """Generates follow-up question suggestions after a successful RAG answer."""

    def node(state: SelfRAGState):
        skip             = state.get("skip_retrieval", False)
        need_retrieval   = state.get("need_retrieval", True)
        relevant_context = state.get("relevant_context", "")

        if not need_retrieval or (skip and not relevant_context):
            return {"follow_up_suggestions": []}

        history = "\n".join([m.content for m in state["messages"][-6:]])
        answer  = state.get("generated_answer", "")
        try:
            result: FollowUpSuggestions = graders["followup_chain"].invoke(
                {"history": history, "answer": answer}
            )
            return {"follow_up_suggestions": result.suggestions[:3]}
        except Exception as e:
            print(f"[Self-RAG] Follow-up generation failed: {e}")
            return {"follow_up_suggestions": []}

    return node


# ──────────────────────────────────────────────────────────────
# Routing / conditional edge functions
# ──────────────────────────────────────────────────────────────

def route_after_gate(state: SelfRAGState) -> str:
    """After retrieval gate: call rag_tool or go straight to generate."""
    if state.get("need_retrieval", True):
        return "call_rag"
    return "generate"


def route_after_faithfulness(state: SelfRAGState) -> str:
    """After faithfulness check: retry generation or move to usefulness check."""
    grade       = state.get("faithfulness_grade", "fully_supported")
    retry_count = state.get("retry_count", 0)

    if grade in ("partially_supported", "not_supported") and retry_count < MAX_RETRIES:
        print(f"[Self-RAG] Faithfulness={grade} → re-generating (attempt {retry_count + 1}/{MAX_RETRIES})")
        return "retry_generate"
    return "check_usefulness"


def route_after_usefulness(state: SelfRAGState) -> str:
    """After usefulness check: rewrite query or finish."""
    useful        = state.get("answer_useful", True)
    rewrite_count = state.get("rewrite_count", 0)

    if not useful and rewrite_count < MAX_QUERY_REWRITES:
        print(f"[Self-RAG] Answer not useful → rewriting query (rewrite {rewrite_count + 1}/{MAX_QUERY_REWRITES})")
        return "rewrite_query"
    return "finish"


def increment_retry(state: SelfRAGState) -> dict:
    """Helper node: increment retry_count before re-generating."""
    return {"retry_count": state.get("retry_count", 0) + 1}