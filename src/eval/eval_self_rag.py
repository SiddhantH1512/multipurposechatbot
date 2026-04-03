"""
RAGAS Evaluation using the FULL Self-RAG pipeline (mirrors production).

Run locally with:
    python -m src.eval.eval_self_rag

Requires:
    - Local Postgres on port 5433 with chatbot_db
    - Local Redis on port 6379
    - Env vars: OPENAI_API_KEY, SECRET_KEY, etc.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

# ── Force localhost connections BEFORE any src imports ──────────────────────
os.environ["POSTGRES_CONNECTION"] = "postgresql+asyncpg://postgres:Siddhant1512!@localhost:5433/chatbot_db"
os.environ["REDIS_URL"] = "redis://localhost:6379"
print("🔧 EVAL MODE: Pointing to localhost Postgres (5433) + Redis (6379)")

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings

from src.eval.evaluation_dataset import load_golden_dataset
from src.eval.eval_graph_self_rag import get_self_rag_eval_chatbot

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset


# ── Collect per-question metadata for the dashboard ─────────────────────────

async def run_self_rag_pipeline(question: str, question_id: str, difficulty: str) -> dict:
    """Run one question through the full Self-RAG graph and collect all signals."""
    print(f"\n🔍 [{question_id}] ({difficulty}) {question[:80]}...")

    chatbot = get_self_rag_eval_chatbot()

    config = {
        "configurable": {
            "thread_id": f"eval-selfrag-{question_id}-{int(time.time())}",
            "user_department": "ALL",
            "user_id": 999,
        }
    }

    t0 = time.perf_counter()
    try:
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "original_query": question,
            "current_query": question,
            "retrieved_context": "",
            "relevant_context": "",
            "generated_answer": "",
            "faithfulness_grade": "",
            "unsupported_claims": [],
            "retry_count": 0,
            "rewrite_count": 0,
            "need_retrieval": True,
            "skip_retrieval": False,
            "answer_useful": False,
            "follow_up_suggestions": [],
            "conflict_note": "",
        }

        # result = await chatbot.ainvoke(initial_state, config=config)
        result = await asyncio.to_thread(chatbot.invoke, initial_state, config)

        latency = round(time.perf_counter() - t0, 2)

        # ── Extract the final AI answer ──────────────────────────────────────
        answer = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                answer = msg.content.strip()
                break

        # ── Extract retrieved contexts from ToolMessages ─────────────────────
        contexts = []
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "rag_tool":
                content = getattr(msg, "content", "")
                if content:
                    contexts.append(content)

        # ── Self-RAG metadata ────────────────────────────────────────────────
        faithfulness_grade = result.get("faithfulness_grade", "unknown")
        retry_count        = result.get("retry_count", 0)
        rewrite_count      = result.get("rewrite_count", 0)
        need_retrieval     = result.get("need_retrieval", True)
        skip_retrieval     = result.get("skip_retrieval", False)
        answer_useful      = result.get("answer_useful", False)
        follow_ups         = result.get("follow_up_suggestions", [])
        conflict_note      = result.get("conflict_note", "")

        print(f"   ✅ Answer: {len(answer)} chars | faith={faithfulness_grade} "
              f"| retry={retry_count} | rewrite={rewrite_count} | latency={latency}s")

        return {
            "question":          question,
            "answer":            answer if answer else f"ERROR: empty response",
            "contexts":          contexts,
            "latency":           latency,
            "faithfulness_grade": faithfulness_grade,
            "retry_count":       retry_count,
            "rewrite_count":     rewrite_count,
            "need_retrieval":    need_retrieval,
            "skip_retrieval":    skip_retrieval,
            "answer_useful":     answer_useful,
            "follow_ups":        follow_ups,
            "conflict_note":     conflict_note,
            "error":             None,
        }

    except Exception as e:
        latency = round(time.perf_counter() - t0, 2)
        print(f"   ❌ Error: {type(e).__name__}: {e}")
        return {
            "question":          question,
            "answer":            f"ERROR: {type(e).__name__}: {str(e)}",
            "contexts":          [],
            "latency":           latency,
            "faithfulness_grade": "error",
            "retry_count":       0,
            "rewrite_count":     0,
            "need_retrieval":    True,
            "skip_retrieval":    False,
            "answer_useful":     False,
            "follow_ups":        [],
            "conflict_note":     "",
            "error":             str(e),
        }


async def evaluate_pipeline():
    print("=" * 90)
    print("🚀  RAGAS EVALUATION — Full Self-RAG Pipeline")
    print("=" * 90)

    dataset = load_golden_dataset()
    print(f"📚 Loaded {len(dataset)} golden examples\n")

    questions     = []
    ground_truths = []
    answers       = []
    contexts_list = []
    per_item_meta = []   # rich metadata for the dashboard

    for i, item in enumerate(dataset, 1):
        print(f"[{i:2d}/{len(dataset)}] {item['id']} | {item.get('difficulty', 'N/A')} | "
              f"{item.get('question_type', 'N/A')}")

        result = await run_self_rag_pipeline(
            question=item["question"],
            question_id=item["id"],
            difficulty=item.get("difficulty", "N/A"),
        )

        questions.append(item["question"])
        ground_truths.append(item["ground_truth"])
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])

        per_item_meta.append({
            "id":                item["id"],
            "policy":            item.get("policy", ""),
            "difficulty":        item.get("difficulty", "N/A"),
            "question_type":     item.get("question_type", "N/A"),
            "question":          item["question"],
            "ground_truth":      item["ground_truth"],
            "answer":            result["answer"],
            "contexts":          result["contexts"],
            "latency":           result["latency"],
            "faithfulness_grade": result["faithfulness_grade"],
            "retry_count":       result["retry_count"],
            "rewrite_count":     result["rewrite_count"],
            "need_retrieval":    result["need_retrieval"],
            "skip_retrieval":    result["skip_retrieval"],
            "answer_useful":     result["answer_useful"],
            "follow_ups":        result["follow_ups"],
            "conflict_note":     result["conflict_note"],
            "error":             result["error"],
        })

        await asyncio.sleep(0.3)   # gentle rate-limiting between questions

    print("\n✅ All questions processed. Running RAGAS metrics...\n")

    # ── RAGAS evaluation ─────────────────────────────────────────────────────
    ragas_dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts_list,
        "ground_truth": ground_truths,
    })

    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    ragas_results = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    df         = ragas_results.to_pandas()
    numeric_df = df.select_dtypes(include=["number"])
    avg_metrics = numeric_df.mean().to_dict()

    # ── Merge RAGAS per-row scores back into per_item_meta ───────────────────
    score_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]

    df = ragas_results.to_pandas()

    for idx, row in df.iterrows():
        if idx < len(per_item_meta):
            for col in score_cols:
                if col in row:
                    val = row[col]
                    try:
                        # Only convert to float if it's actually a number (and not NaN)
                        per_item_meta[idx][f"ragas_{col}"] = float(val) if val == val else None
                    except (ValueError, TypeError):
                        # If it's a string (like the question), set to None or skip
                        per_item_meta[idx][f"ragas_{col}"] = None

    # ── Save JSON report ─────────────────────────────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir  = Path("eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"self_rag_report_{timestamp}.json"

    report = {
        "timestamp":        timestamp,
        "pipeline":         "Self-RAG",
        "total_questions":  len(dataset),
        "avg_metrics":      avg_metrics,
        "per_item":         per_item_meta,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅ Report saved → {output_file}")
    print("\n📊 Average Scores:")
    for k, v in avg_metrics.items():
        emoji = "🟢" if v >= 0.7 else ("🟡" if v >= 0.4 else "🔴")
        print(f"   {emoji} {k.replace('_', ' ').title():30s}: {v:.4f}")

    # ── Auto-generate the HTML dashboard ─────────────────────────────────────
    from src.eval.generate_dashboard import generate_dashboard
    dashboard_file = generate_dashboard(report, output_dir, timestamp)
    print(f"\n🌐 Dashboard → {dashboard_file}")

    return str(output_file)


if __name__ == "__main__":
    asyncio.run(evaluate_pipeline())