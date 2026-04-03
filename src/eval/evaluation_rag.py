# src/eval/evaluate_rag.py
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# ==================== FORCE LOCALHOST FOR EVALUATION ====================
os.environ["POSTGRES_CONNECTION"] = "postgresql+psycopg://postgres:Siddhant1512!@localhost:5433/chatbot_db"
print("🔧 EVAL MODE: Using localhost Postgres connection")

from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings

from src.eval.evaluation_dataset import load_golden_dataset
from src.eval.simple_rag_eval_graph import get_minimal_eval_chatbot

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset


async def run_rag_pipeline(question: str, question_id: str) -> dict:
    print(f"\n🔍 [{question_id}] Processing: {question[:75]}...")

    chatbot = get_minimal_eval_chatbot()

    config = {
        "configurable": {
            "thread_id": f"eval-{question_id}",
            "user_department": "ALL",
            "user_id": 999,
        }
    }

    try:
        print(f"   → Sending to Minimal RAG Graph...")
        result = await chatbot.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        final_message = result["messages"][-1]
        answer = final_message.content if hasattr(final_message, "content") else str(final_message)

        contexts = []
        for msg in result.get("messages", []):
            if hasattr(msg, "name") and msg.name == "rag_tool":
                contexts.append(getattr(msg, "content", ""))
                print(f"   → Retrieved {len(contexts)} context chunk(s)")

        print(f"   → Answer generated ({len(answer)} chars)")
        return {"question": question, "answer": answer, "contexts": contexts}

    except Exception as e:
        print(f"   ❌ Error: {type(e).__name__}: {e}")
        return {"question": question, "answer": f"ERROR: {str(e)}", "contexts": []}


async def evaluate_pipeline():
    print("=" * 90)
    print("🚀 STARTING RAGAS EVALUATION (Minimal RAG Mode)")
    print("=" * 90)

    dataset = load_golden_dataset()
    print(f"📚 Loaded {len(dataset)} golden examples\n")

    questions = []
    ground_truths = []
    answers = []
    contexts_list = []

    for i, item in enumerate(dataset, 1):
        print(f"[{i:2d}/{len(dataset)}] Evaluating {item['id']} | {item.get('difficulty', 'N/A')}")

        result = await run_rag_pipeline(item["question"], item["id"])

        questions.append(item["question"])
        ground_truths.append(item["ground_truth"])
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])

        await asyncio.sleep(0.5)

    print("\n✅ All questions processed. Running RAGAS metrics...\n")

    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

        # === FIXED EVALUATION CALL + SAFE METRICS HANDLING ===
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]

    results = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    # Safe metrics extraction (avoid pandas mean error)
    df = results.to_pandas()
    numeric_df = df.select_dtypes(include=['number'])  # Only numeric columns
    avg_metrics = numeric_df.mean().to_dict()

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("eval_results") / f"ragas_report_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": timestamp,
        "total_questions": len(dataset),
        "metrics": avg_metrics,
        "detailed_results": df.to_dict(orient="records")
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Evaluation finished! Report saved to: {output_file}")
    print("\n📊 Average Scores:")
    for k, v in avg_metrics.items():
        print(f"   • {k.replace('_', ' ').title()}: {v:.4f}")


if __name__ == "__main__":
    asyncio.run(evaluate_pipeline())