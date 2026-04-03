import os
from dotenv import load_dotenv

load_dotenv()

class EvalConfig:
    """
    Configuration specifically for RAGAS evaluation.
    Uses localhost because the script runs outside Docker.
    """

    # ── Database (Pointing to localhost) ──
    POSTGRES_CONNECTION = (
        "postgresql+psycopg://postgres:Siddhant1512!@localhost:5433/chatbot_db"
    )

    # ── LLM / Models ──
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL_NAME = "gpt-4o-mini"          # or your preferred model

    # ── Embeddings ──
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    # ── Evaluation Settings ──
    MAX_RETRIES = 3
    EVAL_BATCH_SIZE = 5                 # Process 5 questions at a time

    # ── Debug / Logging ──
    DEBUG = True

    @classmethod
    def print_config(cls):
        """Helpful debug print for evaluation environment."""
        print("🔧 Evaluation Config Loaded:")
        print(f"   • Postgres: localhost:5433")
        print(f"   • Model: {cls.MODEL_NAME}")
        print(f"   • Embedding: {cls.EMBEDDING_MODEL}")
        print(f"   • Max Retries: {cls.MAX_RETRIES}")