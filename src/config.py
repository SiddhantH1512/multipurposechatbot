import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ── LLM / External API Keys ──
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", None)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
    MODEL_NAME = "llama-3.1-70b-versatile"

    # ── LangChain / LangSmith ──
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", None)
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", None)
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", None)
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", None)

    # ── External Services ──
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", None)

    # ── Database ──
    POSTGRES_CONNECTION = os.getenv("POSTGRES_CONNECTION")
    POSTGRES_CONNINFO = (
        "host=postgres "
        "port=5432 "
        "dbname=chatbot_db "
        "user=postgres "
        "password=Siddhant1512!"
    )

    # ── JWT / Auth ──
    SECRET_KEY = os.getenv("SECRET_KEY", None)
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    # ── Rate Limiting ──   ← ADD THIS SECTION
    RATE_LIMIT_WINDOW_SECONDS = 60   # 60-second sliding window

    # Per-minute rate limits by designation (job title) — case-insensitive lookup
    RATE_LIMITS_BY_DESIGNATION = {
        # Executive tier
        "ceo": 50,
        "cto": 50,
        "cfo": 50,
        "coo": 50,
        "vp": 50,
        "vice president": 50,
        "director": 50,
        "executive": 50,
        # Manager / HR tier
        "manager": 30,
        "senior manager": 30,
        "hr": 30,
        "hr manager": 30,
        # Senior engineer tier
        "senior engineer": 20,
        "senior developer": 20,
        "staff engineer": 20,
        "principal engineer": 20,
        "lead engineer": 20,
        "tech lead": 20,
        # Regular employee tier
        "employee": 15,
        "engineer": 15,
        "developer": 15,
        "junior engineer": 15,
        "junior developer": 15,
        "analyst": 15,
        "associate": 15,
        # Intern tier
        "intern": 5,
        "trainee": 5,
    }

    # Fallback limits by role (when designation is missing)
    RATE_LIMITS_BY_ROLE = {
        "INTERN": 5,
        "EMPLOYEE": 15,
        "HR": 30,
        "EXECUTIVE": 50,
    }

    RATE_LIMIT_DEFAULT = 10   # fallback if nothing matches

    # ── File Upload ──
    MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024   # 50 MB

    # ── NEW: Redis (for caching + rate limiting) ─────────────────────
    REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")