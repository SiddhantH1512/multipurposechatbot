import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
    MODEL_NAME = "llama-3.3-70b-versatile"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", None)
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", None)
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", None)
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", None)
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", None)