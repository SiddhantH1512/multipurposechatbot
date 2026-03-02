# backend/models.py
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama
from src.config import Config

def ChatGrokModel():
    return ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model_name=Config.MODEL_NAME,
        temperature=0
    )

def ChatGeminiModel():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",          # or gemini-1.5-flash etc
        temperature=0,
        google_api_key=Config.GEMINI_API_KEY,
        max_retries=3
    )

# def ChatOllamaModel():
#     return ChatOllama(
#         model="mistral",
#         temperature=0,
#         num_ctx=4096,
#         num_predict=256,
#     )