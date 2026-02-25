import uuid
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from config import Config
from langchain_ollama import ChatOllama


def ChatGrokModel():
    model = ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model_name=Config.MODEL_NAME,
        temperature=0
    )
    return model


def ChatGeminiModel():
    model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=Config.GEMINI_API_KEY,
            max_retries=3
        )
    return model

def ChatOllamaModel():
    return ChatOllama(
    model="mistral",
    temperature=0,
    num_ctx=4096,      # safer for 8GB
    num_predict=256,   # prevents long generations
)