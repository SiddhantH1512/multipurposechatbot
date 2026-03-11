from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from src.config import Config
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

def ChatGrokModel():
    return ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model_name=Config.MODEL_NAME,
        temperature=0
    )

def ChatGeminiModel():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=Config.GEMINI_API_KEY,
        max_retries=3
    )


def ChatOpenAIModel():
    return ChatOpenAI(
        api_key=Config.OPENAI_API_KEY,
        model_name="gpt-4o-mini",
        temperature=0.4,
        max_tokens=2000,
        timeout=60,
        max_retries=3
    )




Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)