import uuid
from langchain_groq import ChatGroq
import streamlit as st
from config import Config


def ChatGrokModel():
    model = ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model_name=Config.MODEL_NAME,
        temperature=0
    )
    return model


