# PolicyIQ - Intelligent Organizational Policy Assistant

**A production-grade AI-powered chatbot for HR policies, leave policies, and organizational documents.**

Built with LangGraph, Self-RAG, PostgreSQL + pgvector, and enterprise-grade security features. Designed to demonstrate senior-level AI engineering skills.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6F00)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1)
![Docker](https://img.shields.io/badge/Docker-2496ED)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B)

---

## ✨ Key Features

- **Advanced Self-RAG System**
  - Intelligent routing: Decides whether document retrieval is needed
  - Relevance grading of retrieved chunks
  - Support grading (Fully Supported / Partially Supported / Not Supported)
  - Automatic retry logic with max_retries to eliminate hallucinations
  - Usefulness scoring + query rewriting for better performance

- **Enterprise Security & Access Control**
  - Manual JWT authentication with access + refresh token rotation
  - Role-Based Access Control (RBAC) — Only HR users can upload documents
  - Row-Level Security (RLS) in PostgreSQL for department-based document isolation
  - Hierarchical rate limiting based on user designation (Executive → Intern)
  - Prompt injection detection + PII redaction + input sanitization

- **Organizational RAG Capabilities**
  - Global document access across all threads
  - Secure department-scoped visibility
  - Hybrid retrieval (dense vector + BM25 + FlashRank reranker)
  - Support for PDFs and research papers with rich metadata

- **Observability & Evaluation**
  - Full integration with LangSmith for tracing
  - RAGAS framework for systematic evaluation of retrieval and generation quality

---

## 🛠️ Tech Stack

- **Backend**: FastAPI + LangGraph + LangChain
- **Vector Database**: PostgreSQL + pgvector
- **Frontend**: Streamlit
- **Authentication**: Custom JWT (manual implementation)
- **Security**: RBAC + RLS + Rate Limiting + Sanitization
- **Evaluation**: RAGAS
- **Containerization**: Docker + docker-compose

---

## 📁 Project Structure

```bash
src/
├── api/                    # FastAPI routes, middleware, rate limiting
├── auth/                   # JWT utilities, OAuth2, dependencies
├── backend/                # LangGraph graph, Self-RAG nodes, tools, rag_tool
├── database/               # SQLAlchemy models, Alembic migrations, RLS policies
├── security/               # Prompt injection guard, PII redaction, sanitization
├── config.py               # Centralized configuration
├── main.py                 # Streamlit frontend
└── requirements.txt
```

---

## 🚀 Key Technical Highlights

- Implemented Self-RAG with multiple grader nodes and conditional edges in LangGraph  
- Used `cmetadata` JSONB field + PostgreSQL RLS for secure, scalable document filtering  
- Built a clean modular monolith architecture  
- Hierarchical rate limiting based on user designation/role  
- Full grounding of responses with mandatory citations  
- Production-ready Docker setup with multi-stage builds  

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd multipurposechatbot
```

### 2. Environment Setup
```bash
# Copy the example environment file
cp .env.example .env
```

Edit `.env` and add your keys:
- OPENAI_API_KEY  
- SECRET_KEY (generate using `openssl rand -hex 32`)  
- LANGCHAIN_API_KEY (optional but recommended)  
- Database credentials  

---

### 3. Run with Docker Compose (Recommended)
```bash
docker-compose up --build
```

---

### 4. Access the application

- **Streamlit UI**: http://localhost:8501  
- **FastAPI Docs**: http://localhost:8000/docs  

---

## 🔑 Environment Variables

See `.env.example` for all required and optional variables.

Key variables include:

- OPENAI_API_KEY  
- SECRET_KEY  
- POSTGRES_CONNECTION  
- REDIS_URL  
- LANGCHAIN_API_KEY  

---

## 🔮 Future Enhancements

- CI/CD pipeline with GitHub Actions  
- Cloud deployment (Render / Railway)  
- Admin dashboard for document management  
- User feedback collection and fine-tuning loop  
- Multi-modal document support  