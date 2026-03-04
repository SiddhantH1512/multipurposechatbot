# run_reset.py  (place in project root and run with python run_reset.py)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings, collection_name="ai_ml_documents")

print(db.get(where={"thread_id": "some-fake-id"}, include=["documents"]))