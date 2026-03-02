import sqlite3

def initialize_db(db_path="chatbot.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Add table for thread metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thread_metadata (
            thread_id TEXT PRIMARY KEY,
            filename TEXT,
            documents INTEGER,
            chunks INTEGER,
            index_path TEXT  -- Path to FAISS index on disk
        )
    """)
    conn.commit()
    conn.close()