from src.database.table_models import User
from src.auth.jwt import get_password_hash
from src.database.engine import sync_engine
from sqlalchemy import insert

with sync_engine.connect() as conn:
    conn.execute(
        insert(User),
        {
            "email": "employee@company.com",
            "hashed_password": get_password_hash("Test@123456"),
            "role": "EMPLOYEE",
            "department": "Engineering",
            "designation": "Software Engineer",
            "is_active": True
        }
    )
    conn.commit()