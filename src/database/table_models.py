from enum import Enum as PyEnum
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, text
import sqlalchemy as sa
from sqlalchemy.types import Enum as SAEnum
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class UserRole(str, PyEnum):   # ← inherit from str so .value is the string we want in DB
    HR = "HR"
    EMPLOYEE = "EMPLOYEE"
    EXECUTIVE = "EXECUTIVE"
    INTERN = "INTERN"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    # Correct way:
    role = Column(
        SAEnum(UserRole, name="userrole"),
        nullable=False,
        default=UserRole.EMPLOYEE
    )
    
    department = Column(String, nullable=False, default="General")
    designation = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ThreadMetadata(Base):
    __tablename__ = "thread_metadata"

    thread_id = Column(Text, primary_key=True, nullable=False)
    filename = Column(Text, nullable=True)
    documents = Column(Integer, server_default=text("0"), nullable=True)
    chunks = Column(Integer, server_default=text("0"), nullable=True)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=True)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP"), nullable=True)
    user_id = Column(Integer, nullable=True)
    department = Column(String, nullable=False, server_default='General')
    is_global = Column(Boolean, nullable=False, server_default=text('true'))

    __table_args__ = (
        sa.Index(
            "idx_thread_metadata_created_at",
            "created_at",
            postgresql_using="btree",
            postgresql_ops={"created_at": "DESC"}
        ),
    )