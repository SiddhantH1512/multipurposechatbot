from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, text
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP"))


class ThreadMetadata(Base):
    __tablename__ = "thread_metadata"

    thread_id = Column(
        Text,
        primary_key=True,
        nullable=False 
    )

    filename = Column(
        Text,
        nullable=True
    )

    documents = Column(
        Integer,
        server_default=text("0"),
        nullable=True
    )

    chunks = Column(
        Integer,
        server_default=text("0"),
        nullable=True
    )

    created_at = Column(
        DateTime,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=True
    )

    updated_at = Column(
        DateTime,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
        nullable=True
    )

    __table_args__ = (
        sa.Index(
            "idx_thread_metadata_created_at",
            "created_at",
            postgresql_using="btree",
            postgresql_ops={"created_at": "DESC"}
        ),
    )