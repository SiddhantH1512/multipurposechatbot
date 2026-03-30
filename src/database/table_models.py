from enum import Enum as PyEnum
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean, DateTime, Text, text
import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.types import Enum as SAEnum
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()


class UserRole(str, PyEnum):
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
    tenant_id = Column(String, nullable=False, default="default", index=True)
    designation = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc).isoformat())


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
    document_id = Column(String, nullable=True, index=True)
    tenant_id = Column(String, nullable=False, default="default", index=True)

    __table_args__ = (
        sa.Index(
            "idx_thread_metadata_created_at",
            "created_at",
            postgresql_using="btree",
            postgresql_ops={"created_at": "DESC"}
        ),
    )


class AuditLog(Base):
    """Audit log for tracking user actions (ingest, chat, etc.)."""
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    action = Column(String, nullable=False)
    resource = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(
    DateTime(timezone=True),
    server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    # Optional relationship (for easier querying later)
    user = relationship("User", backref="audit_logs")