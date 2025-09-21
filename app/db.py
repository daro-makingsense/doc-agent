"""Database models and session management."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base, relationship, scoped_session, sessionmaker

from .config import get_settings


Base = declarative_base()


class Document(Base):
    """Represents a processed financial document."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    page_count = Column(Integer, nullable=True)
    source_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    metadata_items = relationship(
        "DocumentMetadata",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentMetadata(Base):
    """Key-value metadata extracted from the document."""

    __tablename__ = "document_metadata"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    key = Column(String, nullable=False)
    value = Column(Text, nullable=False)

    document = relationship("Document", back_populates="metadata_items")


class DocumentChunk(Base):
    """Individual text chunk along with its embedding."""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(ARRAY(Float), nullable=True)
    metadata = Column(JSON, nullable=True)

    document = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    """A user chat session."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    user_id = Column(String, nullable=False)
    topic = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """A message exchanged in a chat session."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    references = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("ChatSession", back_populates="messages")


settings = get_settings()
engine = create_engine(settings.postgres_dsn, echo=settings.enable_debug_logging)
SessionLocal = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "Base",
    "Document",
    "DocumentMetadata",
    "DocumentChunk",
    "ChatSession",
    "ChatMessage",
    "engine",
    "SessionLocal",
    "session_scope",
]
