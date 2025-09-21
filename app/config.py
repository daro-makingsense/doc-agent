"""Application configuration utilities."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "cim-documents")
    postgres_dsn: str = os.getenv(
        "POSTGRES_DSN",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/doc_agent",
    )
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    research_model: str = os.getenv("RESEARCH_MODEL", "gpt-4o-mini")
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "2000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    enable_debug_logging: bool = os.getenv("DEBUG", "false").lower() == "true"

    @property
    def has_pinecone(self) -> bool:
        return bool(self.pinecone_api_key and self.pinecone_index)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    return Settings()


__all__ = ["Settings", "get_settings"]
