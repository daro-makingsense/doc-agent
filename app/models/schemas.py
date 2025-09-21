"""Pydantic models for the public API."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class UploadRequest(BaseModel):
    file_path: str = Field(..., description="Absolute or relative path to the document")
    topic: Optional[str] = Field(
        default=None,
        description="Optional topic to feed into the deep research agent",
    )


class UploadResponse(BaseModel):
    upload_summary: Dict[str, object]
    research: Optional[Dict[str, object]] = None


class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    message: str
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    references: List[Dict[str, object]]


__all__ = [
    "UploadRequest",
    "UploadResponse",
    "ChatRequest",
    "ChatResponse",
]
