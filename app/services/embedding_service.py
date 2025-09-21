"""Embedding utilities backed by OpenAI."""
from __future__ import annotations

from typing import List, Sequence

from openai import OpenAI

from ..config import get_settings


class EmbeddingService:
    """Thin wrapper around OpenAI's embedding API."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.has_openai:
            raise RuntimeError("OPENAI_API_KEY is required for embedding generation")
        self.client = OpenAI(api_key=self.settings.openai_api_key)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]


__all__ = ["EmbeddingService"]
