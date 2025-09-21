"""Pinecone vector store integration."""
from __future__ import annotations

from typing import List, Sequence

from pinecone import Pinecone, ServerlessSpec

from ..config import get_settings


class PineconeVectorStore:
    """Utility wrapper around Pinecone's Python SDK."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.has_pinecone:
            raise RuntimeError("Pinecone configuration missing; set PINECONE_API_KEY and PINECONE_INDEX")

        self.client = Pinecone(api_key=self.settings.pinecone_api_key)
        self.index_name = self.settings.pinecone_index
        self.environment = self.settings.pinecone_environment or "us-east-1-aws"

    def _ensure_index(self, dimension: int) -> None:
        indexes = {idx["name"] for idx in self.client.list_indexes().indexes}
        if self.index_name not in indexes:
            self.client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.environment),
            )

    def upsert_document(
        self,
        *,
        document_id: int,
        chunks: Sequence[dict],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if not chunks:
            return

        self._ensure_index(len(embeddings[0]))
        index = self.client.Index(self.index_name)
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(
                {
                    "id": f"doc-{document_id}-chunk-{chunk['chunk_index']}",
                    "values": list(embedding),
                    "metadata": {
                        "document_id": document_id,
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        **chunk.get("metadata", {}),
                    },
                }
            )
        index.upsert(vectors=vectors)

    def similarity_search(self, embedding: Sequence[float], *, top_k: int = 5) -> List[dict]:
        self._ensure_index(len(embedding))
        index = self.client.Index(self.index_name)
        response = index.query(vector=list(embedding), top_k=top_k, include_metadata=True)
        return response.matches if hasattr(response, "matches") else []


__all__ = ["PineconeVectorStore"]
