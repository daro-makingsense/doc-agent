"""High level document upload orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from ..db import Document, DocumentChunk, DocumentMetadata, session_scope
from .doc_processing import DoclingDocumentProcessor, ProcessedDocument
from .embedding_service import EmbeddingService
from .metadata_extractor import ExtractedMetadata, MetadataExtractor
from .vector_store import PineconeVectorStore


class UploadService:
    """Coordinates document parsing, embedding, and persistence."""

    def __init__(self) -> None:
        self.processor = DoclingDocumentProcessor()
        self.metadata_extractor = MetadataExtractor()
        self.embedding_service = EmbeddingService()
        self.vector_store = PineconeVectorStore()

    def ingest(
        self,
        file_path: str | Path,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        processed: ProcessedDocument | None = None,
        metadata: ExtractedMetadata | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> Dict[str, object]:
        processed = processed or self.processor.load(file_path)
        metadata = metadata or self.metadata_extractor.extract(processed)
        chunk_texts = [chunk.text for chunk in processed.chunks]
        embeddings = list(embeddings or self.embedding_service.embed_texts(chunk_texts))

        with session_scope() as session:
            document = Document(
                filename=filename or Path(file_path).name,
                content_type=content_type or processed.content_type,
                page_count=metadata.page_count,
                source_path=str(processed.source_path),
            )
            session.add(document)
            session.flush()  # assign id
            document_id = document.id

            metadata_items = self._metadata_to_rows(metadata)
            for item in metadata_items:
                session.add(
                    DocumentMetadata(
                        document_id=document_id,
                        key=item["key"],
                        value=item["value"],
                    )
                )

            chunk_payloads = []
            for chunk, embedding in zip(processed.chunks, embeddings):
                session.add(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk.index,
                        text=chunk.text,
                        embedding=list(embedding),
                        metadata=chunk.metadata,
                    )
                )
                chunk_payloads.append(
                    {
                        "chunk_index": chunk.index,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    }
                )

        self.vector_store.upsert_document(
            document_id=document_id,
            chunks=chunk_payloads,
            embeddings=embeddings,
        )

        return {
            "document_id": document_id,
            "page_count": metadata.page_count,
            "word_count": metadata.word_count,
            "financial_facts": metadata.financial_facts,
        }

    def _metadata_to_rows(self, metadata: ExtractedMetadata) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        if metadata.page_count is not None:
            rows.append({"key": "page_count", "value": str(metadata.page_count)})
        rows.append({"key": "word_count", "value": str(metadata.word_count)})
        if metadata.financial_facts:
            rows.append({"key": "financial_facts", "value": json.dumps(metadata.financial_facts)})
        if metadata.investment_highlights:
            rows.append({"key": "investment_highlights", "value": json.dumps(metadata.investment_highlights)})
        return rows


__all__ = ["UploadService"]
