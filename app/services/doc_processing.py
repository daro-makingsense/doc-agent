"""Document ingestion utilities powered by Docling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from docling.document_converter import DocumentConverter
from docling.document_converter.document import ConvertedDocument

from ..config import get_settings


@dataclass
class DocumentChunk:
    """Represents a text chunk and its position."""

    index: int
    text: str
    metadata: dict


@dataclass
class ProcessedDocument:
    """Container for the parsed document."""

    source_path: Path
    content_type: str
    page_count: int | None
    text: str
    chunks: Sequence[DocumentChunk]


class DoclingDocumentProcessor:
    """Loads documents with Docling and prepares them for embedding."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.converter = DocumentConverter()

    def load(self, file_path: str | Path) -> ProcessedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)

        result = self.converter.convert(path)
        document: ConvertedDocument = result.document
        text = document.export_to_markdown()
        page_count = self._extract_page_count(document)
        chunks = list(
            self._chunk_text(
                text,
                max_tokens=self.settings.max_chunk_size,
                overlap=self.settings.chunk_overlap,
                metadata={"source": str(path)},
            )
        )
        return ProcessedDocument(
            source_path=path,
            content_type=path.suffix.lower().lstrip("."),
            page_count=page_count,
            text=text,
            chunks=chunks,
        )

    def _extract_page_count(self, document: ConvertedDocument) -> int | None:
        try:
            return document.info.page_count  # type: ignore[attr-defined]
        except AttributeError:
            return None

    def _chunk_text(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap: int,
        metadata: dict | None = None,
    ) -> Iterable[DocumentChunk]:
        if not text:
            return

        metadata = metadata or {}
        start = 0
        index = 0
        length = len(text)
        while start < length:
            end = min(start + max_tokens, length)
            chunk_text = text[start:end]
            yield DocumentChunk(index=index, text=chunk_text, metadata={**metadata, "offset": start})
            index += 1
            if end == length:
                break
            start = max(0, end - overlap)


__all__ = [
    "DocumentChunk",
    "ProcessedDocument",
    "DoclingDocumentProcessor",
]
