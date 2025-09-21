"""Extract structured metadata from processed documents."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable

from .doc_processing import ProcessedDocument


FINANCIAL_FACT_PATTERNS = {
    "revenue": re.compile(r"revenue[^0-9$%]*([$€£]?\s?[0-9.,]+\s?(?:million|billion|m|bn)?)", re.IGNORECASE),
    "ebitda": re.compile(r"ebitda[^0-9$%]*([$€£]?\s?[0-9.,]+\s?(?:million|billion|m|bn)?)", re.IGNORECASE),
    "growth_rate": re.compile(r"growth[^0-9%]*([0-9.]+%)", re.IGNORECASE),
    "investment_required": re.compile(r"investment[^0-9$%]*([$€£]?\s?[0-9.,]+\s?(?:million|billion|m|bn)?)", re.IGNORECASE),
}


def _extract_financial_facts(text: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    for key, pattern in FINANCIAL_FACT_PATTERNS.items():
        match = pattern.search(text)
        if match:
            facts[key] = match.group(1).strip()
    return facts


def _extract_topics(chunks: Iterable[str], keyword: str, max_examples: int = 5) -> list[str]:
    keyword_lower = keyword.lower()
    results: list[str] = []
    for chunk in chunks:
        if keyword_lower in chunk.lower():
            results.append(chunk.strip())
        if len(results) >= max_examples:
            break
    return results


@dataclass
class ExtractedMetadata:
    """Machine readable metadata used for storage and retrieval."""

    page_count: int | None
    word_count: int
    financial_facts: Dict[str, str]
    investment_highlights: list[str]


class MetadataExtractor:
    """Simple rule-based metadata extractor."""

    def extract(self, document: ProcessedDocument) -> ExtractedMetadata:
        text = document.text
        words = re.findall(r"\w+", text)
        financial_facts = _extract_financial_facts(text)
        investment_highlights = _extract_topics((chunk.text for chunk in document.chunks), "highlight")

        return ExtractedMetadata(
            page_count=document.page_count,
            word_count=len(words),
            financial_facts=financial_facts,
            investment_highlights=investment_highlights,
        )


__all__ = ["MetadataExtractor", "ExtractedMetadata"]
