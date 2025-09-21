"""LangGraph workflow connecting ingestion and agents."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from ..agents.research_agent import DeepResearchAgent
from ..services.doc_processing import DoclingDocumentProcessor, ProcessedDocument
from ..services.embedding_service import EmbeddingService
from ..services.metadata_extractor import ExtractedMetadata, MetadataExtractor
from ..services.upload_service import UploadService


class PipelineState(TypedDict, total=False):
    file_path: str
    topic: str
    processed: ProcessedDocument
    metadata: ExtractedMetadata
    embeddings: List[List[float]]
    upload_summary: Dict[str, object]
    research: Dict[str, object]


class DocumentPipeline:
    """High-level workflow orchestrating document ingestion and research."""

    def __init__(self) -> None:
        self.processor = DoclingDocumentProcessor()
        self.metadata_extractor = MetadataExtractor()
        self.embedding_service = EmbeddingService()
        self.upload_service = UploadService()
        self.research_agent = DeepResearchAgent()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(PipelineState)
        workflow.add_node("parse", self._parse_document)
        workflow.add_node("extract_metadata", self._extract_metadata)
        workflow.add_node("embed", self._generate_embeddings)
        workflow.add_node("persist", self._persist_document)
        workflow.add_node("research", self._research_topic)

        workflow.set_entry_point("parse")
        workflow.add_edge("parse", "extract_metadata")
        workflow.add_edge("extract_metadata", "embed")
        workflow.add_edge("embed", "persist")
        workflow.add_conditional_edges(
            "persist",
            self._should_research,
            {
                "research": "research",
                "end": END,
            },
        )
        workflow.add_edge("research", END)
        return workflow.compile()

    def _parse_document(self, state: PipelineState) -> PipelineState:
        processed = self.processor.load(state["file_path"])
        return {"processed": processed}

    def _extract_metadata(self, state: PipelineState) -> PipelineState:
        metadata = self.metadata_extractor.extract(state["processed"])
        return {"metadata": metadata}

    def _generate_embeddings(self, state: PipelineState) -> PipelineState:
        texts = [chunk.text for chunk in state["processed"].chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        return {"embeddings": embeddings}

    def _persist_document(self, state: PipelineState) -> PipelineState:
        summary = self.upload_service.ingest(
            state["file_path"],
            filename=Path(state["file_path"]).name,
            processed=state["processed"],
            metadata=state["metadata"],
            embeddings=state["embeddings"],
        )
        return {"upload_summary": summary}

    def _should_research(self, state: PipelineState) -> str:
        return "research" if state.get("topic") else "end"

    def _research_topic(self, state: PipelineState) -> PipelineState:
        topic = state.get("topic")
        if not topic:
            return {}
        context = state["processed"].text
        result = self.research_agent.run(topic=topic, context=context)
        return {"research": result}

    def run(self, file_path: str | Path, *, topic: str | None = None) -> Dict[str, object]:
        initial_state: PipelineState = {"file_path": str(file_path)}
        if topic:
            initial_state["topic"] = topic
        result = self.graph.invoke(initial_state)
        output: Dict[str, object] = {"upload_summary": result.get("upload_summary")}
        if topic:
            output["research"] = result.get("research")
        return output


__all__ = ["DocumentPipeline"]
