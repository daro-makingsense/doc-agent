"""FastAPI application exposing ingestion and chat endpoints."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ..agents.chat_agent import ChatRAGAgent
from ..models.schemas import ChatRequest, ChatResponse, UploadRequest, UploadResponse
from ..workflows.pipeline import DocumentPipeline


app = FastAPI(title="Financial Document Agent")


@app.on_event("startup")
def _startup() -> None:
    # Lazily construct heavy dependencies during startup.
    app.state.pipeline = DocumentPipeline()
    app.state.chat_agent = ChatRAGAgent()


@app.post("/upload", response_model=UploadResponse)
def upload_document(request: UploadRequest) -> UploadResponse:
    try:
        result = app.state.pipeline.run(request.file_path, topic=request.topic)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return UploadResponse(**result)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = app.state.chat_agent.chat(
        session_id=request.session_id,
        user_id=request.user_id,
        message=request.message,
        top_k=request.top_k,
    )
    return ChatResponse(**result)


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


__all__ = ["app"]
