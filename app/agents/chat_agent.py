"""Chat-oriented RAG agent."""
from __future__ import annotations

from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import get_settings
from ..db import ChatMessage, ChatSession, session_scope
from ..services.embedding_service import EmbeddingService
from ..services.vector_store import PineconeVectorStore


class ChatRAGAgent:
    """Answers user questions grounded in embedded CIM documents."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.has_openai:
            raise RuntimeError("OPENAI_API_KEY is required for chat agent")
        self.llm = ChatOpenAI(model=settings.chat_model, temperature=0.1)
        self.embedding_service = EmbeddingService()
        self.vector_store = PineconeVectorStore()

    def chat(self, *, session_id: str, user_id: str, message: str, top_k: int = 5) -> Dict[str, object]:
        with session_scope() as session:
            session_obj = (
                session.query(ChatSession)
                .filter(ChatSession.session_id == session_id)
                .one_or_none()
            )
            if session_obj is None:
                session_obj = ChatSession(session_id=session_id, user_id=user_id)
                session.add(session_obj)
                session.flush()
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in sorted(session_obj.messages, key=lambda m: m.created_at)
            ]
            session_db_id = session_obj.id

        query_embedding = self.embedding_service.embed_texts([message])[0]
        matches = self.vector_store.similarity_search(query_embedding, top_k=top_k)
        context_snippets = []
        references: List[Dict[str, object]] = []
        for match in matches:
            metadata = getattr(match, "metadata", {}) or {}
            doc_id = metadata.get("document_id")
            chunk_index = metadata.get("chunk_index")
            text = metadata.get("text", "")
            score = getattr(match, "score", None)
            context_snippets.append(f"[doc {doc_id} chunk {chunk_index}] {text}")
            references.append(
                {
                    "document_id": doc_id,
                    "chunk_index": chunk_index,
                    "score": score,
                }
            )
        context_block = "\n".join(context_snippets)

        messages = [
            SystemMessage(
                content=(
                    "You are a financial research assistant. Use only the provided context snippets "
                    "from Confidential Information Memoranda. Always cite the snippet ids in your answer."
                )
            )
        ]
        for item in history:
            if item["role"] == "user":
                messages.append(HumanMessage(content=item["content"]))
            else:
                messages.append(AIMessage(content=item["content"]))
        messages.append(
            HumanMessage(
                content=(
                    f"Context:\n{context_block}\n\nUser question: {message}\n"
                    "Provide a concise answer followed by bullet citations referencing snippet ids."
                )
            )
        )

        response = self.llm.invoke(messages)
        answer = response.content

        with session_scope() as session:
            session_obj = (
                session.query(ChatSession)
                .filter(ChatSession.id == session_db_id)
                .one()
            )
            session.add(ChatMessage(session_id=session_obj.id, role="user", content=message))
            session.add(
                ChatMessage(
                    session_id=session_obj.id,
                    role="assistant",
                    content=answer,
                    references=references,
                )
            )

        return {"answer": answer, "references": references}


__all__ = ["ChatRAGAgent"]
