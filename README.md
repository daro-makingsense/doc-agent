# Financial Document Agent Prototype

This repository contains a prototype system for ingesting Confidential Information Memoranda (CIMs),
extracting key facts, storing embeddings in Pinecone, and exposing both a deep research agent and a
chat-oriented Retrieval-Augmented Generation (RAG) agent.

## Features

- **Document Ingestion Pipeline** powered by [Docling](https://docling-project.github.io/) for parsing
  PDFs and splitting them into chunks.
- **Metadata Extraction** that surfaces financial facts and investment highlights.
- **Vector Storage** backed by Pinecone for semantic search over document chunks.
- **PostgreSQL Persistence** of document metadata, chunks, and chat transcripts.
- **Deep Research Agent** implemented with [LangGraph](https://github.com/langchain-ai/langgraph) and OpenAI models
  to generate structured outlines for CIM topics.
- **Chat/RAG Agent** that answers user questions grounded in embedded document snippets.
- **FastAPI Service** exposing `/upload` and `/chat` endpoints to simulate ingestion and Q&A.

## Architecture Overview

```
┌────────────┐       ┌────────────────────┐       ┌─────────────────┐
│  Upload    │ ───▶ │ LangGraph Pipeline │ ───▶ │ PostgreSQL +    │
│ Endpoint   │       │  (parse → embed →  │       │ Pinecone        │
└────────────┘       │   persist → agents)│       └─────────────────┘
                                 │
                                 ▼
                         Deep Research Agent
                                 │
                                 ▼
                           Chat / RAG Agent
```

## Getting Started

1. **Install dependencies** (Python 3.10+):

   ```bash
   pip install -e .
   ```

2. **Provision infrastructure**:
   - PostgreSQL database (see `db/schema.sql` for schema).
   - Pinecone index (serverless recommended).

3. **Configure environment variables** (e.g. via `.env`):

   ```bash
   export OPENAI_API_KEY=sk-...
   export PINECONE_API_KEY=...
   export PINECONE_ENVIRONMENT=us-east-1
   export PINECONE_INDEX=cim-documents
   export POSTGRES_DSN=postgresql+psycopg2://user:pass@localhost:5432/doc_agent
   ```

4. **Run the API**:

   ```bash
   uvicorn app.api.main:app --reload
   ```

## API Usage

### Upload & Research

`POST /upload`

```json
{
  "file_path": "./sample-documents/acme-cim.pdf",
  "topic": "Key Investment Highlights"
}
```

Response:

```json
{
  "upload_summary": {
    "document_id": 1,
    "page_count": 42,
    "word_count": 12345,
    "financial_facts": {"revenue": "$120M", "ebitda": "$24M"}
  },
  "research": {
    "topic": "Key Investment Highlights",
    "plan": "...",
    "notes": ["..."],
    "outline": ["..."]
  }
}
```

### Chat / Retrieval-Augmented Generation

`POST /chat`

```json
{
  "session_id": "demo-session",
  "user_id": "analyst-1",
  "message": "What is the projected revenue growth?",
  "top_k": 5
}
```

Response:

```json
{
  "answer": "The company projects 18% YoY revenue growth [doc 1 chunk 3].",
  "references": [
    {"document_id": 1, "chunk_index": 3, "score": 0.91}
  ]
}
```

## Database Schema

See [`db/schema.sql`](db/schema.sql) for the PostgreSQL DDL used by the ORM models defined in `app/db.py`.

## Notes

- The prototype relies on external services (OpenAI and Pinecone); ensure the environment variables are set before running the API.
- Docling currently focuses on PDF parsing; other formats can be supported by extending `DoclingDocumentProcessor`.
- The LangGraph workflow defined in `app/workflows/pipeline.py` orchestrates parsing, embedding, persistence, and optional deep research execution.

## Development

- Run `uvicorn app.api.main:app --reload` during development for live reloads.
- Optional tooling such as `black`, `ruff`, and `mypy` can be installed via `pip install -e .[dev]`.

