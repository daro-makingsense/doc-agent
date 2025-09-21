# AI-Powered Financial Document Processing System

A comprehensive AI system for uploading, processing, and researching financial documents using advanced language models and vector search capabilities.

## 🚀 Features

### Core Functionality
- **Document Upload & Processing**: Upload PDFs and extract structured content using docling
- **AI-Powered Analysis**: Extract financial facts, investment data, and key metrics using OpenAI GPT-4
- **Vector Search**: Store document embeddings in Pinecone for semantic search and retrieval
- **Deep Research Agent**: Generate comprehensive content outlines and analysis using LangGraph
- **Chat/RAG Agent**: Interactive chat with document context and retrieval-augmented generation

### Technical Highlights
- **LangGraph Agents**: Two specialized agents for research and chat functionality
- **Pinecone Integration**: Vector database for efficient semantic search
- **PostgreSQL**: Metadata storage and chat session management
- **FastAPI**: RESTful API with automatic documentation
- **Docling**: Advanced document parsing for financial documents

## 📋 System Requirements

- Python 3.8+
- PostgreSQL 12+
- OpenAI API Key
- Pinecone API Key

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd financial-document-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```env
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/financial_docs

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=financial-documents

# Application Configuration
DEBUG=false
UPLOAD_DIRECTORY=/tmp/uploads
```

### 4. Set Up Database
Make sure PostgreSQL is running and create the database:

```bash
createdb financial_docs
```

The application will automatically create the required tables on startup.

### 5. Run the Application
```bash
python -m app.main
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## 🔧 API Usage

### Document Management

#### Upload Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@financial_report.pdf"
```

Response:
```json
{
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "financial_report.pdf",
  "file_size": 1024000,
  "status": "uploaded",
  "processing_started": true
}
```

#### Check Processing Status
```bash
curl -X GET "http://localhost:8000/api/v1/documents/{document_id}/status"
```

#### List Documents
```bash
curl -X GET "http://localhost:8000/api/v1/documents"
```

### Deep Research

#### Start Research Task
```bash
curl -X POST "http://localhost:8000/api/v1/research/start" \
     -H "Content-Type: application/json" \
     -d '{
       "document_id": "123e4567-e89b-12d3-a456-426614174000",
       "topic": "Key Investment Highlights",
       "custom_query": "What are the main investment opportunities and risks?"
     }'
```

#### Get Research Results
```bash
curl -X GET "http://localhost:8000/api/v1/research/tasks/{task_id}"
```

### Chat Interface

#### Send Message
```bash
curl -X POST "http://localhost:8000/api/v1/chat/send" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What is the company revenue?",
       "document_id": "123e4567-e89b-12d3-a456-426614174000",
       "use_rag": true
     }'
```

#### Create Chat Session
```bash
curl -X POST "http://localhost:8000/api/v1/chat/sessions" \
     -H "Content-Type: application/json" \
     -d '{
       "document_id": "123e4567-e89b-12d3-a456-426614174000",
       "session_name": "Financial Analysis Session"
     }'
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   LangGraph      │    │   PostgreSQL    │
│   Web Server    │────│   Agents         │────│   Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Docling       │    │   OpenAI         │    │   Pinecone      │
│   Parser        │    │   LLM            │    │   Vector Store  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Agent Architecture

#### Deep Research Agent
The Deep Research Agent uses a multi-step workflow:

1. **Topic Analysis**: Understand research requirements
2. **Question Generation**: Create specific research questions
3. **Information Retrieval**: Search document embeddings
4. **Content Outline**: Structure findings into organized outline
5. **Detailed Content**: Generate comprehensive content for each section

#### Chat/RAG Agent
The Chat Agent provides conversational interaction:

1. **Context Loading**: Load chat history and session context
2. **Retrieval Decision**: Determine if RAG is needed
3. **Information Retrieval**: Search relevant document chunks
4. **Response Generation**: Generate contextual response
5. **Conversation Storage**: Save interaction history

### Data Models

#### Documents
- Document metadata and processing status
- Financial facts and investment data
- File information and processing timestamps

#### Chat Sessions
- Session configuration and user context
- Message history and conversation state
- RAG context and retrieval settings

#### Research Tasks
- Research topics and custom queries
- Generated content outlines and findings
- Source citations and processing metadata

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
# Unit tests only
pytest tests/test_basic.py::TestUtilityHelpers -v

# API tests only
pytest tests/test_basic.py::TestAPIEndpoints -v
```

Note: Integration tests require actual API keys and database setup.

## 📚 API Documentation

The system provides interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔍 Example Workflow

### Complete Document Analysis Workflow

1. **Upload Document**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/documents/upload" \
        -F "file=@cim_document.pdf"
   ```

2. **Wait for Processing**
   ```bash
   curl "http://localhost:8000/api/v1/documents/{document_id}/status"
   ```

3. **Conduct Research**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/research/start" \
        -H "Content-Type: application/json" \
        -d '{"document_id": "{document_id}", "topic": "Investment Highlights"}'
   ```

4. **Start Chat Session**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/chat/sessions" \
        -H "Content-Type: application/json" \
        -d '{"document_id": "{document_id}"}'
   ```

5. **Ask Questions**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/chat/send" \
        -H "Content-Type: application/json" \
        -d '{"message": "What are the key risks?", "session_id": "{session_id}"}'
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `PINECONE_API_KEY` | Pinecone API key | Required |
| `PINECONE_INDEX_NAME` | Pinecone index name | `financial-documents` |
| `DEBUG` | Enable debug mode | `false` |
| `MAX_FILE_SIZE` | Maximum upload size (bytes) | `52428800` (50MB) |
| `UPLOAD_DIRECTORY` | File upload directory | `/tmp/uploads` |

### Model Configuration

The system uses the following AI models:
- **LLM**: `gpt-4-1106-preview` (configurable via `OPENAI_MODEL`)
- **Embeddings**: `text-embedding-3-large` (configurable via `OPENAI_EMBEDDING_MODEL`)

## 🚨 Error Handling

The system includes comprehensive error handling:

- **File Upload Errors**: Size limits, format validation
- **Processing Errors**: Document parsing failures
- **API Errors**: Rate limiting, authentication issues
- **Database Errors**: Connection issues, constraint violations

All errors are logged and returned with appropriate HTTP status codes.

## 📊 Monitoring

### System Statistics

Get system statistics:
```bash
curl "http://localhost:8000/api/v1/system/stats"
```

### Health Check

Check system health:
```bash
curl "http://localhost:8000/api/v1/system/health"
```

## 🔐 Security Considerations

- Environment variables for sensitive configuration
- File upload validation and size limits
- Input sanitization for chat messages
- Database connection security
- API rate limiting (to be implemented)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:

1. Check the API documentation at `/docs`
2. Review the error logs
3. Check system status at `/api/v1/system/health`
4. Open an issue on GitHub

## 🎯 Future Enhancements

- [ ] User authentication and authorization
- [ ] Document comparison and analysis
- [ ] Batch document processing
- [ ] Advanced search filters
- [ ] Export functionality for research reports
- [ ] WebSocket support for real-time updates
- [ ] Docker deployment configuration
- [ ] Kubernetes manifests
- [ ] Advanced monitoring and metrics