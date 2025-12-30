# Book Content Chat Integration

This project implements a FastAPI backend that exposes HTTP endpoints to interact with the RAG agent, allowing users to ask questions about book content and receive grounded answers with source citations. The backend connects to a Docusaurus frontend via a full-page chatbot UI, supporting both book-wide queries and selected-text mode.

## Prerequisites

- Python 3.11+
- Node.js 16+ (for Docusaurus frontend)
- Access to Qdrant Cloud instance with populated book data
- OpenAI API key

## Setup Backend

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export QDRANT_URL="your-qdrant-cloud-url"
   export QDRANT_API_KEY="your-qdrant-api-key"
   ```

3. **Run the backend server**:
   ```bash
   uvicorn backend.api:app --reload --port 8000
   ```

4. **Verify the server is running**:
   - Visit http://localhost:8000/health to check the health status
   - Visit http://localhost:8000/docs for API documentation

## Setup Frontend

1. **Navigate to the book frontend directory**:
   ```bash
   cd book_frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the Docusaurus development server**:
   ```bash
   npm run start
   ```

4. **Access the chat interface**:
   - The chat interface will be available at http://localhost:3000/chat

## API Usage

### Send a question to the chat endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main concepts in this book?",
    "selected_text": "Optional selected text from the page...",
    "top_k": 5
  }'
```

### Health check

```bash
curl http://localhost:8000/health
```

## Features

1. **Interactive Book Chat**: Ask questions about book content and receive answers with source citations
2. **Selected Text Mode**: Ask questions specifically about selected portions of text
3. **Error Handling**: Comprehensive error handling with user-friendly messages
4. **Rate Limiting**: API abuse prevention with rate limiting
5. **Input Sanitization**: Protection against injection attacks
6. **Performance Monitoring**: Response time tracking and alerts

## Architecture

- `backend/api.py`: Main FastAPI application with CORS and endpoints
- `backend/agent.py`: Integration with the existing RAG agent
- `backend/models/`: Pydantic models for request/response validation
- `backend/services/`: Business logic for processing questions and responses
- `book_frontend/src/components/ChatInterface`: React component for the chat UI
- `book_frontend/src/services/api`: Service for API communication