# Quickstart Guide: Book Content Chat Integration

## Prerequisites

- Python 3.11+
- Node.js 16+ (for Docusaurus frontend)
- Access to Qdrant Cloud instance with populated book data
- OpenAI API key

## Setup Backend

1. **Install Python dependencies**:
   ```bash
   pip install fastapi uvicorn python-multipart pydantic openai
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export QDRANT_URL="your-qdrant-cloud-url"
   export QDRANT_API_KEY="your-qdrant-api-key"
   ```

3. **Run the backend server**:
   ```bash
   uvicorn api:app --reload --port 8000
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

## Development Workflow

1. **Backend development**:
   - Implement the API endpoints in `api.py`
   - Create data models in the `models/` directory
   - Implement business logic in the `services/` directory
   - Write tests in the `tests/` directory

2. **Frontend development**:
   - Create React components for the chat interface
   - Implement API service calls in the `services/` directory
   - Add the chat page to the Docusaurus site

3. **Testing**:
   - Run backend tests: `pytest`
   - Run frontend tests: `npm test`
   - Perform integration tests between frontend and backend

## Key Components

- `api.py`: Main FastAPI application with CORS and endpoints
- `agent.py`: Integration with the existing RAG agent
- `models/`: Pydantic models for request/response validation
- `services/`: Business logic for processing questions and responses
- `book_frontend/src/components/ChatInterface`: React component for the chat UI
- `book_frontend/src/services/api`: Service for API communication