# Deployment Guide: Book Content Chat Integration

## Prerequisites

### Backend
- Python 3.11 or higher
- pip (Python package manager)
- Access to Qdrant Cloud instance with book content indexed
- OpenAI API key

### Frontend
- Node.js 16+ and npm
- Modern web browser

## Environment Setup

### 1. Backend Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Qdrant Configuration
QDRANT_URL=your-qdrant-cloud-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=book-content

# Server Configuration (optional)
HOST=127.0.0.1
PORT=8000
```

### 2. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- pydantic==2.5.0
- openai==1.3.5
- python-multipart==0.0.6
- qdrant-client==1.7.0
- slowapi==0.1.8
- pytest==8.0.0
- pytest-asyncio==0.23.5
- httpx>=0.27.1

### 3. Install Frontend Dependencies

```bash
cd book_frontend
npm install
```

## Running the Application

### Backend Server

From the project root directory:

```bash
# Development mode with auto-reload
python -m uvicorn backend.api:app --reload --port 8000

# Production mode
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

The backend will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Frontend Application

```bash
cd book_frontend
npm run dev
```

The frontend will be available at:
- Application: http://localhost:5173 (Vite default)

## Testing

### Run Backend Tests

```bash
# Run all tests
python -m pytest backend/tests/ -v

# Run specific test file
python -m pytest backend/tests/test_api.py -v

# Run with coverage
python -m pytest backend/tests/ --cov=backend --cov-report=html
```

### Manual Testing

1. **Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Chat Endpoint:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What is this book about?", "top_k": 5}'
   ```

3. **Frontend:** Open http://localhost:5173 and test the chat interface

## Architecture

### Backend Structure
```
backend/
├── api.py                    # FastAPI application
├── agent.py                  # RAG agent integration
├── rag_models.py             # Data models for RAG pipeline
├── config.py                 # Configuration
├── logging_config.py         # Logging setup
├── rate_limiter.py           # Rate limiting middleware
├── models/                   # Pydantic models
│   ├── chat_models.py        # Chat request/response models
│   └── response_models.py    # API response models
├── routers/                  # API routers
│   ├── health.py             # Health check endpoint
│   └── chat.py               # Chat endpoint
├── services/                 # Business logic
│   └── rag_service.py        # RAG service integration
├── utils/                    # Utilities
│   ├── sanitization.py       # Input sanitization
│   └── performance_monitoring.py  # Performance tracking
└── tests/                    # Test suite
    ├── test_api.py
    ├── test_error_handling.py
    ├── test_polish.py
    └── test_selected_text.py
```

### Frontend Structure
```
book_frontend/
├── src/
│   ├── App.js                # Main application component
│   ├── main.js               # Application entry point
│   ├── components/           # React components
│   │   ├── ChatInterface.js  # Chat UI component
│   │   └── ChatInterface.css # Styles
│   └── services/             # Services
│       ├── api.js            # API client
│       └── config.js         # Configuration
├── index.html                # HTML entry point
├── package.json              # Dependencies
└── vite.config.js            # Vite configuration
```

## Troubleshooting

### Backend Issues

**ModuleNotFoundError: No module named 'backend'**
- Ensure you're running commands from the project root directory
- Run: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"` (Linux/Mac)
- Run: `set PYTHONPATH=%PYTHONPATH%;%CD%` (Windows)

**Rate Limit Errors During Testing**
- Tests may hit rate limits when running multiple times quickly
- Wait 60 seconds between test runs or restart the server

**OpenAI API Errors**
- Verify `OPENAI_API_KEY` is set correctly in `.env`
- Check API key has sufficient credits

**Qdrant Connection Errors**
- Verify `QDRANT_URL` and `QDRANT_API_KEY` are correct
- Ensure Qdrant collection exists and is populated

### Frontend Issues

**Cannot connect to backend**
- Ensure backend server is running on port 8000
- Check CORS settings in `backend/api.py`
- Verify `API_BASE_URL` in `book_frontend/src/services/config.js`

**npm install errors**
- Clear cache: `npm cache clean --force`
- Delete `node_modules` and `package-lock.json`, then reinstall

## Performance Optimization

### Backend
- Rate limiting: 20 requests/minute (configurable in `backend/routers/chat.py`)
- Response timeout: 10 seconds
- Vector search: top_k limited to 1-20 results

### Frontend
- Debounce user input to reduce API calls
- Cache responses for repeated questions
- Show loading states during API calls

## Security Considerations

- Input sanitization enabled (XSS protection)
- Rate limiting prevents API abuse
- CORS restricted to localhost:3000 for development
- No authentication (add for production deployment)

## Next Steps for Production

1. **Authentication:** Add OAuth2 or JWT authentication
2. **Database:** Add conversation persistence
3. **Monitoring:** Integrate logging service (e.g., Sentry, CloudWatch)
4. **Deployment:** Containerize with Docker
5. **CI/CD:** Set up automated testing and deployment
6. **CORS:** Restrict to production domains
7. **Rate Limiting:** Adjust based on production traffic patterns
8. **Caching:** Add Redis for response caching
9. **Load Balancing:** Add for high availability
10. **SSL/TLS:** Enable HTTPS for secure communication
