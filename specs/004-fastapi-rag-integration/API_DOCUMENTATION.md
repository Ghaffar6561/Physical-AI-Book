# API Documentation: Book Content Chat Integration

## Base URL
```
http://localhost:8000
```

## Authentication
No authentication required for local development.

## Endpoints

### Health Check
Check the health status of the API service.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-31T00:00:00.000000"
}
```

**Status Codes:**
- `200 OK`: Service is healthy

---

### Chat
Submit a question about book content and receive an AI-generated answer with source citations.

**Endpoint:** `POST /chat`

**Request Body:**
```json
{
  "message": "What is this book about?",
  "selected_text": "Optional text selected by user",
  "top_k": 5
}
```

**Request Parameters:**
| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `message` | string | Yes | The user's question | 1-1000 characters |
| `selected_text` | string | No | Selected text for context-specific questions | 1-5000 characters |
| `top_k` | integer | No | Number of top results to retrieve | 1-20 (default: 5) |

**Response:**
```json
{
  "answer": "This book is about...",
  "sources": [
    {
      "content": "Excerpt from the book...",
      "page_number": 42,
      "section_title": "Introduction",
      "url": "https://example.com/book/intro",
      "confidence_score": 0.95
    }
  ],
  "question_id": null
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | AI-generated answer to the question |
| `sources` | array | List of source citations (1-10 items) |
| `sources[].content` | string | Text excerpt from the source |
| `sources[].page_number` | integer | Page number (optional) |
| `sources[].section_title` | string | Section title (optional) |
| `sources[].url` | string | URL to the source location (optional) |
| `sources[].confidence_score` | float | Relevance score 0.0-1.0 (optional) |
| `question_id` | string | Unique question identifier (optional) |

**Status Codes:**
- `200 OK`: Successful response
- `422 Unprocessable Entity`: Validation error (invalid parameters)
- `429 Too Many Requests`: Rate limit exceeded (20 requests per minute)
- `500 Internal Server Error`: Server error

**Error Response:**
```json
{
  "detail": [
    {
      "loc": ["body", "message"],
      "msg": "String should have at least 1 character",
      "type": "string_too_short",
      "ctx": {"min_length": 1}
    }
  ]
}
```

## Rate Limiting
- **Limit:** 20 requests per minute per IP address
- **Response:** 429 Too Many Requests when limit exceeded

## CORS Policy
**Allowed Origins:** `http://localhost:3000` (Docusaurus default)
**Allowed Methods:** GET, POST
**Allowed Headers:** All

## Examples

### Basic Question
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main concepts in this book?",
    "top_k": 5
  }'
```

### Selected Text Mode
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What does this mean?",
    "selected_text": "The RAG system combines retrieval with generation...",
    "top_k": 3
  }'
```

### Health Check
```bash
curl -X GET http://localhost:8000/health
```

## Interactive Documentation
FastAPI provides automatic interactive API documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
