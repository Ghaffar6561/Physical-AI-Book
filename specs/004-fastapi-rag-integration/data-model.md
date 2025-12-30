# Data Model: Book Content Chat Integration

## Entities

### Question
**Description**: A text query submitted by the user, with optional context about selected text

**Fields**:
- `message` (string, required): The user's question text
- `selected_text` (string, optional): Text selected by the user on the page (for selected-text mode)
- `top_k` (integer, optional): Number of top results to retrieve from the RAG system (default: 5)

**Validation Rules**:
- `message` must be between 1 and 1000 characters
- `selected_text` must be between 1 and 5000 characters if provided
- `top_k` must be between 1 and 20 if provided

### Answer
**Description**: A response to the user's question, including the answer text and source citations

**Fields**:
- `answer` (string, required): The AI-generated answer to the user's question
- `sources` (array of SourceCitation, required): List of sources used to generate the answer
- `question_id` (string, optional): Unique identifier for the question (for potential future use)

**Validation Rules**:
- `answer` must be between 1 and 10000 characters
- `sources` must contain at least 1 and at most 10 citations

### SourceCitation
**Description**: Reference to the specific location in the book where the answer information was found

**Fields**:
- `content` (string, required): The text content from the source
- `page_number` (integer, optional): Page number where the content appears
- `section_title` (string, optional): Title of the section containing the content
- `url` (string, optional): URL to the specific location in the online book
- `confidence_score` (float, optional): Confidence score of the citation relevance (0.0 to 1.0)

**Validation Rules**:
- `content` must be between 1 and 2000 characters
- `page_number` must be a positive integer if provided
- `confidence_score` must be between 0.0 and 1.0 if provided

### ChatSession
**Description**: A sequence of interactions between the user and the system (though no persistence required for this implementation)

**Fields**:
- `session_id` (string, required): Unique identifier for the session
- `created_at` (datetime, required): Timestamp when the session was created
- `messages` (array of objects, optional): List of question-answer pairs in the session

**Validation Rules**:
- `session_id` must be a valid UUID string
- `messages` can be empty but if present, must contain valid Question/Answer pairs

## State Transitions

### Question Processing Flow
1. **Received**: Question is received via API endpoint
2. **Processing**: Question is sent to RAG agent for processing
3. **Completed**: Answer is generated and returned to the user
4. **Error**: If processing fails, an error response is returned

## Relationships

- A `Question` generates one `Answer`
- An `Answer` contains multiple `SourceCitation` objects
- Multiple `Question`/`Answer` pairs can belong to a `ChatSession`

## API Request/Response Models

### ChatRequest
```json
{
  "message": "What is the main concept of this book?",
  "selected_text": "Selected text from the page...",
  "top_k": 5
}
```

### ChatResponse
```json
{
  "answer": "The main concept of this book is...",
  "sources": [
    {
      "content": "The book focuses on building AI-powered applications...",
      "page_number": 15,
      "section_title": "Introduction",
      "url": "http://localhost:3000/docs/intro",
      "confidence_score": 0.95
    }
  ],
  "question_id": "uuid-string"
}
```

### ErrorResponse
```json
{
  "error": "Error message describing what went wrong",
  "code": "ERROR_CODE"
}
```