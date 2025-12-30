# API Contracts: RAG Ingestion Pipeline

## Component Interfaces

### Crawler Interface
**Module**: `crawler.py`

#### Function: `discover_urls(base_url: str) -> List[str]`
**Purpose**: Discover all URLs in the Docusaurus book
**Input**: Base URL of the book
**Output**: List of all discovered URLs
**Error Handling**: Raises `CrawlerError` if unable to discover URLs

#### Function: `fetch_content(url: str) -> str`
**Purpose**: Fetch and return the HTML content of a URL
**Input**: URL to fetch
**Output**: HTML content as string
**Error Handling**: Raises `CrawlerError` if unable to fetch content

### Extractor Interface
**Module**: `extractor.py`

#### Function: `extract_text(html_content: str, source_url: str) -> str`
**Purpose**: Extract clean text from HTML content
**Input**: HTML content string and source URL
**Output**: Clean text content as string
**Error Handling**: Raises `ExtractionError` if unable to extract content

### Chunker Interface
**Module**: `chunker.py`

#### Function: `chunk_text(text: str, source_url: str, chunk_size: int = 512, chunk_overlap: int = 51) -> List[TextChunk]`
**Purpose**: Split text into chunks with specified size and overlap
**Input**: Text content, source URL, chunk size, chunk overlap
**Output**: List of TextChunk objects
**Error Handling**: Raises `ChunkingError` if unable to chunk content properly

### Embedder Interface
**Module**: `embedder.py`

#### Function: `generate_embeddings(chunks: List[TextChunk], batch_size: int = 10) -> List[EmbeddingVector]`
**Purpose**: Generate embeddings for text chunks using Cohere
**Input**: List of TextChunk objects and batch size
**Output**: List of EmbeddingVector objects
**Error Handling**: Raises `EmbeddingError` if unable to generate embeddings

### Storage Interface
**Module**: `storage.py`

#### Function: `store_embeddings(embedding_vectors: List[EmbeddingVector], collection_name: str) -> None`
**Purpose**: Store embedding vectors in Qdrant
**Input**: List of EmbeddingVector objects and collection name
**Output**: None
**Error Handling**: Raises `StorageError` if unable to store embeddings

#### Function: `check_duplicate(chunk_id: str, collection_name: str) -> bool`
**Purpose**: Check if a chunk with the given ID already exists in Qdrant
**Input**: Chunk ID and collection name
**Output**: Boolean indicating if duplicate exists
**Error Handling**: Raises `StorageError` if unable to check for duplicates

## Data Contracts

### TextChunk Object
```json
{
  "id": "string (deterministic ID based on URL and position)",
  "content": "string (text content)",
  "source_url": "string (original URL)",
  "position": "integer (position in original document)",
  "token_count": "integer (number of tokens in chunk)"
}
```

### EmbeddingVector Object
```json
{
  "chunk_id": "string (reference to TextChunk.id)",
  "vector": "array<float> (embedding vector)",
  "model": "string (embedding model name)",
  "dimension": "integer (vector dimensionality)"
}
```

## Configuration Contract

### Environment Variables
```env
BOOK_BASE_URL = "string (base URL of Docusaurus book)"
COHERE_API_KEY = "string (Cohere API key)"
QDRANT_URL = "string (Qdrant cluster URL)"
QDRANT_API_KEY = "string (Qdrant API key)"
QDRANT_COLLECTION_NAME = "string (name of Qdrant collection)"
CHUNK_SIZE = "integer (default: 512)"
CHUNK_OVERLAP = "integer (default: 51)"
BATCH_SIZE = "integer (default: 10)"
```

## Error Contract

### Standard Error Format
```json
{
  "error_code": "string (e.g., 'CRAWLER_ERROR', 'EXTRACTION_ERROR')",
  "message": "string (human-readable error message)",
  "details": "object (optional technical details)",
  "timestamp": "string (ISO 8601 format)"
}
```

## Success Criteria Contract

### Pipeline Execution Contract
- All public book URLs are successfully crawled and text is extracted (100% success rate)
- Text is consistently chunked according to 512-token size and 51-token overlap strategy (95% accuracy)
- Embeddings are generated for 100% of text chunks with successful API responses
- All embeddings and metadata are stored with unique IDs and retrievable without errors
- Pipeline re-execution does not create any duplicate vectors in the database (0% duplication rate)
- Logging system provides clear status updates for ingestion, embedding, and storage operations (95% completeness)