# Quickstart: RAG Ingestion Pipeline

## Prerequisites

- Python 3.11 or higher
- `uv` package manager installed
- Cohere API key
- Qdrant Cloud account and API key
- Access to the Docusaurus book URLs to be ingested

## Setup

1. **Initialize the project:**
   ```bash
   # Navigate to the project root
   cd /path/to/your/project

   # Create backend directory
   mkdir -p backend backup

   # Initialize Python project with uv (if not already done)
   cd backend
   uv init
   ```

2. **Install dependencies:**
   ```bash
   # Install required packages
   uv pip install requests beautifulsoup4 cohere qdrant-client python-dotenv pytest
   
   # Or create requirements.txt and install:
   # requests==2.31.0
   # beautifulsoup4==4.12.2
   # cohere==4.2.7
   # qdrant-client==1.7.2
   # python-dotenv==1.0.0
   # pytest==7.4.0
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file with your credentials
   cp .env.example .env
   
   # Edit .env and add your keys:
   COHERE_API_KEY=your_cohere_api_key_here
   QDRANT_URL=your_qdrant_cluster_url
   QDRANT_API_KEY=your_qdrant_api_key
   BOOK_BASE_URL=https://your-book-url.com
   QDRANT_COLLECTION_NAME=your_collection_name
   ```

## Project Structure

After setup, your project structure should look like:

```
backend/
├── main.py              # Main ingestion pipeline
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
├── config.py            # Configuration management
├── crawler.py           # URL discovery and content fetching
├── extractor.py         # Text extraction logic
├── chunker.py           # Text chunking logic
├── embedder.py          # Embedding generation using Cohere
└── storage.py           # Qdrant storage operations

tests/
├── unit/
│   ├── test_crawler.py
│   ├── test_extractor.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   └── test_storage.py
├── integration/
│   ├── test_pipeline.py
│   └── test_end_to_end.py
└── fixtures/
    └── sample_docs/
```

## Running the Pipeline

1. **Execute the full pipeline:**
   ```bash
   python main.py
   ```

2. **Run with specific options:**
   ```bash
   # Run with custom base URL
   python main.py --base-url https://different-book.com
   
   # Process only specific URLs
   python main.py --urls https://book.com/page1,https://book.com/page2
   ```

## Configuration

The pipeline can be configured via environment variables in your `.env` file:

- `BOOK_BASE_URL`: Base URL of the Docusaurus book to crawl
- `COHERE_API_KEY`: Your Cohere API key for generating embeddings
- `QDRANT_URL`: URL of your Qdrant instance
- `QDRANT_API_KEY`: API key for Qdrant access
- `QDRANT_COLLECTION_NAME`: Name of the collection to store vectors in
- `CHUNK_SIZE`: Size of text chunks in tokens (default: 512)
- `CHUNK_OVERLAP`: Overlap between chunks in tokens (default: 51)
- `BATCH_SIZE`: Number of chunks to process in each Cohere API call (default: 10)

## Testing

1. **Run all tests:**
   ```bash
   pytest
   ```

2. **Run specific test suites:**
   ```bash
   # Run unit tests
   pytest tests/unit/
   
   # Run integration tests
   pytest tests/integration/
   ```

## Development Workflow

1. **Implement crawler functionality** (`crawler.py`):
   - Implement URL discovery from sitemap.xml or by crawling from base URL
   - Ensure URLs are normalized and deduplicated

2. **Implement text extraction** (`extractor.py`):
   - Extract clean text from HTML content
   - Preserve semantic meaning and structure

3. **Implement chunking logic** (`chunker.py`):
   - Split text into 512-token chunks with 51-token overlap
   - Maintain semantic boundaries when possible

4. **Implement embedding generation** (`embedder.py`):
   - Generate embeddings using Cohere API
   - Handle batching and rate limiting

5. **Implement storage logic** (`storage.py`):
   - Store embeddings in Qdrant with stable IDs
   - Ensure idempotent operations to prevent duplicates

6. **Orchestrate in main** (`main.py`):
   - Coordinate the entire pipeline
   - Handle errors and logging appropriately

## Validation

After running the pipeline, verify:

1. Check Qdrant dashboard to confirm vectors were stored
2. Verify no duplicate vectors exist when running multiple times
3. Confirm all expected book URLs were processed
4. Check logs for any errors or warnings during processing