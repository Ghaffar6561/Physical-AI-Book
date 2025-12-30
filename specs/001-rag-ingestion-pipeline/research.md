# Research Summary: RAG Ingestion Pipeline

## Decision: Project Structure
**Rationale**: Selected a single CLI application with modular components to keep the implementation simple while maintaining separation of concerns. This approach follows the KISS principle and makes the tool easy to run locally as required by the specification.

## Decision: Dependency Management
**Rationale**: Using `uv` for Python project initialization as specified in the user's plan. This is a modern, fast Python package manager that will handle dependencies and virtual environment management.

## Decision: URL Discovery Method
**Rationale**: Will implement both sitemap.xml parsing and crawling from a base URL as fallback options. This provides robustness in case one method fails. The approach will be configurable via environment variables.

## Decision: Text Extraction Strategy
**Rationale**: Using BeautifulSoup4 to extract clean text from HTML content, focusing on main content areas while filtering out navigation, headers, and footers. This is a proven approach for extracting readable content from web pages.

## Decision: Chunking Algorithm
**Rationale**: Implementing a recursive character-based text splitter that maintains semantic boundaries. The 512-token chunks with 51-token overlap specified in the requirements will be implemented using a sliding window approach.

## Decision: Embedding Service
**Rationale**: Using Cohere's embedding API as specified in the feature requirements. Will implement proper batching to optimize API usage and cost.

## Decision: Qdrant Storage Strategy
**Rationale**: Using Qdrant's upsert functionality with stable IDs to ensure idempotent operations. The stable ID will be generated based on the URL and chunk content to prevent duplicates on re-runs.

## Alternatives Considered

### For Project Structure:
- **Monolithic single file**: Rejected because it would be harder to maintain and test
- **Multiple microservices**: Rejected because it's overkill for a simple ingestion pipeline

### For Text Extraction:
- **Trafilatura library**: Considered but BeautifulSoup4 is more widely used and understood
- **Newspaper3k**: Considered but too focused on news articles, not documentation sites

### For Chunking:
- **Sentence-based splitting**: Rejected because it doesn't guarantee consistent token counts
- **Recursive splitting**: Chosen as it maintains semantic boundaries while meeting size requirements

### For Storage:
- **Other vector databases (Pinecone, Weaviate)**: Rejected as Qdrant was specified in requirements
- **Different ID strategies**: Stable ID based on URL+content chosen for reliable deduplication

## Technical Unknowns Resolved

1. **How to ensure idempotency in Qdrant storage?**
   - Solution: Use upsert operations with stable, deterministic IDs based on URL and content

2. **How to handle large documents that exceed Cohere's token limits?**
   - Solution: Pre-chunk large documents before sending to Cohere API to stay within limits

3. **How to implement the 512-token chunking with 51-token overlap?**
   - Solution: Use a recursive text splitter that creates overlapping windows

4. **How to extract text from Docusaurus pages specifically?**
   - Solution: Target common Docusaurus content selectors (e.g., main.docMainContainer, article.markdown) while falling back to general content extraction

## Implementation Notes

- Need to handle rate limiting for Cohere API calls
- Need to implement retry logic for network requests
- Need to handle various content types (text, code blocks, tables) differently during extraction
- Need to store additional metadata (URL, document title, etc.) with each vector