"""
RAG Ingestion Pipeline
Main entry point for the ingestion pipeline that crawls Docusaurus books,
extracts content, chunks it, generates embeddings, and stores in Qdrant.
"""
import argparse
import logging
import time
from config import Config
from crawler import crawl_and_extract_content
from chunker import chunk_text
from embedder import generate_embeddings
from storage import store_embeddings_idempotent
from exceptions import CrawlerError, ExtractionError, ChunkingError, EmbeddingError, StorageError


def main():
    """
    Main function to orchestrate the entire ingestion pipeline
    """
    parser = argparse.ArgumentParser(description='RAG Ingestion Pipeline')
    parser.add_argument('--base-url', type=str, help='Base URL of the Docusaurus book to crawl')
    parser.add_argument('--urls', type=str, help='Comma-separated URLs to process')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for failed operations')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Starting RAG Ingestion Pipeline")

    # Load configuration
    config = Config()

    # Determine URLs to process
    book_content_list = []
    for attempt in range(args.max_retries + 1):
        try:
            if args.urls:
                urls = [url.strip() for url in args.urls.split(',')]
                book_content_list = crawl_and_extract_content(urls=urls)
            elif args.base_url:
                base_url = args.base_url
                book_content_list = crawl_and_extract_content(base_url)
            else:
                base_url = config.BOOK_BASE_URL
                book_content_list = crawl_and_extract_content(base_url)

            logger.info(f"Extracted content from {len(book_content_list)} pages")
            break  # Success, exit retry loop
        except (CrawlerError, ExtractionError) as e:
            logger.warning(f"Attempt {attempt + 1} failed during content extraction: {str(e)}")
            if attempt == args.max_retries:
                logger.error(f"All {args.max_retries + 1} attempts failed during content extraction")
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

    # Step 2: Chunk the extracted content
    logger.info("Starting content chunking...")
    all_chunks = []
    for book_content in book_content_list:
        chunks = chunk_text(book_content.content, book_content.url)
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} text chunks")

    # Step 3: Generate embeddings with retry logic
    logger.info("Generating embeddings...")
    embedding_vectors = []
    for attempt in range(args.max_retries + 1):
        try:
            embedding_vectors = generate_embeddings(all_chunks)
            logger.info(f"Generated {len(embedding_vectors)} embeddings")
            break  # Success, exit retry loop
        except EmbeddingError as e:
            logger.warning(f"Attempt {attempt + 1} failed during embedding generation: {str(e)}")
            if attempt == args.max_retries:
                logger.error(f"All {args.max_retries + 1} attempts failed during embedding generation")
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

    # Step 4: Store embeddings in Qdrant with retry logic
    logger.info("Storing embeddings in Qdrant...")
    for attempt in range(args.max_retries + 1):
        try:
            store_embeddings_idempotent(embedding_vectors, config.QDRANT_COLLECTION_NAME)
            logger.info("Pipeline completed successfully!")
            break  # Success, exit retry loop
        except StorageError as e:
            logger.warning(f"Attempt {attempt + 1} failed during storage: {str(e)}")
            if attempt == args.max_retries:
                logger.error(f"All {args.max_retries + 1} attempts failed during storage")
                raise
            time.sleep(2 ** attempt)  # Exponential backoff


if __name__ == "__main__":
    main()