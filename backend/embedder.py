"""
Embedding generation module for the RAG Ingestion Pipeline
"""
import cohere
import logging
import time
from typing import List
from rag_models import TextChunk, EmbeddingVector
from config import Config
from exceptions import EmbeddingError
from datetime import datetime


logger = logging.getLogger(__name__)


def generate_embeddings(chunks: List[TextChunk], batch_size: int = None) -> List[EmbeddingVector]:
    """
    Generate embeddings for text chunks using Cohere

    Args:
        chunks: List of TextChunk objects to generate embeddings for
        batch_size: Number of chunks to process in each Cohere API call (default from config)

    Returns:
        List of EmbeddingVector objects
    """
    config = Config()
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Initialize Cohere client
    co = cohere.Client(config.COHERE_API_KEY)

    embedding_vectors = []

    # Process chunks in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        try:
            # Extract text content from chunks for embedding
            texts = [chunk.content for chunk in batch]

            # Generate embeddings using Cohere
            response = co.embed(
                texts=texts,
                model='embed-english-v3.0',  # Using a standard Cohere embedding model
                input_type="search_document"  # Specify the input type
            )

            # Create EmbeddingVector objects from the response
            for idx, embedding in enumerate(response.embeddings):
                chunk = batch[idx]
                embedding_vector = EmbeddingVector(
                    chunk_id=chunk.id,
                    vector=embedding,
                    model='embed-english-v3.0',
                    dimension=len(embedding),
                    created_at=datetime.now(),
                    source_url=chunk.source_url,
                    content=chunk.content
                )
                embedding_vectors.append(embedding_vector)

            logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

            # Add a small delay between batches to respect API rate limits
            # This is a basic rate limiting approach - in production you might want more sophisticated logic
            time.sleep(0.1)  # 100ms delay between batches

        except Exception as e:
            logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")

    logger.info(f"Generated embeddings for {len(embedding_vectors)} chunks")
    return embedding_vectors