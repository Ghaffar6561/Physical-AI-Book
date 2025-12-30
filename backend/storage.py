"""
Storage module for the RAG Ingestion Pipeline
"""
import logging
import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rag_models import EmbeddingVector, QdrantRecord
from config import Config
from exceptions import StorageError
import hashlib


def chunk_id_to_uuid(chunk_id: str) -> str:
    """
    Convert a SHA256 chunk_id to a valid UUID format for Qdrant.
    Takes the first 32 hex characters and formats as UUID.

    Args:
        chunk_id: SHA256 hash string (64 hex characters)

    Returns:
        UUID formatted string
    """
    # Take first 32 hex chars and format as UUID
    hex_str = chunk_id[:32]
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


logger = logging.getLogger(__name__)


def store_embeddings(embedding_vectors: List[EmbeddingVector], collection_name: str) -> None:
    """
    Store embedding vectors in Qdrant
    
    Args:
        embedding_vectors: List of EmbeddingVector objects to store
        collection_name: Name of the Qdrant collection to store vectors in
    """
    config = Config()
    
    # Initialize Qdrant client
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )
    
    # Ensure collection exists
    _ensure_collection_exists(client, collection_name, embedding_vectors[0].dimension if embedding_vectors else 1536)
    
    # Prepare points for upsert
    points = []
    for embedding_vector in embedding_vectors:
        # Create payload with metadata including source_url and content for RAG
        payload = {
            "chunk_id": embedding_vector.chunk_id,
            "model": embedding_vector.model,
            "created_at": embedding_vector.created_at.isoformat(),
            "source_url": embedding_vector.source_url,
            "content": embedding_vector.content,
        }

        # Create point for upsert (convert chunk_id to UUID format for Qdrant)
        point = models.PointStruct(
            id=chunk_id_to_uuid(embedding_vector.chunk_id),
            vector=embedding_vector.vector,
            payload=payload
        )
        points.append(point)

    # Upsert points to Qdrant (this handles duplicates automatically)
    try:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Successfully stored {len(points)} vectors in collection '{collection_name}'")
    except Exception as e:
        logger.error(f"Error storing vectors in Qdrant: {e}")
        raise StorageError(f"Failed to store embeddings in Qdrant: {e}")


def check_duplicate(chunk_id: str, collection_name: str) -> bool:
    """
    Check if a chunk with the given ID already exists in Qdrant

    Args:
        chunk_id: ID of the chunk to check
        collection_name: Name of the Qdrant collection to check in

    Returns:
        Boolean indicating if duplicate exists
    """
    config = Config()

    # Initialize Qdrant client
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )

    try:
        # Try to retrieve the point by ID (convert to UUID format)
        records = client.retrieve(
            collection_name=collection_name,
            ids=[chunk_id_to_uuid(chunk_id)]
        )
        return len(records) > 0
    except Exception as e:
        logger.error(f"Error checking for duplicate in Qdrant: {e}")
        raise StorageError(f"Failed to check for duplicate in Qdrant: {e}")


def store_embeddings_idempotent(embedding_vectors: List[EmbeddingVector], collection_name: str) -> None:
    """
    Store embedding vectors in Qdrant with idempotent behavior (no duplicates on re-run)

    Args:
        embedding_vectors: List of EmbeddingVector objects to store
        collection_name: Name of the Qdrant collection to store vectors in
    """
    config = Config()

    # Initialize Qdrant client
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )

    # Ensure collection exists
    _ensure_collection_exists(client, collection_name, embedding_vectors[0].dimension if embedding_vectors else 1536)

    # Prepare points for upsert, skipping duplicates
    points = []
    skipped_count = 0

    for embedding_vector in embedding_vectors:
        # Check if this chunk_id already exists in the collection
        if not check_duplicate(embedding_vector.chunk_id, collection_name):
            # Create payload with metadata including source_url and content for RAG
            payload = {
                "chunk_id": embedding_vector.chunk_id,
                "model": embedding_vector.model,
                "created_at": embedding_vector.created_at.isoformat(),
                "source_url": embedding_vector.source_url,
                "content": embedding_vector.content,
            }

            # Create point for upsert (convert chunk_id to UUID format for Qdrant)
            point = models.PointStruct(
                id=chunk_id_to_uuid(embedding_vector.chunk_id),
                vector=embedding_vector.vector,
                payload=payload
            )
            points.append(point)
        else:
            skipped_count += 1
            logger.debug(f"Skipping duplicate chunk: {embedding_vector.chunk_id}")

    # Upsert points to Qdrant (this handles the new entries)
    try:
        if points:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Successfully stored {len(points)} new vectors in collection '{collection_name}'")

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} duplicate vectors")

    except Exception as e:
        logger.error(f"Error storing vectors in Qdrant: {e}")
        raise StorageError(f"Failed to store embeddings in Qdrant: {e}")


def _ensure_collection_exists(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """
    Ensure the specified collection exists in Qdrant, create if it doesn't
    """
    try:
        # Try to get collection info
        client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' already exists")
    except:
        # Collection doesn't exist, create it
        logger.info(f"Creating collection '{collection_name}'")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"Collection '{collection_name}' created successfully")