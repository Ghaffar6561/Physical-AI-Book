"""
Custom exceptions for the RAG Ingestion Pipeline
"""


class CrawlerError(Exception):
    """
    Raised when there's an error during URL discovery or content fetching
    """
    pass


class ExtractionError(Exception):
    """
    Raised when there's an error during text extraction from HTML
    """
    pass


class ChunkingError(Exception):
    """
    Raised when there's an error during text chunking
    """
    pass


class EmbeddingError(Exception):
    """
    Raised when there's an error during embedding generation
    """
    pass


class StorageError(Exception):
    """
    Raised when there's an error during storage operations
    """
    pass


class RetrievalError(Exception):
    """
    Raised when there's an error during retrieval operations
    """
    pass