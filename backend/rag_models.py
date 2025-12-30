"""
Data models for the RAG Ingestion Pipeline
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class BookContent:
    """
    Represents the text extracted from Docusaurus book URLs
    """
    url: str
    title: str
    content: str
    created_at: datetime
    updated_at: datetime
    checksum: str


@dataclass
class TextChunk:
    """
    A segment of book content processed according to the chunking strategy
    """
    id: str
    content: str
    source_url: str
    position: int
    overlap_prev: Optional[str] = None
    overlap_next: Optional[str] = None
    token_count: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EmbeddingVector:
    """
    A numerical representation of a text chunk for semantic search
    """
    chunk_id: str
    vector: list
    model: str
    dimension: int
    created_at: datetime
    source_url: str = ""
    content: str = ""


@dataclass
class QdrantRecord:
    """
    An entry in the Qdrant vector database containing an embedding vector and metadata
    """
    id: str
    vector: list
    payload: dict
    collection: str


@dataclass
class RetrievalResult:
    """
    A single result from a RAG retrieval query
    """
    rank: int
    score: float
    source_url: str
    chunk_id: str
    content: str
    content_preview: str


@dataclass
class ValidationResult:
    """
    Result of a validation test query
    """
    query: str
    passed: bool
    top_score: float
    top_source_url: str
    result_count: int
