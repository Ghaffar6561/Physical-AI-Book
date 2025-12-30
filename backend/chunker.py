"""
Text chunking module for the RAG Ingestion Pipeline
"""
import logging
from typing import List
from rag_models import TextChunk
from config import Config
import hashlib


logger = logging.getLogger(__name__)


def chunk_text(text: str, source_url: str, chunk_size: int = None, chunk_overlap: int = None) -> List[TextChunk]:
    """
    Split text into chunks with specified size and overlap
    
    Args:
        text: Text content to chunk
        source_url: Source URL for the text
        chunk_size: Size of each chunk in tokens (default from config)
        chunk_overlap: Overlap between chunks in tokens (default from config)
        
    Returns:
        List of TextChunk objects
    """
    # Use config values if not provided
    config = Config()
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP
    
    # For this implementation, we'll use a character-based approach
    # In a real implementation, you might want to use tokenization
    
    # Convert token-based size to character-based approximation
    # Assuming average word length of 6 characters + 1 space = 7 characters per token
    avg_char_per_token = 7
    chunk_chars = chunk_size * avg_char_per_token
    overlap_chars = chunk_overlap * avg_char_per_token
    
    # Split text into chunks
    chunks = []
    start = 0
    position = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_chars
        
        # If this is the last chunk, make sure to include all remaining text
        if end >= len(text):
            end = len(text)
        else:
            # Try to break at word boundary
            while end > start and text[end] != ' ' and end < len(text):
                end -= 1
            # If we couldn't find a space, just break at the original position
            if end <= start:
                end = start + chunk_chars
        
        # Extract the chunk content
        chunk_content = text[start:end]
        
        # Calculate overlap with previous and next chunks
        overlap_prev = ""
        if position > 0:
            prev_start = max(0, start - overlap_chars)
            overlap_prev = text[prev_start:start]
        
        overlap_next = ""
        if end < len(text):
            next_end = min(len(text), end + overlap_chars)
            overlap_next = text[end:next_end]
        
        # Create chunk ID based on source URL and position for consistency
        chunk_id = hashlib.sha256(f"{source_url}_{position}".encode()).hexdigest()
        
        # Create TextChunk object
        text_chunk = TextChunk(
            id=chunk_id,
            content=chunk_content,
            source_url=source_url,
            position=position,
            overlap_prev=overlap_prev,
            overlap_next=overlap_next,
            token_count=len(chunk_content.split())  # Approximate token count
        )
        
        chunks.append(text_chunk)
        
        # Move to the next position
        start = end
        position += 1
        
        # If we reached the end, break
        if start >= len(text):
            break
    
    logger.info(f"Chunked text from {source_url} into {len(chunks)} chunks")
    return chunks