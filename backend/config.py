"""
Configuration management for the RAG Ingestion Pipeline
Loads settings from environment variables with sensible defaults
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Configuration class that loads settings from environment variables
    """
    def __init__(self):
        # Book configuration
        self.BOOK_BASE_URL = os.getenv('BOOK_BASE_URL', 'https://example-book.com')
        
        # Cohere API configuration
        self.COHERE_API_KEY = os.getenv('COHERE_API_KEY')
        if not self.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY environment variable must be set")
        
        # Qdrant configuration
        self.QDRANT_URL = os.getenv('QDRANT_URL')
        if not self.QDRANT_URL:
            raise ValueError("QDRANT_URL environment variable must be set")
        
        self.QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
        if not self.QDRANT_API_KEY:
            raise ValueError("QDRANT_API_KEY environment variable must be set")
        
        self.QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'book_embeddings')
        
        # Chunking configuration
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '512'))
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '51'))
        
        # Batch size for Cohere API calls
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
        
        # Validate configuration values
        self._validate_config()
    
    def _validate_config(self):
        """
        Validate that all required configuration values are set correctly
        """
        # Only validate BOOK_BASE_URL if it's not empty and not a placeholder
        if self.BOOK_BASE_URL and not self.BOOK_BASE_URL.startswith('# Add your actual'):
            if not self.BOOK_BASE_URL.startswith(('http://', 'https://')):
                raise ValueError(f"BOOK_BASE_URL must be a valid URL: {self.BOOK_BASE_URL}")

        if self.CHUNK_SIZE <= 0:
            raise ValueError(f"CHUNK_SIZE must be positive: {self.CHUNK_SIZE}")

        if self.CHUNK_OVERLAP < 0 or self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(f"CHUNK_OVERLAP must be between 0 and CHUNK_SIZE-1: {self.CHUNK_OVERLAP}")

        if self.BATCH_SIZE <= 0:
            raise ValueError(f"BATCH_SIZE must be positive: {self.BATCH_SIZE}")