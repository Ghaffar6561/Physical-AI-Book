"""
Text extraction module for the RAG Ingestion Pipeline
"""
import logging
from bs4 import BeautifulSoup
from typing import List
from exceptions import ExtractionError


logger = logging.getLogger(__name__)


def extract_text(html_content: str, source_url: str) -> str:
    """
    Extract clean text from HTML content
    
    Args:
        html_content: Raw HTML content
        source_url: Source URL for context (used for logging)
        
    Returns:
        Clean text content as string
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Focus on main content areas, especially for Docusaurus
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='container') or soup
        
        # Get text content
        text = main_content.get_text(separator=' ')
        
        # Clean up the text: remove extra whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        if not text.strip():
            logger.warning(f"No text extracted from {source_url}")
            return ""
        
        logger.info(f"Extracted {len(text)} characters from {source_url}")
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from {source_url}: {e}")
        raise ExtractionError(f"Could not extract text from {source_url}: {e}")