"""
URL discovery and content fetching module for the RAG Ingestion Pipeline
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from typing import List
from rag_models import BookContent
from exceptions import CrawlerError
import hashlib
from datetime import datetime


logger = logging.getLogger(__name__)


def discover_urls(base_url: str) -> List[str]:
    """
    Discover all URLs in the Docusaurus book by checking sitemap.xml first,
    then crawling from the base URL if needed.
    
    Args:
        base_url: Base URL of the Docusaurus book
        
    Returns:
        List of discovered URLs
    """
    urls = set()
    
    # Try to get URLs from sitemap first
    sitemap_url = urljoin(base_url, 'sitemap.xml')
    try:
        response = requests.get(sitemap_url, timeout=10)
        if response.status_code == 200:
            logger.info(f"Found sitemap at {sitemap_url}")
            urls.update(_parse_sitemap(response.text))
        else:
            logger.info(f"No sitemap found at {sitemap_url}, will crawl from base URL")
    except requests.RequestException as e:
        logger.warning(f"Could not fetch sitemap: {e}")
    
    # If no URLs found from sitemap, crawl from base URL
    if not urls:
        logger.info(f"Crawling from base URL: {base_url}")
        urls.update(_crawl_from_base_url(base_url))
    
    return list(urls)


def _parse_sitemap(sitemap_content: str) -> List[str]:
    """
    Parse URLs from sitemap XML content
    """
    urls = []
    try:
        soup = BeautifulSoup(sitemap_content, 'xml')
        for loc in soup.find_all('loc'):
            url = loc.get_text().strip()
            if url:
                urls.append(url)
    except Exception as e:
        logger.error(f"Error parsing sitemap: {e}")
    
    return urls


def _crawl_from_base_url(base_url: str) -> List[str]:
    """
    Crawl from the base URL to discover all internal links
    """
    urls = set()
    to_visit = [base_url]
    visited = set()
    
    while to_visit:
        current_url = to_visit.pop(0)
        
        # Skip if already visited
        if current_url in visited:
            continue
        
        visited.add(current_url)
        
        try:
            response = requests.get(current_url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {current_url}: Status {response.status_code}")
                continue
            
            # Add current URL to the list
            urls.add(current_url)
            
            # Parse links from the page
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(current_url, href)
                
                # Only add URLs from the same domain
                if _is_same_domain(base_url, absolute_url):
                    if absolute_url not in visited and absolute_url not in to_visit:
                        to_visit.append(absolute_url)
                        
        except requests.RequestException as e:
            logger.error(f"Error crawling {current_url}: {e}")
            continue
    
    return list(urls)


def _is_same_domain(base_url: str, url: str) -> bool:
    """
    Check if the URL is from the same domain as the base URL
    """
    base_domain = urlparse(base_url).netloc
    url_domain = urlparse(url).netloc
    return base_domain == url_domain


def fetch_content(url: str) -> str:
    """
    Fetch and return the HTML content of a URL
    
    Args:
        url: URL to fetch content from
        
    Returns:
        HTML content as string
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch content from {url}: {e}")
        raise CrawlerError(f"Could not fetch content from {url}: {e}")


def crawl_and_extract_content(base_url: str = None, urls: List[str] = None) -> List[BookContent]:
    """
    Main function to crawl and extract content from book URLs
    
    Args:
        base_url: Base URL of the Docusaurus book (optional if urls provided)
        urls: List of specific URLs to process (optional if base_url provided)
        
    Returns:
        List of BookContent objects
    """
    if not urls and not base_url:
        raise CrawlerError("Either base_url or urls must be provided")
    
    if not urls:
        urls = discover_urls(base_url)
    
    book_contents = []
    for url in urls:
        try:
            html_content = fetch_content(url)
            # Extract title and text content
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.title.string if soup.title else "No Title"
            
            # Extract text content, focusing on main content areas
            text_content = _extract_text_content(soup)
            
            if text_content.strip():  # Only add if content is not empty
                # Calculate checksum
                checksum = hashlib.sha256(text_content.encode()).hexdigest()
                
                book_content = BookContent(
                    url=url,
                    title=title.strip(),
                    content=text_content,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    checksum=checksum
                )
                book_contents.append(book_content)
            else:
                logger.warning(f"No content extracted from {url}")
                
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            continue  # Continue with other URLs
    
    return book_contents


def _extract_text_content(soup: BeautifulSoup) -> str:
    """
    Extract clean text content from BeautifulSoup object, focusing on main content
    """
    # Try to find Docusaurus-specific content containers
    content_selectors = [
        'main.docMainContainer',  # Common Docusaurus container
        'article.markdown',       # Docusaurus markdown articles
        'main',                  # General main content
        '.container',            # General container
        '.content'               # General content class
    ]
    
    content_element = None
    for selector in content_selectors:
        content_element = soup.select_one(selector)
        if content_element:
            break
    
    if content_element:
        # Remove script and style elements
        for script in content_element(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = content_element.get_text(separator=' ')
    else:
        # Fallback to body content if specific containers not found
        body = soup.body
        if body:
            # Remove script and style elements
            for script in body(["script", "style"]):
                script.decompose()
            
            text = body.get_text(separator=' ')
        else:
            text = soup.get_text(separator=' ')
    
    # Clean up the text: remove extra whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text