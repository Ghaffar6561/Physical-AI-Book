"""
Chat endpoint for Vercel serverless function
RAG-powered Q&A for the Physical AI book
"""
from http.server import BaseHTTPRequestHandler
import json
import os
import re
from typing import Optional


def sanitize_input(text: Optional[str]) -> Optional[str]:
    """Remove potentially harmful content from user input."""
    if not text:
        return None
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove script tags and content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_section_title(url: str) -> str:
    """Extract a readable section title from URL."""
    if not url:
        return "Book Content"
    parts = url.rstrip('/').split('/')
    if parts:
        title = parts[-1].replace('-', ' ').replace('_', ' ').title()
        return title if title else "Book Content"
    return "Book Content"


def search_book_content(query: str, top_k: int = 5) -> dict:
    """
    Search the Physical AI book for relevant content using Qdrant and Cohere.
    """
    import cohere
    from qdrant_client import QdrantClient

    # Get API keys from environment
    cohere_api_key = os.getenv('COHERE_API_KEY')
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'physical-ai-book')

    if not all([cohere_api_key, qdrant_url, qdrant_api_key]):
        return {
            "status": "error",
            "message": "Missing API credentials",
            "results": []
        }

    try:
        # Initialize clients
        co_client = cohere.Client(cohere_api_key)
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        # Generate embedding for query
        response = co_client.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query",
        )
        embedding = response.embeddings[0]

        # Search Qdrant
        search_response = qdrant_client.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for point in search_response.points:
            payload = point.payload or {}
            results.append({
                "index": len(results) + 1,
                "score": round(point.score, 2),
                "source_url": payload.get("source_url", ""),
                "content": payload.get("content", "")
            })

        if not results:
            return {
                "status": "no_results",
                "query": query,
                "message": "No relevant content found in the book.",
                "results": []
            }

        return {
            "status": "success",
            "query": query,
            "result_count": len(results),
            "results": results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "results": []
        }


def generate_answer(message: str, context: str) -> str:
    """Generate an answer using OpenAI based on retrieved context."""
    from openai import OpenAI

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        return "I couldn't generate an answer. OpenAI API key is not configured."

    try:
        client = OpenAI(api_key=openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant answering questions about a Physical AI and Robotics book. Answer based only on the provided context. Be concise and direct. Do not include citations or source references."
                },
                {
                    "role": "user",
                    "content": f"Context from the book:\n{context}\n\nQuestion: {message}"
                }
            ],
            max_tokens=500,
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"I encountered an error generating the answer: {str(e)}"


def process_chat_request(message: str, selected_text: Optional[str] = None, top_k: int = 5) -> dict:
    """Process a chat request using RAG."""
    # Construct query with selected text context if provided
    if selected_text:
        query = f"{selected_text[:300]} {message}"
    else:
        query = message

    # Search for relevant content
    search_results = search_book_content(query, top_k)

    sources = []
    context_text = ""

    if search_results.get("status") == "success" and search_results.get("results"):
        results = search_results["results"]
        for r in results:
            content = r["content"]
            sources.append({
                "content": content[:200] + "..." if len(content) > 200 else content,
                "page_number": r["index"],
                "section_title": extract_section_title(r["source_url"]),
                "url": r["source_url"],
                "confidence_score": r["score"]
            })
            context_text += f"\n\n{content[:500]}"

    # Generate answer
    if context_text:
        answer = generate_answer(message, context_text)
    else:
        answer = "I couldn't find relevant information about that in the book. Please try rephrasing your question."

    return {
        "answer": answer,
        "sources": sources,
        "question_id": f"rag-{hash(message) % 10000}"
    }


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()

    def do_POST(self):
        """Handle chat POST requests."""
        # CORS headers
        origin = self.headers.get('Origin', '*')

        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body) if body else {}

            # Extract and validate parameters
            message = sanitize_input(data.get('message', ''))
            selected_text = sanitize_input(data.get('selected_text'))
            top_k = data.get('top_k', 5)

            # Validate message
            if not message or len(message.strip()) == 0:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', origin)
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "Question message cannot be empty"}).encode())
                return

            # Validate selected_text length
            if selected_text and len(selected_text) > 5000:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', origin)
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "Selected text is too long. Maximum allowed length is 5000 characters."}).encode())
                return

            # Validate top_k
            if top_k < 1 or top_k > 20:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', origin)
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "top_k must be between 1 and 20"}).encode())
                return

            # Process the chat request
            result = process_chat_request(message, selected_text, top_k)

            # Send success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', origin)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"detail": "Invalid JSON in request body"}).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"detail": "An error occurred while processing your request"}).encode())
