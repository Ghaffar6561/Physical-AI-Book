import asyncio
import sys
import os
from typing import Dict, List, Optional
from ..models.chat_models import SourceCitation
from ..logging_config import logger
from ..utils.performance_monitoring import monitor_performance

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RAGService:
    """
    Service class to handle RAG (Retrieval-Augmented Generation) operations
    Connects to the actual RAG agent for real answers
    """

    def __init__(self):
        self._agent = None
        self._config = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of the agent."""
        if self._initialized:
            return

        try:
            # Import agent components
            from backend.agent import create_agent, AgentConfig, search_book_content, init_retrieval_clients

            self._config = AgentConfig.from_env()
            self._agent = create_agent(self._config)

            # Initialize retrieval clients
            init_retrieval_clients()

            self._search_book_content = search_book_content
            self._initialized = True
            logger.info("RAG Service initialized successfully with real agent")
        except Exception as e:
            logger.warning(f"Failed to initialize real RAG agent: {e}. Using fallback mode.")
            self._initialized = False

    @monitor_performance
    async def process_question(
        self,
        message: str,
        selected_text: Optional[str] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Process a question using the RAG system and return an answer with sources
        """
        try:
            # Implement timeout handling
            try:
                result = await asyncio.wait_for(
                    self._process_question_internal(message, selected_text, top_k),
                    timeout=15.0  # 15 second timeout
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"Timeout processing question: {message[:50]}...")
                raise Exception("Request timeout: The system is taking too long to respond")
        except Exception as e:
            logger.error(f"Error in RAG service: {str(e)}")
            raise

    async def _process_question_internal(
        self,
        message: str,
        selected_text: Optional[str],
        top_k: int
    ) -> Dict:
        """
        Internal method to process the question using the real RAG agent
        """
        # Try to use the real agent
        try:
            self._ensure_initialized()

            if self._initialized and self._agent:
                # Run the real agent in a thread pool to not block
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._run_real_agent,
                    message,
                    selected_text,
                    top_k
                )
                return result
        except Exception as e:
            logger.error(f"Real agent failed, using fallback: {e}")

        # Fallback to basic retrieval if agent fails
        return await self._fallback_response(message, selected_text, top_k)

    def _run_real_agent(self, message: str, selected_text: Optional[str], top_k: int) -> Dict:
        """
        Fast direct RAG without agent overhead - searches and generates in one pass
        """
        from backend.agent import search_book_content, init_retrieval_clients
        from openai import OpenAI
        import os

        init_retrieval_clients()

        # Construct query with selected text context if provided
        if selected_text:
            query = f"{selected_text[:300]} {message}"
        else:
            query = message

        # Direct search (faster than agent tool call)
        search_results = search_book_content(query, top_k)

        sources = []
        context_text = ""

        if search_results.get("status") == "success" and search_results.get("results"):
            results = search_results["results"]
            for r in results:
                sources.append(SourceCitation(
                    content=r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                    page_number=r["index"],
                    section_title=self._extract_section_title(r["source_url"]),
                    url=r["source_url"],
                    confidence_score=r["score"]
                ))
                context_text += f"\n\n{r['content'][:500]}"

        # Direct OpenAI completion (faster than agent)
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions about a Physical AI and Robotics book. Answer based only on the provided context. Be concise and direct. Do not include citations or source references."},
                {"role": "user", "content": f"Context from the book:\n{context_text}\n\nQuestion: {message}"}
            ],
            max_tokens=500,
            temperature=0.3
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources,
            "question_id": f"rag-{hash(message) % 10000}"
        }

    def _extract_section_title(self, url: str) -> str:
        """Extract a readable section title from URL."""
        if not url:
            return "Book Content"

        # Extract the last part of the URL path
        parts = url.rstrip('/').split('/')
        if parts:
            title = parts[-1].replace('-', ' ').replace('_', ' ').title()
            return title if title else "Book Content"
        return "Book Content"

    async def _fallback_response(
        self,
        message: str,
        selected_text: Optional[str],
        top_k: int
    ) -> Dict:
        """
        Fallback response when the real agent is not available.
        Attempts basic retrieval without the full agent.
        """
        await asyncio.sleep(0.1)  # Simulate processing

        try:
            # Try basic retrieval without the agent
            from backend.agent import search_book_content, init_retrieval_clients

            init_retrieval_clients()

            # Search for relevant content
            search_query = message
            if selected_text:
                search_query = f"{selected_text[:200]} {message}"

            search_results = search_book_content(search_query, top_k)

            if search_results.get("status") == "success" and search_results.get("results"):
                # Build answer from retrieved content
                results = search_results["results"]

                # Create a simple answer from the content
                content_parts = [r["content"][:300] for r in results[:3]]
                answer = f"Based on the book content:\n\n" + "\n\n".join(content_parts)

                # Create sources
                sources = []
                for r in results:
                    sources.append(SourceCitation(
                        content=r["content"][:200],
                        page_number=r["index"],
                        section_title=self._extract_section_title(r["source_url"]),
                        url=r["source_url"],
                        confidence_score=r["score"]
                    ))

                return {
                    "answer": answer,
                    "sources": sources,
                    "question_id": f"fallback-{hash(message) % 10000}"
                }
        except Exception as e:
            logger.error(f"Fallback retrieval also failed: {e}")

        # Ultimate fallback - return error message
        return {
            "answer": "I'm sorry, I couldn't retrieve information from the book at this time. Please make sure the RAG system is properly configured with OPENAI_API_KEY, QDRANT_URL, and QDRANT_API_KEY environment variables.",
            "sources": [],
            "question_id": "error"
        }
