"""
RAG Agent for Physical AI Book

An interactive CLI agent that answers questions about the Physical AI book
using OpenAI Agent SDK as the reasoning engine and Qdrant for semantic search.
The agent enforces strict grounding: answers derive from retrieved content
with citations, and questions outside the book's scope are handled gracefully.

Usage:
    python agent.py                        # Interactive mode
    python agent.py --query "What is ROS2?" # Single question mode
    python agent.py --query "..." --top-k 3 # With custom retrieval count
    python agent.py --verbose              # Show retrieval details
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from agents import Agent, Runner

# Load environment variables from .env file
load_dotenv()

# Fix Windows console encoding for Unicode output
# Note: Python 3.7+ supports reconfigure() for safe encoding changes
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Ignore if reconfigure fails


# Import existing retrieval infrastructure from Spec 2
from retrieve import (
    get_clients,
    validate_connection,
    generate_query_embedding,
    search_qdrant,
)
from rag_models import RetrievalResult
from exceptions import RetrievalError


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration specific to the RAG agent."""
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    top_k: int = 3  # Reduced for faster responses
    score_threshold: float = 0.0
    max_conversation_turns: int = 50

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        return cls(
            openai_api_key=api_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            top_k=int(os.getenv("AGENT_TOP_K", "5")),
            score_threshold=float(os.getenv("AGENT_SCORE_THRESHOLD", "0.0")),
            max_conversation_turns=int(os.getenv("AGENT_MAX_TURNS", "50")),
        )


# =============================================================================
# System Prompt - Strict Grounding Rules (per research.md section 6)
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the Physical AI book.

IMPORTANT RULES:
1. ONLY use information from the search_book_content tool to answer questions.
2. If the tool returns no results or irrelevant content, say "I couldn't find relevant information about that in the book."
3. NEVER make up information. If unsure, acknowledge the limitation.
4. NEVER include citations like [1], [2], source URLs, or a "Sources" section in your response. The system handles sources separately.
5. Just provide the answer directly without any references or citations.

When answering:
- Search the book content first using the provided tool
- Formulate a clear, direct answer based ONLY on retrieved content
- Do NOT add any citations, references, footnotes, or source lists"""


# =============================================================================
# Retrieval Function (wraps existing retrieve.py)
# =============================================================================

# Global clients (initialized on first use)
_co_client = None
_qdrant_client = None
_config = None
_last_retrieval_results: list[RetrievalResult] = []


def init_retrieval_clients():
    """Initialize Cohere and Qdrant clients."""
    global _co_client, _qdrant_client, _config
    if _co_client is None:
        _co_client, _qdrant_client, _config = get_clients()
        # Validate connection
        if not validate_connection(_qdrant_client, _config.QDRANT_COLLECTION_NAME):
            raise RetrievalError(f"Collection '{_config.QDRANT_COLLECTION_NAME}' not found. Run ingestion first.")


def search_book_content(query: str, top_k: int = 5) -> dict:
    """
    Search the Physical AI book for relevant content.

    Args:
        query: The search query
        top_k: Number of results to return

    Returns:
        dict with status, results, and optional message
    """
    global _last_retrieval_results

    if not query or not query.strip():
        return {
            "status": "error",
            "message": "Query cannot be empty",
            "results": []
        }

    try:
        init_retrieval_clients()

        # Generate embedding and search
        embedding = generate_query_embedding(_co_client, query)
        results = search_qdrant(
            _qdrant_client,
            _config.QDRANT_COLLECTION_NAME,
            embedding,
            top_k=top_k
        )

        # Store results for source footer formatting
        _last_retrieval_results = results

        if not results:
            return {
                "status": "no_results",
                "query": query,
                "message": "No relevant content found in the book for this query.",
                "results": []
            }

        # Format results for agent context
        formatted_results = []
        for i, r in enumerate(results, 1):
            formatted_results.append({
                "index": i,
                "score": round(r.score, 2),
                "source_url": r.source_url,
                "content": r.content
            })

        return {
            "status": "success",
            "query": query,
            "result_count": len(results),
            "results": formatted_results
        }

    except RetrievalError as e:
        return {
            "status": "error",
            "message": str(e),
            "results": []
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Retrieval failed: {e}",
            "results": []
        }


def format_sources_footer() -> str:
    """Format the last retrieval results as a sources footer."""
    global _last_retrieval_results

    if not _last_retrieval_results:
        return ""

    lines = ["", "---", "**Sources:**"]
    for i, r in enumerate(_last_retrieval_results, 1):
        lines.append(f"[{i}] {r.source_url} (score: {r.score:.2f})")

    return "\n".join(lines)


# =============================================================================
# Agent Core
# =============================================================================

from agents.tool import function_tool

# Define the search_book_content function as a tool for the Agent SDK
@function_tool
def search_book_content_tool(query: str, top_k: int = 5) -> dict:
    """
    Search the Physical AI book for content relevant to the user's question.
    Returns chunks of text with source URLs and relevance scores.
    Use this tool to find information before answering any question about the book.

    Args:
        query: The search query to find relevant content. Should be a clear, specific question or topic.
        top_k: Number of results to return. More results provide broader context but may include less relevant content.
    """
    return search_book_content(query, top_k)

def create_agent(config: AgentConfig):
    """Create an agent with the specified configuration."""
    # Create the agent with instructions and tools
    agent = Agent(
        name="PhysicalAIBookAssistant",
        instructions=SYSTEM_PROMPT,
        tools=[search_book_content_tool],
        model=config.openai_model,
    )
    return agent


def run_agent_query(agent: Agent, query: str, config: AgentConfig, verbose: bool = False) -> str:
    """
    Process a single query and return the agent's response.

    Args:
        agent: The agent to run
        query: The user's question
        config: Agent configuration
        verbose: If True, print debug info to stderr

    Returns:
        The agent's response text
    """
    global _last_retrieval_results
    _last_retrieval_results = []  # Reset for each turn

    if verbose:
        print(f"[DEBUG] User query: \"{query}\"", file=sys.stderr)

    # Run the agent with the query
    try:
        result = Runner.run_sync(
            agent,
            query,
        )

        response_content = result.final_output

        if verbose:
            print(f"[DEBUG] Response generated", file=sys.stderr)

        return response_content

    except Exception as e:
        error_msg = f"Agent error: {e}"
        if verbose:
            print(f"[DEBUG] {error_msg}", file=sys.stderr)
        return f"I'm sorry, I encountered an error: {error_msg}"


# =============================================================================
# CLI Interface (per contracts/cli-interface.md)
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Agent for Physical AI Book",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py                           # Interactive mode
  python agent.py --query "What is ROS2?"   # Single question
  python agent.py -q "..." -k 3             # With top-k
  python agent.py --verbose                 # Show debug info
        """
    )

    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Single question to ask (exits after response)"
    )

    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (shows tool calls)"
    )

    return parser.parse_args()


def main():
    """Main entry point for the RAG Agent CLI."""
    args = parse_args()

    # Load configuration
    try:
        config = AgentConfig.from_env()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Create the agent
    agent = create_agent(config)

    # Single-shot mode
    if args.query:
        query = args.query.strip()
        if not query:
            print("Error: Please provide a question to ask about the book.")
            sys.exit(1)

        if args.verbose:
            print(f"[DEBUG] Single-shot mode with model: {config.openai_model}", file=sys.stderr)

        response = run_agent_query(agent, query, config, verbose=args.verbose)
        print(f"\nAgent: {response}")
        return

    # Interactive mode
    print("RAG Agent ready. Type 'quit' to exit.\n")

    if args.verbose:
        print(f"[DEBUG] Interactive mode with model: {config.openai_model}", file=sys.stderr)

    exit_commands = {"quit", "exit", "q"}

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                print("Please provide a question to ask about the book.")
                continue

            if user_input.lower() in exit_commands:
                break

            # Run the agent (without session since Session is a Protocol)
            try:
                result = Runner.run_sync(
                    agent,
                    user_input
                )

                response_content = result.final_output

                print(f"\nAgent: {response_content}\n")
            except Exception as e:
                error_msg = f"Agent error: {e}"
                if args.verbose:
                    print(f"[DEBUG] {error_msg}", file=sys.stderr)
                print(f"\nAgent: I'm sorry, I encountered an error: {error_msg}\n")

    except KeyboardInterrupt:
        pass

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
