"""
RAG Retrieval Pipeline Validation

Query stored embeddings in Qdrant and validate the RAG ingestion pipeline.
Supports single-query mode, filtered retrieval, and batch validation.

Usage:
    python retrieve.py --query "What is ROS2?"
    python retrieve.py --query "locomotion" --filter "module-3" --top-k 3
    python retrieve.py --validate
"""

import argparse
import sys
import io
from typing import Optional

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from config import Config
from rag_models import RetrievalResult, ValidationResult
from exceptions import RetrievalError


# Predefined test queries for validation mode
TEST_QUERIES = [
    {"query": "What is ROS2?", "expected_module": "module-1-ros2"},
    {"query": "How does Gazebo simulation work?", "expected_module": "module-2"},
    {"query": "What is Isaac Sim?", "expected_module": "module-3-isaac"},
    {"query": "How do vision language models work?", "expected_module": "module-4-vla"},
    {"query": "What is bipedal locomotion?", "expected_module": "module-3"},
]

# Validation threshold - queries must score above this to pass
VALIDATION_THRESHOLD = 0.5


def get_clients() -> tuple[cohere.Client, QdrantClient, Config]:
    """Initialize Cohere and Qdrant clients with config from environment."""
    config = Config()
    co_client = cohere.Client(config.COHERE_API_KEY)
    qdrant_client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )
    return co_client, qdrant_client, config


def validate_connection(qdrant_client: QdrantClient, collection_name: str) -> bool:
    """Check if Qdrant collection exists and is accessible."""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        return collection_name in collection_names
    except Exception as e:
        raise RetrievalError(f"Cannot connect to Qdrant: {e}")


def generate_query_embedding(co_client: cohere.Client, query: str) -> list[float]:
    """Generate embedding for a search query using Cohere."""
    try:
        response = co_client.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query",
        )
        return response.embeddings[0]
    except Exception as e:
        raise RetrievalError(f"Cohere API failed: {e}")


def search_qdrant(
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding: list[float],
    top_k: int = 5,
    filter_prefix: Optional[str] = None,
) -> list[RetrievalResult]:
    """
    Search Qdrant for similar vectors and return ranked results.

    Args:
        qdrant_client: Qdrant client instance
        collection_name: Name of the collection to search
        embedding: Query embedding vector
        top_k: Number of results to return
        filter_prefix: Optional URL path prefix to filter results

    Returns:
        List of RetrievalResult objects sorted by score
    """
    # Build filter if prefix is provided (requires text index on source_url)
    query_filter = None
    if filter_prefix:
        query_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="source_url",
                    match=qdrant_models.MatchText(text=filter_prefix),
                )
            ]
        )

    try:
        # Use query_points API (qdrant-client >= 1.7)
        response = qdrant_client.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        results = response.points
    except Exception as e:
        raise RetrievalError(f"Qdrant search failed: {e}")

    retrieval_results = []
    for rank, point in enumerate(results, start=1):
        payload = point.payload or {}
        content = payload.get("content", "")
        content_preview = content[:200] + "..." if len(content) > 200 else content

        retrieval_results.append(
            RetrievalResult(
                rank=rank,
                score=point.score,
                source_url=payload.get("source_url", "unknown"),
                chunk_id=payload.get("chunk_id", "unknown"),
                content=content,
                content_preview=content_preview,
            )
        )

    return retrieval_results


def format_results(query: str, results: list[RetrievalResult]) -> str:
    """Format retrieval results for CLI output."""
    if not results:
        return f'No results found for query: "{query}"'

    lines = [
        f'Query: "{query}"',
        f"Results ({len(results)} found):",
        "",
    ]

    for result in results:
        # Shorten source URL for display
        source_display = result.source_url
        if len(source_display) > 60:
            source_display = "..." + source_display[-57:]

        lines.append(f"[{result.rank}] Score: {result.score:.2f} | Source: {source_display}")
        lines.append(f"    {result.content_preview}")
        lines.append("")

    return "\n".join(lines)


def run_validation(
    co_client: cohere.Client,
    qdrant_client: QdrantClient,
    collection_name: str,
    top_k: int = 5,
) -> list[ValidationResult]:
    """Run predefined test queries and return validation results."""
    validation_results = []

    for test in TEST_QUERIES:
        query = test["query"]

        try:
            embedding = generate_query_embedding(co_client, query)
            results = search_qdrant(
                qdrant_client, collection_name, embedding, top_k=top_k
            )

            if results:
                top_result = results[0]
                passed = top_result.score >= VALIDATION_THRESHOLD
                validation_results.append(
                    ValidationResult(
                        query=query,
                        passed=passed,
                        top_score=top_result.score,
                        top_source_url=top_result.source_url,
                        result_count=len(results),
                    )
                )
            else:
                validation_results.append(
                    ValidationResult(
                        query=query,
                        passed=False,
                        top_score=0.0,
                        top_source_url="no results",
                        result_count=0,
                    )
                )
        except RetrievalError as e:
            validation_results.append(
                ValidationResult(
                    query=query,
                    passed=False,
                    top_score=0.0,
                    top_source_url=f"error: {e}",
                    result_count=0,
                )
            )

    return validation_results


def format_validation_results(results: list[ValidationResult]) -> str:
    """Format validation results for CLI output."""
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    lines = [
        f"Validation Results ({passed_count}/{total_count} passed):",
        "",
    ]

    for result in results:
        # Extract module from source URL for display
        source_display = result.top_source_url
        if "/" in source_display:
            parts = source_display.split("/")
            # Find the module part (e.g., "module-1-ros2")
            for part in parts:
                if part.startswith("module-"):
                    source_display = part
                    break

        if result.passed:
            lines.append(
                f'[PASS] "{result.query}" - Score: {result.top_score:.2f}, Source: {source_display}'
            )
        else:
            if result.result_count == 0:
                lines.append(f'[FAIL] "{result.query}" - No results found')
            else:
                lines.append(
                    f'[FAIL] "{result.query}" - Score: {result.top_score:.2f} (below threshold {VALIDATION_THRESHOLD})'
                )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Retrieval Pipeline Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python retrieve.py --query "What is ROS2?"
  python retrieve.py -q "locomotion" -f "module-3" -k 3
  python retrieve.py --validate
        """,
    )

    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Natural language query to search for",
    )

    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )

    parser.add_argument(
        "-f", "--filter",
        type=str,
        help="Filter results by source URL prefix (e.g., 'module-3')",
    )

    parser.add_argument(
        "-v", "--validate",
        action="store_true",
        help="Run predefined test queries for validation",
    )

    return parser.parse_args()


def main():
    """Main entry point for the retrieval CLI."""
    args = parse_args()

    # Validate arguments
    if not args.validate and not args.query:
        print("Error: Either --query or --validate is required")
        sys.exit(1)

    if args.query and not args.query.strip():
        print("Error: Query cannot be empty")
        sys.exit(1)

    # Initialize clients
    try:
        co_client, qdrant_client, config = get_clients()
    except ValueError as e:
        print(f"Error: Configuration error - {e}")
        sys.exit(1)

    # Validate connection
    try:
        if not validate_connection(qdrant_client, config.QDRANT_COLLECTION_NAME):
            print(f"Error: Collection '{config.QDRANT_COLLECTION_NAME}' not found. Run ingestion first.")
            sys.exit(1)
    except RetrievalError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Run validation mode
    if args.validate:
        try:
            results = run_validation(
                co_client, qdrant_client, config.QDRANT_COLLECTION_NAME, args.top_k
            )
            print(format_validation_results(results))

            # Exit with error if any validation failed
            if not all(r.passed for r in results):
                sys.exit(1)
        except RetrievalError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Run single query mode
    else:
        try:
            embedding = generate_query_embedding(co_client, args.query)
            results = search_qdrant(
                qdrant_client,
                config.QDRANT_COLLECTION_NAME,
                embedding,
                top_k=args.top_k,
                filter_prefix=args.filter,
            )
            print(format_results(args.query, results))
        except RetrievalError as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
