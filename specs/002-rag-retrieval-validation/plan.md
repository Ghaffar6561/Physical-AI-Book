# Implementation Plan: RAG Retrieval Pipeline Validation

**Branch**: `002-rag-retrieval-validation` | **Date**: 2024-12-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-rag-retrieval-validation/spec.md`

## Summary

Build a CLI tool (`retrieve.py`) that validates the RAG ingestion pipeline by querying stored embeddings in Qdrant and returning relevant book chunks. The tool supports single-query mode, filtered retrieval, and batch validation with predefined test queries.

## Technical Context

**Language/Version**: Python 3.11+ (matches Spec 1 ingestion pipeline)
**Primary Dependencies**: cohere, qdrant-client, python-dotenv, argparse (stdlib)
**Storage**: Qdrant Cloud (existing collection from Spec 1)
**Testing**: Manual CLI validation + predefined test queries
**Target Platform**: Local CLI (Windows/Linux/macOS)
**Project Type**: Single module addition to existing backend/
**Performance Goals**: Query response under 5 seconds
**Constraints**: Reuse existing config.py, models.py from Spec 1
**Scale/Scope**: 152 vectors in collection, 5+ test queries

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Gate | Status | Notes |
|------|--------|-------|
| Library-First | ✅ Pass | Single retrieve.py module, self-contained |
| CLI Interface | ✅ Pass | CLI with argparse, text output |
| Test-First | ✅ Pass | Batch validation mode serves as built-in tests |
| Simplicity | ✅ Pass | Single file, reuses existing infrastructure |

## Project Structure

### Documentation (this feature)

```text
specs/002-rag-retrieval-validation/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/sp.tasks command)
```

### Source Code (repository root)

```text
backend/
├── config.py            # [EXISTING] Reuse for Qdrant/Cohere credentials
├── models.py            # [EXISTING] Reuse EmbeddingVector, add RetrievalResult
├── exceptions.py        # [EXISTING] Add RetrievalError
├── retrieve.py          # [NEW] Main retrieval module
└── .env                 # [EXISTING] Contains all credentials
```

**Structure Decision**: Single file addition to existing backend/ directory. Reuses config.py for credentials and models.py for data structures.

## Complexity Tracking

> No violations - simple single-file addition following existing patterns.

---

## Phase 0: Research

### Research Topics

1. **Cohere Query Embedding Best Practices**
   - Decision: Use `input_type="search_query"` for query embeddings
   - Rationale: Cohere recommends different input types for documents vs queries for optimal retrieval
   - Alternatives: Using same "search_document" type - rejected as Cohere docs explicitly recommend asymmetric types

2. **Qdrant Filter Syntax**
   - Decision: Use `models.Filter` with `FieldCondition` and `MatchText` for prefix matching
   - Rationale: Qdrant's native filtering on payload fields is efficient and well-documented
   - Alternatives: Post-filtering in Python - rejected as inefficient for large collections

3. **CLI Argument Parsing**
   - Decision: Use stdlib `argparse` for CLI interface
   - Rationale: No external dependencies, sufficient for query/k/filter/validate options
   - Alternatives: click, typer - rejected as overkill for simple CLI

### Key Findings

- **Cohere embed-english-v3.0**: 1024 dimensions, supports "search_query" input type
- **Qdrant search**: `client.search()` returns ScoredPoint with id, score, payload
- **Payload structure** (from Spec 1): `source_url`, `content`, `chunk_id`, `model`, `created_at`

---

## Phase 1: Design

### Data Model

```python
@dataclass
class RetrievalResult:
    """Single search result from Qdrant"""
    rank: int                # 1-indexed position
    score: float             # Similarity score (0-1)
    source_url: str          # Page URL
    content: str             # Full chunk text
    chunk_id: str            # Unique chunk identifier
    content_preview: str     # First 200 chars for display

@dataclass
class ValidationResult:
    """Result of a single test query"""
    query: str
    passed: bool             # True if score > threshold
    top_score: float
    top_source_url: str
    result_count: int
```

### API Contract (CLI Interface)

```text
# Single query mode
python backend/retrieve.py --query "What is ROS2?" [--top-k 5] [--filter "module-1"]

# Batch validation mode
python backend/retrieve.py --validate [--top-k 5]

Arguments:
  --query, -q     Natural language query (required unless --validate)
  --top-k, -k     Number of results to return (default: 5)
  --filter, -f    Filter by source_url path prefix (optional)
  --validate, -v  Run predefined test queries

Output Format (single query):
  Query: "What is ROS2?"
  Results (5 found):

  [1] Score: 0.87 | Source: .../module-1-ros2/intro
      ROS2 (Robot Operating System 2) is the next generation...

  [2] Score: 0.82 | Source: .../module-1-ros2/architecture
      The ROS2 architecture provides improved real-time...

Output Format (validation):
  Validation Results (5/5 passed):

  ✓ "What is ROS2?" - Score: 0.87, Source: module-1-ros2/intro
  ✓ "How does Gazebo simulation work?" - Score: 0.79, Source: module-2-digital-twin/gazebo
  ✗ "What is quantum computing?" - Score: 0.31 (below threshold 0.5)
```

### Predefined Test Queries

```python
TEST_QUERIES = [
    {"query": "What is ROS2?", "expected_module": "module-1-ros2"},
    {"query": "How does Gazebo simulation work?", "expected_module": "module-2-digital-twin"},
    {"query": "What is Isaac Sim?", "expected_module": "module-3-isaac"},
    {"query": "How do vision language models work?", "expected_module": "module-4-vla"},
    {"query": "What is bipedal locomotion?", "expected_module": "module-3-isaac"},
]
```

### Error Handling

| Error Condition | Message | Exit Code |
|-----------------|---------|-----------|
| Empty query | "Error: Query cannot be empty" | 1 |
| Qdrant connection failed | "Error: Cannot connect to Qdrant. Check QDRANT_URL and QDRANT_API_KEY" | 1 |
| Collection not found | "Error: Collection 'physical-ai-book' not found. Run ingestion first." | 1 |
| Cohere API error | "Error: Cohere API failed: {details}" | 1 |
| No results found | "No results found for query: {query}" | 0 |

---

## Implementation Tasks (High-Level)

1. **Add RetrievalResult model** to models.py
2. **Add RetrievalError** to exceptions.py
3. **Create retrieve.py** with:
   - `generate_query_embedding(query: str) -> list[float]`
   - `search_qdrant(embedding, top_k, filter_prefix) -> list[RetrievalResult]`
   - `format_results(results) -> str`
   - `run_validation() -> list[ValidationResult]`
   - `main()` with argparse CLI
4. **Test manually** with sample queries
5. **Run validation mode** to verify all test queries pass

---

## Definition of Done

- [ ] `python backend/retrieve.py --query "What is ROS2?"` returns relevant chunks
- [ ] Results include score, source_url, and content preview
- [ ] `--filter` option restricts results to matching URLs
- [ ] `--validate` runs 5+ test queries with pass/fail summary
- [ ] All predefined test queries return score > 0.5
- [ ] Error messages are clear and actionable
