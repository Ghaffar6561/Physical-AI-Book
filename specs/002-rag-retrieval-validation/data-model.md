# Data Model: RAG Retrieval Pipeline Validation

**Feature**: 002-rag-retrieval-validation
**Date**: 2024-12-27

## Entities

### RetrievalResult

Represents a single search result from Qdrant vector search.

| Field | Type | Description |
|-------|------|-------------|
| rank | int | 1-indexed position in results |
| score | float | Cosine similarity score (0.0 - 1.0) |
| source_url | str | Full URL of the source page |
| content | str | Complete chunk text |
| chunk_id | str | Unique SHA256 hash identifier |
| content_preview | str | First 200 characters for display |

**Derivation**: `content_preview` is derived from `content[:200] + "..."` if truncated.

---

### ValidationResult

Represents the outcome of a single test query in validation mode.

| Field | Type | Description |
|-------|------|-------------|
| query | str | The test query text |
| passed | bool | True if top score > threshold (0.5) |
| top_score | float | Highest similarity score returned |
| top_source_url | str | Source URL of top result |
| result_count | int | Number of results returned |

---

### TestQuery (Configuration)

Predefined test queries for batch validation.

| Field | Type | Description |
|-------|------|-------------|
| query | str | Natural language query |
| expected_module | str | Expected URL path prefix (e.g., "module-1-ros2") |

**Hardcoded values**:

```python
TEST_QUERIES = [
    {"query": "What is ROS2?", "expected_module": "module-1-ros2"},
    {"query": "How does Gazebo simulation work?", "expected_module": "module-2-digital-twin"},
    {"query": "What is Isaac Sim?", "expected_module": "module-3-isaac"},
    {"query": "How do vision language models work?", "expected_module": "module-4-vla"},
    {"query": "What is bipedal locomotion?", "expected_module": "module-3-isaac"},
]
```

---

## Existing Entities (from Spec 1)

### Qdrant Payload Structure

The stored vectors have this payload structure (set in Spec 1):

```python
{
    "chunk_id": str,      # SHA256 hash
    "model": str,         # "embed-english-v3.0"
    "created_at": str,    # ISO timestamp
    "source_url": str,    # Full page URL
    "content": str,       # Complete chunk text
}
```

---

## Relationships

```
TestQuery --(executed as)--> Query Embedding --(searches)--> Qdrant Collection
                                                                    |
                                                                    v
                                                           RetrievalResult[]
                                                                    |
                                                                    v
                                                           ValidationResult
```

---

## State Transitions

This feature is stateless - no persistent state beyond the existing Qdrant collection.

| Operation | Input | Output |
|-----------|-------|--------|
| Single Query | query string | RetrievalResult[] |
| Filtered Query | query + filter prefix | RetrievalResult[] |
| Batch Validation | (none) | ValidationResult[] |

---

## Validation Rules

1. **Query**: Must be non-empty, non-whitespace string
2. **top_k**: Must be positive integer (default: 5, max: 100)
3. **filter**: If provided, must match at least one stored source_url
4. **score threshold**: 0.5 for validation pass/fail determination
