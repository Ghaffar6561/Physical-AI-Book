# Research: RAG Retrieval Pipeline Validation

**Feature**: 002-rag-retrieval-validation
**Date**: 2024-12-27

## Research Topics

### 1. Cohere Query Embedding Best Practices

**Question**: Should query embeddings use the same `input_type` as document embeddings?

**Finding**: No - Cohere recommends asymmetric embedding types for optimal retrieval:
- Documents: `input_type="search_document"`
- Queries: `input_type="search_query"`

**Evidence**: Cohere documentation states that using different input types for documents and queries improves retrieval quality by optimizing the embedding space for the specific use case.

**Decision**: Use `input_type="search_query"` for all query embeddings in retrieve.py.

---

### 2. Qdrant Filter Syntax for Payload Fields

**Question**: How to filter Qdrant results by `source_url` prefix (e.g., "module-1-ros2")?

**Finding**: Qdrant supports payload filtering using `models.Filter` with conditions:

```python
from qdrant_client.http import models

# For substring/prefix matching
filter = models.Filter(
    must=[
        models.FieldCondition(
            key="source_url",
            match=models.MatchText(text="module-1-ros2")
        )
    ]
)
```

**Alternative considered**: Post-filtering in Python after retrieval
- Rejected: Inefficient - would need to over-fetch results then filter

**Decision**: Use Qdrant's native `MatchText` filter for prefix matching on `source_url` field.

---

### 3. CLI Argument Parsing Library

**Question**: Which library to use for CLI argument parsing?

**Options Evaluated**:

| Library | Pros | Cons |
|---------|------|------|
| argparse (stdlib) | No dependencies, well-known | Verbose syntax |
| click | Decorators, nice help text | Extra dependency |
| typer | Modern, type hints | Extra dependency, overkill |

**Decision**: Use `argparse` (stdlib) - no external dependencies, sufficient for simple CLI with 4 options.

---

### 4. Existing Payload Structure (from Spec 1)

**Question**: What fields are available in the Qdrant payload?

**Finding** (from storage.py in Spec 1):

```python
payload = {
    "chunk_id": embedding_vector.chunk_id,
    "model": embedding_vector.model,
    "created_at": embedding_vector.created_at.isoformat(),
    "source_url": embedding_vector.source_url,
    "content": embedding_vector.content,
}
```

**Available fields for retrieval**:
- `source_url`: Full URL of the source page
- `content`: Complete chunk text
- `chunk_id`: SHA256 hash identifier
- `model`: Embedding model used ("embed-english-v3.0")
- `created_at`: ISO timestamp

**Decision**: Display `source_url`, `content` (truncated to 200 chars), and `score` in results.

---

### 5. Similarity Score Interpretation

**Question**: What is a "good" similarity score threshold?

**Finding**: Cohere embed-english-v3.0 with cosine similarity:
- 0.8+ : Very high relevance
- 0.6-0.8 : Good relevance
- 0.4-0.6 : Moderate relevance
- <0.4 : Low relevance

**Decision**: Use 0.5 as the threshold for "relevant" results in validation mode. This is conservative enough to catch most relevant content while flagging truly irrelevant queries.

---

## Summary

All research questions resolved. No NEEDS CLARIFICATION items remaining.

| Topic | Decision |
|-------|----------|
| Query embedding type | `input_type="search_query"` |
| Qdrant filtering | `MatchText` on `source_url` field |
| CLI library | stdlib `argparse` |
| Payload fields | `source_url`, `content`, `chunk_id` available |
| Relevance threshold | 0.5 for validation pass/fail |
