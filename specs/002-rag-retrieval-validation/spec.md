# Feature Specification: RAG Retrieval Pipeline Validation

**Feature Branch**: `002-rag-retrieval-validation`
**Created**: 2024-12-27
**Status**: Draft
**Input**: User description: "Retrieve stored chunks from Qdrant and validate the RAG retrieval pipeline"

## Overview

This feature provides a command-line tool for developers to validate that the RAG ingestion pipeline (Spec 1) produced correct, searchable vectors. The tool queries Qdrant using embedded user queries and returns relevant book chunks with metadata, enabling end-to-end validation of the retrieval pipeline.

**Target Audience**: Developers validating ingestion quality and retrieval relevance.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Basic Query Retrieval (Priority: P1)

As a developer, I want to run a natural language query against the stored book content so that I can verify the retrieval pipeline returns relevant chunks.

**Why this priority**: Core functionality - without basic query capability, no validation is possible.

**Independent Test**: Can be tested by running a single CLI command with a query and observing returned results.

**Acceptance Scenarios**:

1. **Given** the Qdrant collection contains ingested book vectors, **When** a developer runs a query command with "What is ROS2?", **Then** the system returns top-k relevant chunks with scores, source URLs, and text snippets.

2. **Given** the Qdrant collection is empty or inaccessible, **When** a developer runs a query command, **Then** the system displays a clear error message explaining the issue.

3. **Given** a valid query, **When** the system generates the query embedding, **Then** it uses the same Cohere model (embed-english-v3.0) and input type as ingestion to ensure compatibility.

---

### User Story 2 - Filtered Retrieval (Priority: P2)

As a developer, I want to filter retrieval results by source URL or module path so that I can validate specific sections of the book were ingested correctly.

**Why this priority**: Enables targeted validation of specific book modules without noise from other sections.

**Independent Test**: Can be tested by running a filtered query and verifying only results from the specified module are returned.

**Acceptance Scenarios**:

1. **Given** book content from multiple modules is ingested, **When** a developer queries with a module filter (e.g., "module-1-ros2"), **Then** only chunks from URLs containing that path prefix are returned.

2. **Given** a filter that matches no stored content, **When** a developer runs a filtered query, **Then** the system returns an empty result set with an informative message.

3. **Given** a specific source URL filter, **When** a developer queries, **Then** only chunks from that exact URL are returned.

---

### User Story 3 - Batch Validation with Test Queries (Priority: P2)

As a developer, I want to run a predefined set of representative test queries so that I can systematically validate retrieval quality across different book topics.

**Why this priority**: Enables consistent, repeatable validation without manual query entry each time.

**Independent Test**: Can be tested by running a validation command that executes 5+ test queries and reports pass/fail for each.

**Acceptance Scenarios**:

1. **Given** a set of 5+ representative test queries covering different modules, **When** a developer runs the validation command, **Then** the system executes all queries and displays relevance summary for each.

2. **Given** test queries are executed, **When** results are returned, **Then** each result includes the query, top matches, scores, and source URLs in a readable format.

3. **Given** a test query returns no relevant results (score below threshold), **When** validation completes, **Then** that query is flagged as potentially problematic.

---

### User Story 4 - Configurable Result Count (Priority: P3)

As a developer, I want to configure how many results (top-k) are returned so that I can adjust the depth of my validation.

**Why this priority**: Flexibility feature - default of 5 works for most cases, but adjustability adds value.

**Independent Test**: Can be tested by running the same query with different k values and verifying result count matches.

**Acceptance Scenarios**:

1. **Given** a query with k=3, **When** the developer runs the command, **Then** exactly 3 results are returned (or fewer if collection has less).

2. **Given** a query with k=10, **When** the developer runs the command, **Then** up to 10 results are returned with descending similarity scores.

---

### Edge Cases

- What happens when the query is empty or only whitespace? System rejects with clear error message.
- What happens when Qdrant connection fails? System displays connection error with troubleshooting hints.
- What happens when Cohere API rate limit is hit? System handles gracefully with retry or informative message.
- What happens when a query returns results with very low similarity scores? System displays scores and optionally warns about low relevance.
- What happens when the collection doesn't exist? System informs user to run ingestion first.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST generate query embeddings using the same Cohere model (embed-english-v3.0) and dimensions (1024) as the ingestion pipeline.
- **FR-002**: System MUST use input_type="search_query" for query embeddings (vs "search_document" used in ingestion) per Cohere best practices.
- **FR-003**: System MUST retrieve top-k results from Qdrant with similarity scores.
- **FR-004**: System MUST return payload fields for each result: source_url, content (text snippet), chunk_id.
- **FR-005**: System MUST support filtering results by source URL path prefix (e.g., filter to only "module-1-ros2" pages).
- **FR-006**: System MUST provide a CLI interface runnable via a single command.
- **FR-007**: System MUST display results in a clear, readable format showing: rank, score, source URL, and content preview (first 200 characters).
- **FR-008**: System MUST validate Qdrant connection and collection existence before querying.
- **FR-009**: System MUST include a batch validation mode that runs 5+ predefined test queries.
- **FR-010**: System MUST load configuration (Qdrant URL, API key, Cohere API key, collection name) from environment variables (same .env as Spec 1).

### Key Entities

- **Query**: User's natural language question to search the book content.
- **QueryEmbedding**: Vector representation of the query generated by Cohere (1024 dimensions).
- **RetrievalResult**: A matched chunk containing: score (similarity), source_url, content, chunk_id.
- **ValidationReport**: Summary of batch test query results including pass/fail status per query.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Developers can execute a retrieval query and receive results in under 5 seconds for typical queries.
- **SC-002**: All 5+ predefined test queries return at least one relevant result (score > 0.5) from the appropriate book module.
- **SC-003**: Filtered queries return only results matching the specified path prefix with 100% accuracy.
- **SC-004**: CLI output clearly displays rank, similarity score, source URL, and content preview for each result.
- **SC-005**: System correctly handles error conditions (empty collection, connection failure, invalid query) with informative messages.
- **SC-006**: Query embeddings use the same model and dimensions as ingestion, ensuring vector compatibility.

## Assumptions

- The ingestion pipeline (Spec 1) has already been run and the Qdrant collection contains valid vectors.
- The same .env configuration file from Spec 1 is available with Qdrant and Cohere credentials.
- Cohere embed-english-v3.0 model produces 1024-dimensional vectors.
- A similarity score threshold of 0.5 is reasonable for determining "relevant" results (can be adjusted based on testing).
- Default top-k value of 5 results is sufficient for most validation scenarios.

## Out of Scope

- URL ingestion, crawling, extraction, or chunking (handled by Spec 1)
- OpenAI agent or tool calling integration
- Frontend/web interface
- Reranking or hybrid search capabilities
- Comprehensive evaluation framework beyond basic sanity tests
- FastAPI or any web server
