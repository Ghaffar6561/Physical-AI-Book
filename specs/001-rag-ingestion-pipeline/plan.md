# Implementation Plan: RAG Ingestion Pipeline

**Branch**: `001-rag-ingestion-pipeline` | **Date**: 2025-12-25 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-rag-ingestion-pipeline/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build an idempotent ingestion pipeline that crawls the deployed Docusaurus book, generates Cohere embeddings, and stores them in Qdrant. The pipeline will discover URLs from the book, extract clean text, chunk it deterministically, generate embeddings using the Cohere API, and store the results in Qdrant with stable IDs to prevent duplicates on re-runs.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: requests, beautifulsoup4, cohere, qdrant-client, python-dotenv
**Storage**: Qdrant vector database (external cloud service)
**Testing**: pytest
**Target Platform**: Linux/Mac/Windows server
**Project Type**: Single CLI application
**Performance Goals**: Process 1000 pages within 30 minutes, handle documents up to 10MB
**Constraints**: <200MB memory usage during processing, idempotent execution (no duplicates on re-run)
**Scale/Scope**: Handle up to 10,000 book pages, 50M embedding vectors

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Based on the project constitution:
- Library-First: N/A for this CLI tool approach
- CLI Interface: PASS - Will expose functionality via CLI
- Test-First: PASS - Will follow TDD approach with pytest
- Integration Testing: PASS - Will test integration with Cohere API and Qdrant
- Observability: PASS - Will implement structured logging

## Post-Design Constitution Check

*Re-evaluation after Phase 1 design is complete*

- CLI Interface: PASS - Main module will expose CLI functionality
- Test-First: PASS - Unit and integration tests planned for each component
- Integration Testing: PASS - Contracts defined for all external services (Cohere, Qdrant)
- Observability: PASS - Logging planned in storage and other components

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-ingestion-pipeline/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── main.py              # Main ingestion pipeline
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
├── config.py            # Configuration management
├── crawler.py           # URL discovery and content fetching
├── extractor.py         # Text extraction logic
├── chunker.py           # Text chunking logic
├── embedder.py          # Embedding generation using Cohere
└── storage.py           # Qdrant storage operations

tests/
├── unit/
│   ├── test_crawler.py
│   ├── test_extractor.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   └── test_storage.py
├── integration/
│   ├── test_pipeline.py
│   └── test_end_to_end.py
└── fixtures/
    └── sample_docs/

backup/                  # Backup directory for safety
```

**Structure Decision**: Selected single CLI application approach with modular components. The main.py orchestrates the pipeline, while individual modules handle specific concerns (crawling, extraction, chunking, embedding, storage). This follows the single project structure with clear separation of concerns.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
