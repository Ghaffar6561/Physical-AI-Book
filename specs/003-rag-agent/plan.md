# Implementation Plan: RAG Agent with Book Content Retrieval

**Branch**: `003-rag-agent` | **Date**: 2024-12-28 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-rag-agent/spec.md`

## Summary

Build a single-file CLI agent (`agent.py`) that answers natural-language questions about the Physical AI book using OpenAI as the reasoning engine and Qdrant (via existing retrieval infrastructure from Spec 2) for semantic search. The agent enforces strict grounding: answers must derive from retrieved content with citations, and questions outside the book's scope are handled gracefully.

## Technical Context

**Language/Version**: Python 3.11+ (matching existing `backend/` codebase)
**Primary Dependencies**: `openai` (Agents SDK or function-calling), `cohere` (embeddings), `qdrant-client`, `python-dotenv`
**Storage**: Qdrant Cloud (existing `physical-ai-book` collection with 152+ vectors)
**Testing**: pytest (existing `tests/` directory structure)
**Target Platform**: Windows/Linux/macOS CLI (local development)
**Project Type**: Single file addition to existing backend project
**Performance Goals**: Response time <10 seconds per query (per SC-001)
**Constraints**: Single-user, ephemeral conversation history; no streaming
**Scale/Scope**: Single concurrent user per agent instance

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The constitution template is not populated for this project. Applying default principles:

| Principle | Status | Notes |
|-----------|--------|-------|
| Library-first | ✅ PASS | Reuses existing `retrieve.py` functions; new file is standalone |
| CLI Interface | ✅ PASS | Agent exposes CLI with stdin→stdout pattern |
| Test-first | ⚠️ DEFERRED | Tests will be defined in `/sp.tasks` phase |
| Simplicity | ✅ PASS | Single-file approach; minimal dependencies added |

No blocking violations. Proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/003-rag-agent/
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
├── agent.py             # NEW: RAG Agent CLI (single-file implementation)
├── retrieve.py          # EXISTING: Reuse retrieval functions
├── config.py            # EXISTING: Reuse Config class
├── models.py            # EXISTING: Add new dataclasses (Conversation, Message, AgentResponse)
├── exceptions.py        # EXISTING: Add agent-specific exceptions if needed
└── requirements.txt     # UPDATE: Add openai dependency

tests/
├── integration/
│   └── test_pipeline.py # EXISTING
└── unit/
    └── test_agent.py    # NEW: Agent unit tests (in /sp.tasks phase)
```

**Structure Decision**: Single-file addition to existing `backend/` directory. Leverages existing infrastructure from Specs 1-2 (config, models, retrieve functions). New `agent.py` at project root per user spec requirement.

## Complexity Tracking

No violations requiring justification. Design follows simplicity principle with single-file implementation.

---

## Phase 0: Research Findings

See [research.md](./research.md) for detailed research on:
- OpenAI Agents SDK vs function-calling approach
- Multi-turn conversation context management
- Citation formatting patterns
- Error handling strategies

## Phase 1: Design Artifacts

See:
- [data-model.md](./data-model.md) - Entity definitions for Conversation, Message, AgentResponse
- [quickstart.md](./quickstart.md) - Setup and usage instructions
- `contracts/` - CLI interface contract (stdin/stdout patterns)
