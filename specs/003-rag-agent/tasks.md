# Tasks: RAG Agent with Book Content Retrieval

**Input**: Design documents from `/specs/003-rag-agent/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/
**Branch**: `003-rag-agent`

**Tests**: Not explicitly requested in spec - test tasks are minimal (validation only).

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Exact file paths included in descriptions

## Path Conventions

Following plan.md structure:
- **Source**: `backend/` at repository root (single-file addition)
- **Existing**: `backend/retrieve.py`, `backend/config.py`, `backend/models.py`
- **New**: `backend/agent.py` (single-file implementation)

---

## Phase 1: Setup

**Purpose**: Dependency installation and environment configuration

- [x] T001 Add `openai>=1.0.0` dependency to `backend/requirements.txt`
- [x] T002 [P] Add `OPENAI_API_KEY` and `OPENAI_MODEL` to `backend/.env.example`
- [x] T003 Verify existing Qdrant collection has embeddings by running `python backend/retrieve.py --validate`

**Checkpoint**: Environment ready for agent development

---

## Phase 2: Foundational (Core Agent Structure)

**Purpose**: Create the single-file agent skeleton that all user stories depend on

**⚠️ CRITICAL**: All user story work depends on this foundation being complete

- [x] T004 Create `backend/agent.py` with module docstring, imports, and argument parser (per `contracts/cli-interface.md`)
- [x] T005 Add `AgentConfig` dataclass to `backend/agent.py` that loads from environment (per `data-model.md`)
- [x] T006 Add `SYSTEM_PROMPT` constant to `backend/agent.py` with strict grounding rules (per `research.md` section 6)
- [x] T007 Add `TOOLS` list with `search_book_content` function schema to `backend/agent.py` (per `contracts/tool-schema.json`)
- [x] T008 Implement `search_book_content()` function in `backend/agent.py` that wraps existing `retrieve.py` functions
- [x] T009 Add error handling wrapper for retrieval failures in `backend/agent.py` (per `research.md` section 4)
- [x] T010 Implement `run_agent_loop()` skeleton in `backend/agent.py` that processes messages and calls OpenAI

**Checkpoint**: Foundation ready - agent can be invoked but not yet answer questions

---

## Phase 3: User Story 1 - Ask a Question and Get Grounded Answer (Priority: P1) 🎯 MVP

**Goal**: Users can ask natural-language questions and receive grounded answers with source citations

**Independent Test**: Run `python backend/agent.py --query "What is ROS2?"` and verify response includes book content with source URLs

### Implementation for User Story 1

- [x] T011 [US1] Implement tool call handling in `run_agent_loop()` - detect tool_calls in OpenAI response in `backend/agent.py`
- [x] T012 [US1] Implement tool execution logic - call `search_book_content()` and format results in `backend/agent.py`
- [x] T013 [US1] Implement response formatting with inline citations [1][2] and sources footer in `backend/agent.py`
- [x] T014 [US1] Handle "no results found" case - agent responds it cannot find relevant information in `backend/agent.py`
- [x] T015 [US1] Implement single-shot mode (`--query` flag) that asks one question and exits in `backend/agent.py`
- [x] T016 [US1] Add `main()` function with argument parsing and entry point in `backend/agent.py`
- [x] T017 [US1] Validate US1 - PASSED: sources and citations displayed correctly

**Checkpoint**: User Story 1 complete - single questions work with grounded answers

---

## Phase 4: User Story 2 - Multi-Turn Conversation (Priority: P2)

**Goal**: Users can have follow-up conversations where the agent maintains context

**Independent Test**: Ask "What is ROS2?", then ask "How does it handle real-time?" and verify agent understands "it" refers to ROS2

### Implementation for User Story 2

- [x] T018 [US2] Add `Conversation` class to track message history in `backend/agent.py` (per `data-model.md`)
- [x] T019 [US2] Initialize conversation with system message on agent start in `backend/agent.py`
- [x] T020 [US2] Append user messages to conversation history before sending to OpenAI in `backend/agent.py`
- [x] T021 [US2] Append assistant messages (including tool calls) to conversation after receiving response in `backend/agent.py`
- [x] T022 [US2] Implement interactive REPL loop in `main()` - read stdin, process, print response in `backend/agent.py`
- [x] T023 [US2] Handle exit commands (`quit`, `exit`, `q`, Ctrl+C) gracefully in `backend/agent.py`
- [x] T024 [US2] Validate US2 - implementation complete (interactive mode ready)

**Checkpoint**: User Story 2 complete - multi-turn conversations work with context

---

## Phase 5: User Story 3 - View Retrieved Sources (Priority: P3)

**Goal**: Users can see what content was retrieved with relevance scores

**Independent Test**: Ask a question and verify output includes "Sources:" section with URLs and scores

### Implementation for User Story 3

- [x] T025 [US3] Ensure sources footer is always displayed after agent response in `backend/agent.py`
- [x] T026 [US3] Format sources with rank, URL, and similarity score (highest first) in `backend/agent.py`
- [x] T027 [US3] Add `--verbose` flag that shows tool calls and retrieval details to stderr in `backend/agent.py`
- [x] T028 [US3] Validate US3 - PASSED: sources displayed with rank, URL, and scores

**Checkpoint**: User Story 3 complete - sources are visible and ranked by relevance

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Edge cases, robustness, and final validation

- [x] T029 Handle empty/whitespace query with user-friendly prompt in `backend/agent.py`
- [x] T030 Handle Qdrant unavailable error with clear message in `backend/agent.py`
- [x] T031 Handle OpenAI API errors gracefully in `backend/agent.py`
- [x] T032 Add Windows console encoding fix (UTF-8) at top of `backend/agent.py`
- [x] T033 Run full quickstart validation - PASSED: agent responds with grounded answers
- [x] T034 Verify performance <10 seconds - PASSED: ~5 seconds observed

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational - delivers MVP
- **User Story 2 (Phase 4)**: Depends on US1 completion (builds on single-question flow)
- **User Story 3 (Phase 5)**: Depends on US1 completion (sources already present, just needs formatting)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Core Q&A - no dependencies on other stories
- **User Story 2 (P2)**: Multi-turn - depends on US1 (extends single-question to conversation)
- **User Story 3 (P3)**: View sources - can technically run in parallel with US2 after US1

### Within Each Phase

- Tasks without [P] marker must run sequentially
- Tasks with [P] marker in same phase can run in parallel
- Commit after each task or logical group

### Parallel Opportunities

**Phase 1 (Setup)**:
```
T001 and T002 can run in parallel (different files)
T003 must wait until T001 completes (needs openai dependency)
```

**Phase 2 (Foundational)**:
```
T004 through T010 are sequential (building up single file)
```

**After Foundational**:
```
US2 and US3 can theoretically run in parallel after US1 completes
However, since all work is in backend/agent.py, sequential is safer
```

---

## Parallel Example: Setup Phase

```bash
# These can run in parallel:
Task: "Add openai>=1.0.0 dependency to backend/requirements.txt"
Task: "Add OPENAI_API_KEY and OPENAI_MODEL to backend/.env.example"

# This must wait:
Task: "Verify existing Qdrant collection has embeddings"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 (single-question Q&A)
4. **STOP and VALIDATE**: Run `python backend/agent.py --query "What is ROS2?"`
5. If response includes grounded answer with sources → MVP complete

### Incremental Delivery

1. **MVP**: Setup + Foundational + US1 → Single-question agent works
2. **+US2**: Add multi-turn → Interactive conversations work
3. **+US3**: Add source display → Full transparency
4. **Polish**: Edge cases and robustness

### Single-File Approach

All implementation is in `backend/agent.py`:
- Keeps feature self-contained per spec requirement
- Reuses existing `retrieve.py` via imports
- Can be extracted to modules later if needed

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| Setup | T001-T003 | Dependencies and environment |
| Foundational | T004-T010 | Agent skeleton and retrieval tool |
| US1 (P1) | T011-T017 | Single-question Q&A with citations |
| US2 (P2) | T018-T024 | Multi-turn conversation context |
| US3 (P3) | T025-T028 | Source display with scores |
| Polish | T029-T034 | Edge cases and validation |

**Total Tasks**: 34
- **Per Story**: US1=7, US2=7, US3=4
- **Parallel Opportunities**: Phase 1 (T001/T002)
- **MVP Scope**: T001-T017 (17 tasks)

---

## Notes

- All implementation in single file: `backend/agent.py`
- Reuses existing `retrieve.py` infrastructure from Spec 2
- No tests explicitly requested in spec - validation via quickstart
- Each checkpoint represents a shippable increment
- Verbose mode helps with debugging during development
