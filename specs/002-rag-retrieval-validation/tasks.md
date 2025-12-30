# Tasks: RAG Retrieval Pipeline Validation

**Input**: Design documents from `/specs/002-rag-retrieval-validation/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Built-in validation mode serves as functional tests. No separate test phase required.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- File paths relative to repository root

## Path Conventions

Single module addition to existing `backend/` directory:
- `backend/retrieve.py` - New file
- `backend/models.py` - Existing, add models
- `backend/exceptions.py` - Existing, add exception
- `backend/config.py` - Existing, reuse as-is

---

## Phase 1: Setup ✅

**Purpose**: Prepare infrastructure for retrieval module

- [x] T001 Add RetrievalResult dataclass to backend/models.py
- [x] T002 [P] Add ValidationResult dataclass to backend/models.py
- [x] T003 [P] Add RetrievalError exception to backend/exceptions.py

**Checkpoint**: Data models and exception ready for retrieve.py ✅

---

## Phase 2: Foundational ✅

**Purpose**: Core infrastructure that all user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Create backend/retrieve.py with imports and docstring
- [x] T005 Implement generate_query_embedding(query: str) function in backend/retrieve.py
- [x] T006 Implement validate_connection() function to check Qdrant collection exists in backend/retrieve.py
- [x] T007 Implement CLI argument parser with argparse in backend/retrieve.py (--query, --top-k, --filter, --validate)

**Checkpoint**: Foundation ready - embedding generation, connection validation, and CLI parsing work ✅

---

## Phase 3: User Story 1 - Basic Query Retrieval (Priority: P1) 🎯 MVP ✅

**Goal**: Run a natural language query and return relevant chunks with scores, source URLs, and text snippets

**Independent Test**: `python backend/retrieve.py --query "What is ROS2?"` returns ranked results

### Implementation for User Story 1

- [x] T008 [US1] Implement search_qdrant(embedding, top_k) function in backend/retrieve.py
- [x] T009 [US1] Implement format_results(results) function for CLI output in backend/retrieve.py
- [x] T010 [US1] Implement main() function with single query mode in backend/retrieve.py
- [x] T011 [US1] Add error handling for empty query, connection failure, empty results in backend/retrieve.py
- [x] T012 [US1] Test basic query: `python backend/retrieve.py --query "What is ROS2?"`

**Checkpoint**: Basic query retrieval fully functional - MVP complete ✅

---

## Phase 4: User Story 2 - Filtered Retrieval (Priority: P2) ✅

**Goal**: Filter results by source URL path prefix (e.g., "module-1-ros2")

**Independent Test**: `python backend/retrieve.py --query "What is locomotion?" --filter "module-3-isaac"` returns only module-3 results

### Implementation for User Story 2

- [x] T013 [US2] Add filter_prefix parameter to search_qdrant() in backend/retrieve.py
- [x] T014 [US2] Implement Qdrant MatchText filter on source_url field in backend/retrieve.py
- [x] T015 [US2] Update main() to pass --filter argument to search in backend/retrieve.py
- [x] T016 [US2] Add handling for filter with no matches in backend/retrieve.py
- [x] T017 [US2] Test filtered query: `python backend/retrieve.py --query "What is locomotion?" --filter "module-3"`

**Checkpoint**: Filtered retrieval works independently ✅

---

## Phase 5: User Story 3 - Batch Validation (Priority: P2) ✅

**Goal**: Run predefined test queries and report pass/fail for each

**Independent Test**: `python backend/retrieve.py --validate` runs 5+ queries with summary

### Implementation for User Story 3

- [x] T018 [US3] Define TEST_QUERIES list with 5 queries covering all modules in backend/retrieve.py
- [x] T019 [US3] Implement run_validation() function in backend/retrieve.py
- [x] T020 [US3] Implement format_validation_results() for summary output in backend/retrieve.py
- [x] T021 [US3] Update main() to handle --validate mode in backend/retrieve.py
- [x] T022 [US3] Test validation mode: `python backend/retrieve.py --validate`

**Checkpoint**: Batch validation reports pass/fail for all test queries ✅

---

## Phase 6: User Story 4 - Configurable Result Count (Priority: P3) ✅

**Goal**: Allow --top-k argument to control number of results

**Independent Test**: `python backend/retrieve.py --query "ROS2" --top-k 3` returns exactly 3 results

### Implementation for User Story 4

- [x] T023 [US4] Ensure --top-k argument properly passed through CLI in backend/retrieve.py
- [x] T024 [US4] Update search_qdrant to respect top_k parameter in backend/retrieve.py
- [x] T025 [US4] Test configurable k: `python backend/retrieve.py --query "ROS2" --top-k 3`

**Checkpoint**: Result count is configurable via CLI ✅

---

## Phase 7: Polish & Cross-Cutting Concerns ✅

**Purpose**: Final validation and documentation

- [x] T026 Run full validation: `python backend/retrieve.py --validate` and verify all 5 queries pass
- [x] T027 Test edge cases: empty query, invalid filter, connection error
- [x] T028 Verify performance: queries complete in under 5 seconds
- [x] T029 Update quickstart.md with actual CLI output examples

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup (T001-T003)
- **User Story 1 (Phase 3)**: Depends on Foundational (T004-T007)
- **User Story 2 (Phase 4)**: Depends on User Story 1 (T008-T012)
- **User Story 3 (Phase 5)**: Depends on User Story 1 (T008-T012)
- **User Story 4 (Phase 6)**: Depends on User Story 1 (T008-T012)
- **Polish (Phase 7)**: Depends on all user stories

### User Story Dependencies

- **User Story 1 (P1)**: Core MVP - no dependencies on other stories
- **User Story 2 (P2)**: Extends US1's search_qdrant with filter parameter
- **User Story 3 (P2)**: Uses US1's query/format functions for batch execution
- **User Story 4 (P3)**: Simple parameter pass-through, builds on US1

### Within Each User Story

- Foundation must be complete before any story
- US2, US3, US4 can proceed in parallel after US1 complete

### Parallel Opportunities

```text
Setup Phase:
  T001 → T002 [P] and T003 [P] can run in parallel

Foundational Phase:
  T004 → T005, T006 [P], T007 [P] can run in parallel after T004

After User Story 1 Complete:
  US2 (T013-T017) ─┐
  US3 (T018-T022) ─┼─ Can run in parallel
  US4 (T023-T025) ─┘
```

---

## Parallel Example: After US1 Complete

```bash
# These user stories can be implemented in parallel:
# Team member A: User Story 2 - Filtered Retrieval
# Team member B: User Story 3 - Batch Validation
# Team member C: User Story 4 - Configurable Result Count
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T007)
3. Complete Phase 3: User Story 1 (T008-T012)
4. **STOP and VALIDATE**: Test with `python backend/retrieve.py --query "What is ROS2?"`
5. MVP is deployable/usable at this point

### Incremental Delivery

1. Setup + Foundational → Core infrastructure ready
2. Add User Story 1 → Basic query works → **MVP Ready**
3. Add User Story 2 → Filtered queries work
4. Add User Story 3 → Batch validation works
5. Add User Story 4 → Configurable results
6. Polish phase → Production ready

---

## Task Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| Setup | T001-T003 | Data models and exceptions |
| Foundational | T004-T007 | Core infrastructure |
| User Story 1 (P1) | T008-T012 | Basic query retrieval (MVP) |
| User Story 2 (P2) | T013-T017 | Filtered retrieval |
| User Story 3 (P2) | T018-T022 | Batch validation |
| User Story 4 (P3) | T023-T025 | Configurable result count |
| Polish | T026-T029 | Final validation |

**Total Tasks**: 29
**MVP Scope**: T001-T012 (12 tasks)
**Parallel Opportunities**: Setup phase, Foundational phase, US2/US3/US4 after US1

---

## Notes

- All tasks operate on `backend/retrieve.py` or existing `backend/` files
- No new dependencies required - reuses cohere and qdrant-client from Spec 1
- Built-in validation mode (US3) serves as functional test suite
- Each checkpoint allows validation before proceeding
