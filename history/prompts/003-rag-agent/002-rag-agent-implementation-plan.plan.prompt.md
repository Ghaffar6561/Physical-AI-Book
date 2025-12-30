---
id: 002
title: RAG Agent Implementation Plan
stage: plan
date: 2024-12-28
surface: agent
model: claude-opus-4-5-20251101
feature: 003-rag-agent
branch: 003-rag-agent
user: architect
command: /sp.plan
labels: ["rag", "agent", "openai", "planning", "architecture"]
links:
  spec: specs/003-rag-agent/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/003-rag-agent/plan.md
  - specs/003-rag-agent/research.md
  - specs/003-rag-agent/data-model.md
  - specs/003-rag-agent/quickstart.md
  - specs/003-rag-agent/contracts/cli-interface.md
  - specs/003-rag-agent/contracts/tool-schema.json
tests:
  - None (planning phase)
---

## Prompt

/sp.plan Spec 3 — Retrieval-enabled OpenAI Agent (single-file)

Objective
- Build a local OpenAI agent that answers book questions using Qdrant retrieval only.

Plan
0) Setup
   - Create a single file in project root: agent.py.
   - Load env vars: OPENAI_API_KEY, COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION.

1) Retrieval Function
   - Embed user query with Cohere (same model as ingestion).
   - Search Qdrant (top_k configurable, optional filters).
   - Return cleaned chunks + source metadata.

2) Agent Construction (OpenAI SDK)
   - Define a system policy: answer strictly from retrieved context; say "not found" if missing.
   - Register retrieval as a tool/function the agent can call.

3) Response Assembly
   - Agent uses retrieved chunks as grounding context.
   - Include citations (source_url + chunk_index) in the final answer.

4) CLI Runner
   - main() in agent.py for interactive Q&A loop and single-shot mode.
   - Log: tool calls, retrieved count, and sources used.

Definition of Done
- `python agent.py` runs an interactive agent.
- For test questions, responses reference retrieved chunks and include sources.
- If retrieval returns nothing relevant, agent clearly states it cannot answer from the book.

## Response snapshot

Executed Phase 0 (Research) and Phase 1 (Design) of the /sp.plan workflow:

**Phase 0 - Research Findings:**
- Selected OpenAI Chat Completions with function calling over Assistants API or Agents SDK
- In-memory message list for multi-turn context (no persistence per spec)
- Inline citations with sources footer pattern
- Graceful degradation error handling strategy
- GPT-4o-mini as default model (cost-effective for RAG)

**Phase 1 - Design Artifacts Created:**
- `plan.md` - Full technical context, constitution check, project structure
- `research.md` - Detailed research on all technical decisions
- `data-model.md` - Entity definitions (Message, Conversation, RetrievalContext, AgentResponse)
- `quickstart.md` - Setup and usage instructions
- `contracts/cli-interface.md` - CLI I/O contract specification
- `contracts/tool-schema.json` - OpenAI tool definition schema

## Outcome

- ✅ Impact: Created complete implementation plan for RAG agent with clear technical direction
- 🧪 Tests: Deferred to /sp.tasks phase (test-first principle)
- 📁 Files: 6 files created in specs/003-rag-agent/
- 🔁 Next prompts: Run /sp.tasks to generate implementation tasks
- 🧠 Reflection: Single-file approach aligns with user request; reuses Spec 2 retrieval infrastructure

## Evaluation notes (flywheel)

- Failure modes observed: None - plan completed successfully
- Graders run and results (PASS/FAIL): N/A - planning phase
- Prompt variant (if applicable): Standard /sp.plan workflow
- Next experiment (smallest change to try): Validate research decisions during implementation
