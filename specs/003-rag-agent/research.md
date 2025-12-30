# Research: RAG Agent Implementation

**Feature**: 003-rag-agent | **Date**: 2024-12-28

## Research Tasks

Based on Technical Context unknowns and dependencies identified in plan.md:

1. OpenAI SDK approach for tool-calling agents
2. Multi-turn conversation context management
3. Citation and source attribution patterns
4. Error handling for retrieval failures

---

## 1. OpenAI SDK Approach for Tool-Calling Agents

### Decision: Use OpenAI Chat Completions with Function Calling

**Rationale**: OpenAI's function calling (tools) API is mature, well-documented, and provides the cleanest integration for RAG agents. The newer Agents SDK is designed for multi-agent orchestration which exceeds our scope.

**Alternatives Considered**:

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Chat Completions + Tools** | Stable, well-documented, single API call pattern | Requires manual message history management | ✅ Selected |
| OpenAI Assistants API | Built-in thread management, retrieval | Adds latency (polling), overkill for single-file CLI | ❌ Rejected |
| Agents SDK (beta) | Multi-agent ready, typed tool definitions | Beta status, complex setup, exceeds scope | ❌ Rejected |
| LangChain/LlamaIndex | Rich abstractions | Heavy dependencies, abstracts away control | ❌ Rejected |

**Implementation Pattern**:
```python
# Define retrieval tool schema
tools = [{
    "type": "function",
    "function": {
        "name": "search_book_content",
        "description": "Search the Physical AI book for relevant content",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 5}
            },
            "required": ["query"]
        }
    }
}]

# Agent decides when to call retrieval
response = client.chat.completions.create(
    model="gpt-4o-mini",  # Cost-effective for RAG
    messages=conversation_history,
    tools=tools,
    tool_choice="auto"  # Let model decide
)
```

---

## 2. Multi-Turn Conversation Context Management

### Decision: In-Memory Message List with Role Tagging

**Rationale**: Simple list of message dicts maintains OpenAI's expected format. No persistence needed (spec: ephemeral sessions).

**Alternatives Considered**:

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **List of dicts** | Simple, matches API format, no overhead | Memory-only, no persistence | ✅ Selected |
| SQLite/JSON file | Persistence across sessions | Out of scope per spec | ❌ Rejected |
| Sliding window (last N) | Bounded memory | Loses early context | ❌ Not needed yet |

**Implementation Pattern**:
```python
conversation = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is ROS2?"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "Retrieved: ..."},
    {"role": "assistant", "content": "Based on the book, ROS2 is..."},
]
```

**Context Window Management**: For GPT-4o-mini with 128K context, no immediate truncation needed. Add token counting if conversations exceed ~100K tokens.

---

## 3. Citation and Source Attribution Patterns

### Decision: Inline Citations with Sources Footer

**Rationale**: Matches academic/documentation patterns users expect. Sources section allows verification.

**Format Specification**:
```
Based on the book, ROS2 (Robot Operating System 2) is a flexible framework
for writing robot software [1]. It provides tools, libraries, and conventions
for building complex robotic systems [1][2].

---
**Sources:**
[1] https://physical-ai-book.com/module-1-ros2/intro (score: 0.87)
[2] https://physical-ai-book.com/module-1-ros2/architecture (score: 0.82)
```

**Implementation**:
- Retrieved chunks are numbered in order received
- Assistant references by number in response
- Sources appended with URL and similarity score

---

## 4. Error Handling for Retrieval Failures

### Decision: Graceful Degradation with User Feedback

**Error Categories**:

| Error Type | Handling | User Message |
|------------|----------|--------------|
| Qdrant unavailable | Catch `RetrievalError`, abort tool call | "I'm unable to search the book right now. The retrieval service is unavailable." |
| No results found | Return empty context to agent | Agent responds: "I couldn't find relevant information about that topic in the book." |
| Cohere API failure | Catch embedding error | "I'm unable to process your question right now. Please try again." |
| Empty/whitespace query | Validate before sending | "Please provide a question to ask about the book." |
| Low similarity scores | Pass to agent with scores | Agent decides if content is relevant enough |

**Implementation Pattern**:
```python
def search_book_content(query: str, top_k: int = 5) -> dict:
    try:
        results = retrieve.search_qdrant(...)
        if not results:
            return {"status": "no_results", "content": [], "message": "No relevant content found"}
        return {"status": "success", "content": format_results(results)}
    except RetrievalError as e:
        return {"status": "error", "content": [], "message": str(e)}
```

---

## 5. Model Selection

### Decision: GPT-4o-mini (default) with GPT-4o option

**Rationale**: GPT-4o-mini is 10x cheaper, sufficient for RAG grounding tasks. GPT-4o available via env var for complex reasoning.

**Configuration**:
```python
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
```

---

## 6. System Prompt Design

### Decision: Strict Grounding with Retrieval-Only Policy

**System Prompt**:
```
You are a helpful assistant that answers questions about the Physical AI book.

IMPORTANT RULES:
1. ONLY use information from the search_book_content tool to answer questions.
2. If the tool returns no results or irrelevant content, say "I couldn't find relevant information about that in the book."
3. NEVER make up information. If unsure, acknowledge the limitation.
4. Always cite your sources using [1], [2], etc. notation.
5. After your answer, list the sources with their URLs.

When the user asks a question:
1. First, search the book content using the provided tool
2. Review the retrieved chunks for relevance
3. Formulate your answer based ONLY on the retrieved content
4. Include citations and source URLs
```

---

## Dependencies Summary

**New Dependencies** (add to requirements.txt):
```
openai>=1.0.0
```

**Environment Variables** (add to .env.example):
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## Open Questions (None)

All technical clarifications resolved. Proceed to Phase 1 design.
