# Data Model: RAG Agent

**Feature**: 003-rag-agent | **Date**: 2024-12-28

## Entity Definitions

Based on Key Entities from spec.md, refined for implementation.

---

## 1. Message

A single turn in a conversation, either from user or assistant.

```python
@dataclass
class Message:
    """A single message in the conversation."""
    role: str           # "user" | "assistant" | "system" | "tool"
    content: str        # Message text content
    timestamp: datetime # When the message was created
    tool_calls: Optional[list[dict]] = None  # For assistant tool invocations
    tool_call_id: Optional[str] = None       # For tool responses
```

**Validation Rules**:
- `role` must be one of: "user", "assistant", "system", "tool"
- `content` cannot be empty for "user" role
- `tool_calls` only valid for "assistant" role
- `tool_call_id` only valid for "tool" role

**State Transitions**: N/A (immutable after creation)

---

## 2. Conversation

A session containing a sequence of messages with shared context.

```python
@dataclass
class Conversation:
    """A conversation session with the agent."""
    id: str                    # Unique conversation identifier (UUID)
    messages: list[Message]    # Ordered list of messages
    created_at: datetime       # Session start time
    model: str                 # OpenAI model used (e.g., "gpt-4o-mini")

    def add_message(self, message: Message) -> None:
        """Append a message to the conversation."""
        self.messages.append(message)

    def to_openai_messages(self) -> list[dict]:
        """Convert to OpenAI API format."""
        return [
            {"role": m.role, "content": m.content, **({"tool_calls": m.tool_calls} if m.tool_calls else {})}
            for m in self.messages
        ]
```

**Validation Rules**:
- Must start with a "system" message
- User and assistant messages alternate (with tool calls in between)
- Maximum 100 messages per conversation (soft limit)

**Relationships**:
- Contains many `Message` objects (1:N)

---

## 3. RetrievalContext

The set of chunks retrieved for a given query (extends existing `RetrievalResult`).

```python
@dataclass
class RetrievalContext:
    """Context retrieved from Qdrant for a query."""
    query: str                           # Original user query
    results: list[RetrievalResult]       # Retrieved chunks (from models.py)
    retrieved_at: datetime               # Timestamp of retrieval
    filter_applied: Optional[str] = None # Any filter prefix used

    def format_for_agent(self) -> str:
        """Format retrieved content for agent context."""
        if not self.results:
            return "No relevant content found in the book."

        lines = []
        for i, r in enumerate(self.results, 1):
            lines.append(f"[{i}] Source: {r.source_url}")
            lines.append(f"    Score: {r.score:.2f}")
            lines.append(f"    Content: {r.content}")
            lines.append("")
        return "\n".join(lines)

    def format_sources_footer(self) -> str:
        """Format sources for response footer."""
        if not self.results:
            return ""

        lines = ["", "---", "**Sources:**"]
        for i, r in enumerate(self.results, 1):
            lines.append(f"[{i}] {r.source_url} (score: {r.score:.2f})")
        return "\n".join(lines)
```

**Relationships**:
- Contains many `RetrievalResult` objects (1:N)
- Associated with one user `Message` that triggered retrieval

---

## 4. AgentResponse

The agent's complete answer including response text and metadata.

```python
@dataclass
class AgentResponse:
    """Complete response from the RAG agent."""
    content: str                         # The answer text
    sources: list[RetrievalResult]       # Chunks that were cited
    model: str                           # Model that generated response
    tokens_used: int                     # Total tokens (prompt + completion)
    retrieval_count: int                 # Number of chunks retrieved
    created_at: datetime                 # Response timestamp

    def format_with_sources(self) -> str:
        """Format response with source citations appended."""
        if not self.sources:
            return self.content

        footer_lines = ["", "---", "**Sources:**"]
        for i, s in enumerate(self.sources, 1):
            footer_lines.append(f"[{i}] {s.source_url} (score: {s.score:.2f})")

        return self.content + "\n".join(footer_lines)
```

**Validation Rules**:
- `content` cannot be empty
- `sources` may be empty if agent responds without retrieval
- `tokens_used` must be non-negative

---

## 5. AgentConfig

Runtime configuration for the agent (extends existing `Config`).

```python
@dataclass
class AgentConfig:
    """Configuration specific to the RAG agent."""
    openai_api_key: str         # Required
    openai_model: str           # Default: "gpt-4o-mini"
    top_k: int                  # Default: 5
    score_threshold: float      # Default: 0.0 (accept all)
    max_conversation_turns: int # Default: 50

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            top_k=int(os.getenv("AGENT_TOP_K", "5")),
            score_threshold=float(os.getenv("AGENT_SCORE_THRESHOLD", "0.0")),
            max_conversation_turns=int(os.getenv("AGENT_MAX_TURNS", "50")),
        )
```

---

## Entity Relationship Diagram

```
┌─────────────────┐
│  Conversation   │
│  - id           │
│  - model        │
│  - created_at   │
└────────┬────────┘
         │ 1:N
         ▼
┌─────────────────┐        ┌─────────────────┐
│    Message      │        │ RetrievalContext│
│  - role         │        │  - query        │
│  - content      │◄───────│  - results      │
│  - timestamp    │  1:1   │  - retrieved_at │
│  - tool_calls   │        └────────┬────────┘
└────────┬────────┘                 │ 1:N
         │                          ▼
         │               ┌─────────────────┐
         │               │ RetrievalResult │ (existing)
         │               │  - rank         │
         │               │  - score        │
         │               │  - source_url   │
         │               │  - content      │
         │               └─────────────────┘
         ▼
┌─────────────────┐
│  AgentResponse  │
│  - content      │
│  - sources      │
│  - tokens_used  │
└─────────────────┘
```

---

## Integration with Existing Models

The following entities from `backend/models.py` are reused without modification:

- `RetrievalResult` - Used as-is for search results
- `ValidationResult` - Used for testing/validation

New entities (`Message`, `Conversation`, `RetrievalContext`, `AgentResponse`, `AgentConfig`) will be added to `backend/models.py` or defined inline in `agent.py` for simplicity.

**Recommendation**: For the single-file implementation, define these dataclasses directly in `agent.py` to maintain self-containment. They can be extracted to `models.py` if the agent grows in complexity.
