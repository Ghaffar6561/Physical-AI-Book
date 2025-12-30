# CLI Interface Contract: RAG Agent

**Feature**: 003-rag-agent | **Date**: 2024-12-28

## Overview

The RAG Agent exposes a command-line interface following the project's CLI-first principle. This document defines the input/output contract.

---

## Invocation

### Interactive Mode (default)

```bash
python agent.py [OPTIONS]
```

Starts a REPL loop that reads user questions from stdin and writes responses to stdout.

### Single-Shot Mode

```bash
python agent.py --query "Your question here"
```

Answers a single question and exits.

---

## Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--query` | `-q` | string | None | Single question (exits after response) |
| `--top-k` | `-k` | int | 5 | Number of chunks to retrieve |
| `--verbose` | `-v` | flag | False | Show tool calls and retrieval details |
| `--help` | `-h` | flag | - | Show help message |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for chat completions |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model to use for agent |
| `COHERE_API_KEY` | Yes | - | Cohere API key for embeddings |
| `QDRANT_URL` | Yes | - | Qdrant cluster URL |
| `QDRANT_API_KEY` | Yes | - | Qdrant API key |
| `QDRANT_COLLECTION_NAME` | No | `book_embeddings` | Collection to search |
| `AGENT_TOP_K` | No | `5` | Default top-k for retrieval |
| `AGENT_SCORE_THRESHOLD` | No | `0.0` | Minimum similarity score |

---

## Input/Output Format

### Interactive Mode I/O

**Prompt format**:
```
You: <user_input>
```

**Response format**:
```
Agent: <response_text>

---
**Sources:**
[1] <source_url> (score: <score>)
[2] <source_url> (score: <score>)

You:
```

**Exit commands**: `quit`, `exit`, `q`, Ctrl+C

### Single-Shot Mode I/O

**Input**: Command-line argument `--query`

**Output**: Response text to stdout, sources included

**Exit codes**:
- `0`: Success
- `1`: Error (missing config, API failure, etc.)

---

## Output Examples

### Successful Response

```
Agent: Based on the book, ROS2 (Robot Operating System 2) is a flexible
framework for writing robot software. It provides tools and libraries for
building complex robotic systems [1].

---
**Sources:**
[1] https://physical-ai-book.com/module-1-ros2/intro (score: 0.87)
```

### No Results Found

```
Agent: I couldn't find relevant information about that topic in the book.
Please try rephrasing your question or asking about a different topic.
```

### Empty Query

```
Error: Please provide a question to ask about the book.
```

### Service Unavailable

```
Error: I'm unable to search the book right now. The retrieval service is unavailable.
Please check your connection and try again.
```

---

## Verbose Mode Output

When `--verbose` is enabled, additional information is printed to stderr:

```
[DEBUG] User query: "What is ROS2?"
[DEBUG] Calling tool: search_book_content(query="What is ROS2?", top_k=5)
[DEBUG] Retrieved 5 chunks in 0.34s
[DEBUG] Top score: 0.87, Source: module-1-ros2/intro
[DEBUG] Generating response with gpt-4o-mini...
[DEBUG] Response generated in 1.2s (245 tokens)

Agent: Based on the book, ROS2...
```

---

## Error Taxonomy

| Error Code | Category | Message Pattern | Recovery |
|------------|----------|-----------------|----------|
| E001 | Config | `OPENAI_API_KEY environment variable must be set` | Set env var |
| E002 | Config | `Collection 'X' not found` | Run ingestion |
| E003 | Network | `Cannot connect to Qdrant` | Check URL/key |
| E004 | Network | `OpenAI API error: ...` | Check key/quota |
| E005 | Input | `Query cannot be empty` | Provide query |
| E006 | Retrieval | `Cohere API failed: ...` | Check key/quota |

---

## Idempotency

Single-shot mode is idempotent: the same query with the same environment produces semantically equivalent responses (exact wording may vary due to LLM non-determinism).

Interactive mode maintains conversation state, so subsequent queries in the same session may produce different results based on context.

---

## Timeouts

| Operation | Timeout | Configurable |
|-----------|---------|--------------|
| Cohere embedding | 30s | No |
| Qdrant search | 30s | No |
| OpenAI completion | 60s | No |
| Total response | 90s | No |

If any timeout is exceeded, an error is returned and the operation can be retried.
