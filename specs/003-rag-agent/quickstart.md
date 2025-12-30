# Quickstart: RAG Agent

**Feature**: 003-rag-agent | **Date**: 2024-12-28

## Prerequisites

Before running the RAG agent, ensure:

1. **Specs 1-2 completed**: The Qdrant collection must contain book embeddings
2. **Python 3.11+** installed
3. **API keys** configured (OpenAI, Cohere, Qdrant)

### Verify Prerequisites

```bash
# Check Qdrant collection exists with embeddings
cd backend
python retrieve.py --validate
```

Expected output:
```
Validation Results (5/5 passed):
[PASS] "What is ROS2?" - Score: 0.87, Source: module-1-ros2
...
```

---

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

The agent requires one additional dependency not in Specs 1-2:
```bash
pip install openai>=1.0.0
```

### 2. Configure Environment

Add to `backend/.env`:

```bash
# Existing (from Specs 1-2)
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=physical-ai-book

# New for Spec 3
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini  # Optional, default is gpt-4o-mini
```

### 3. Verify Configuration

```bash
# Test that all API keys are valid
python -c "from config import Config; Config(); print('Config OK')"
```

---

## Usage

### Interactive Mode

Start the agent for a conversational session:

```bash
cd backend
python agent.py
```

Example session:
```
RAG Agent ready. Type 'quit' to exit.

You: What is ROS2?

Agent: Based on the book, ROS2 (Robot Operating System 2) is a flexible
framework for writing robot software. It provides tools, libraries, and
conventions that aim to simplify the task of creating complex and robust
robot behavior across a wide variety of robotic platforms [1].

---
**Sources:**
[1] https://physical-ai-book.com/module-1-ros2/intro (score: 0.87)

You: How does it handle real-time communication?

Agent: ROS2 uses DDS (Data Distribution Service) as its middleware layer
for real-time communication. This enables deterministic message passing
and QoS (Quality of Service) policies for time-critical applications [1][2].

---
**Sources:**
[1] https://physical-ai-book.com/module-1-ros2/dds (score: 0.84)
[2] https://physical-ai-book.com/module-1-ros2/realtime (score: 0.79)

You: quit
Goodbye!
```

### Single-Shot Mode

Ask a single question and exit:

```bash
python agent.py --query "What is bipedal locomotion?"
```

### Advanced Options

```bash
# Specify number of chunks to retrieve
python agent.py --top-k 3

# Use GPT-4 for complex questions
OPENAI_MODEL=gpt-4o python agent.py

# Enable verbose logging
python agent.py --verbose
```

---

## CLI Reference

```
usage: agent.py [-h] [-q QUERY] [-k TOP_K] [-v]

RAG Agent for Physical AI Book

options:
  -h, --help            show this help message and exit
  -q, --query QUERY     Single question to ask (exits after response)
  -k, --top-k TOP_K     Number of chunks to retrieve (default: 5)
  -v, --verbose         Enable verbose logging (shows tool calls)
```

---

## Troubleshooting

### "OPENAI_API_KEY environment variable must be set"

Ensure `OPENAI_API_KEY` is in your `.env` file or exported in your shell.

### "Collection 'physical-ai-book' not found"

Run the ingestion pipeline first (Spec 1):
```bash
python main.py
```

### "I couldn't find relevant information about that in the book"

This is expected behavior when asking about topics not covered in the book. The agent is designed to only answer from retrieved content.

### Slow responses (>10 seconds)

- Check your internet connection
- Consider using `gpt-4o-mini` (default) instead of `gpt-4o`
- Reduce `--top-k` to retrieve fewer chunks

---

## Testing

Run the agent tests:

```bash
cd backend
pytest tests/unit/test_agent.py -v
```

Integration test with real APIs:

```bash
pytest tests/integration/test_agent_e2e.py -v
```

---

## Next Steps

- Run `/sp.tasks` to generate implementation tasks
- Implement `agent.py` following the task list
- Run tests to validate acceptance criteria
