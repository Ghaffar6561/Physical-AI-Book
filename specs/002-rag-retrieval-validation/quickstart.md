# Quickstart: RAG Retrieval Validation

**Feature**: 002-rag-retrieval-validation
**Date**: 2024-12-27

## Prerequisites

1. **Spec 1 Complete**: The ingestion pipeline must have run successfully
2. **Qdrant Collection**: `physical-ai-book` collection contains 152+ vectors
3. **Environment Variables**: `.env` file with Qdrant and Cohere credentials

## Installation

No additional dependencies required - uses existing backend/ environment.

```bash
cd backend
# Verify .env is configured (from Spec 1)
cat .env
```

## Usage

### Single Query Mode

Search for content with a natural language query:

```bash
python retrieve.py --query "What is ROS2?"
python retrieve.py -q "How does Gazebo work?" --top-k 10
```

### Filtered Query Mode

Restrict results to a specific module:

```bash
python retrieve.py --query "What is locomotion?" --filter "module-3-isaac"
python retrieve.py -q "How do sensors work?" -f "module-3" -k 3
```

### Batch Validation Mode

Run predefined test queries to validate retrieval quality:

```bash
python retrieve.py --validate
python retrieve.py -v --top-k 3
```

## Expected Output

### Single Query

```
Query: "What is ROS2?"
Results (5 found):

[1] Score: 0.58 | Source: ...l-ai-book.vercel.app/docs/module-1-ros2/ros2-architecture
    Module 1: The Robotic Nervous System (ROS 2) ROS 2 Introduction On this page ROS 2 Introduction What is ROS 2? ROS 2 (Robot Operating System 2) is the standard middleware for building robotic system...

[2] Score: 0.52 | Source: ...ffar-physical-ai-book.vercel.app/docs/module-1-ros2/intro
    Module 1: The Robotic Nervous System (ROS 2) Module 1: The Robotic Nervous System On this page Module 1: The Robotic Nervous System (ROS 2) Overview The Robot Operating System 2 (ROS 2) serves as th...

[3] Score: 0.48 | Source: https://ghaffar-physical-ai-book.vercel.app/
    ROS 2: The Robotic Nervous System Master ROS 2 nodes, topics, services, and actions. Bridge Python AI agents to robot controllers using rclpy and URDF...
```

### Filtered Query

```
Query: "What is locomotion?"
Results (3 found):

[1] Score: 0.45 | Source: ...ai-book.vercel.app/docs/module-3-isaac/bipedal-locomotion
    Module 3: The AI-Robot Brain (NVIDIA Isaac) Bipedal Locomotion On this page Bipedal Locomotion and Balance Control Overview Bipedal locomotion is one of the most challenging problems in robotics...
```

### Validation

```
Validation Results (5/5 passed):

[PASS] "What is ROS2?" - Score: 0.58, Source: module-1-ros2
[PASS] "How does Gazebo simulation work?" - Score: 0.66, Source: module-2-digital-twin
[PASS] "What is Isaac Sim?" - Score: 0.52, Source: module-3-isaac
[PASS] "How do vision language models work?" - Score: 0.70, Source: module-4-vla
[PASS] "What is bipedal locomotion?" - Score: 0.65, Source: module-3-isaac
```

## CLI Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--query` | `-q` | Natural language query | (required unless --validate) |
| `--top-k` | `-k` | Number of results | 5 |
| `--filter` | `-f` | Filter by URL prefix | (none) |
| `--validate` | `-v` | Run batch validation | false |

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Cannot connect to Qdrant" | Check QDRANT_URL and QDRANT_API_KEY in .env |
| "Collection not found" | Run ingestion pipeline first (Spec 1) |
| "Cohere API failed" | Verify COHERE_API_KEY is valid |
| Low scores (<0.5) | May indicate ingestion issues or off-topic query |
