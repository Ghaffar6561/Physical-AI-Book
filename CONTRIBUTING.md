# Contributing to Physical AI & Humanoid Robotics Book

Thank you for your interest in contributing to this comprehensive technical textbook! This guide explains how to contribute content, code examples, and improvements.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Code Example Guidelines](#code-example-guidelines)
3. [Writing Content](#writing-content)
4. [Testing](#testing)
5. [Pull Request Process](#pull-request-process)
6. [Style Guide](#style-guide)

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git
- ROS 2 (Humble or Jazzy) for testing ROS 2 examples
- Gazebo 11+ for simulation examples (optional)

### Setup Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/asad/PhysicalAI-Book.git
   cd PhysicalAI-Book
   ```

2. **Install Node.js dependencies** (for Docusaurus):
   ```bash
   cd book
   npm install
   cd ..
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r book/examples/requirements.txt
   pip install pytest pytest-cov
   ```

4. **Start development server**:
   ```bash
   cd book
   npm start
   # Opens http://localhost:3000
   ```

---

## Code Example Guidelines

All code examples in the book must adhere to these standards:

### Execution & Performance (SC-009)

- **Execution time**: < 30 seconds (ideally < 10 seconds)
- **No external dependencies**: Examples should work with base libraries (unless documented)
- **Error handling**: Handle common edge cases gracefully
- **Output**: Produce clear, expected output that demonstrates learning

### Code Quality

- **Python 3.9+**: Compatible with ROS 2 distributions
- **Comments**: Explain key concepts; assume senior CS student audience
- **Variable names**: Clear, descriptive (avoid `x`, `y`, `z` for logic)
- **Imports**: Minimal, well-organized
- **No hardcoded secrets**: Use environment variables for API keys, credentials
- **Reproducibility**: Same input → same output every time

### Code Example Template

Every code example should follow this structure:

```python
"""
[Brief description of what this example teaches]

Key concepts:
- [Concept 1]
- [Concept 2]

Expected output:
  [Show what student will see when running]
"""

import [required modules]

def main():
    """Main function demonstrating the concept."""
    # Setup
    print("Running [Example Name]...")

    # Core logic
    result = [operation]

    # Output
    print(f"Result: {result}")

if __name__ == '__main__':
    main()
```

### Example: Simple ROS 2 Publisher

```python
"""
Minimal ROS 2 publisher example.

Key concepts:
- Creating a ROS 2 node
- Publishing to a topic

Expected output:
  Publishing: Hello, ROS 2!
  Publishing: Hello, ROS 2!
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)

    def publish_once(self):
        msg = String()
        msg.data = 'Hello, ROS 2!'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

def main():
    rclpy.init()
    publisher = MinimalPublisher()
    publisher.publish_once()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Writing Content

### Markdown Files

- **File naming**: Use lowercase with hyphens: `embodied-intelligence.md`
- **Structure**: Use clear headings (h2, h3, h4)
- **Length**: Aim for 1000-3000 words per major section
- **Links**: Use relative paths: `[ROS 2](../02-simulation/ros2-intro.md)`
- **Code blocks**: Use GitHub-flavored markdown with language tags

### Module Structure

Each module should include:

1. **intro.md** — Overview, learning objectives, module structure
2. **[topic-1].md** — Core content with examples
3. **[topic-2].md** — Additional content (if applicable)
4. **setup-[tool].md** — Installation instructions
5. **exercises.md** — Practice problems with solutions

### Content Guidelines

- **Audience**: Senior CS students, robotics engineers, AI practitioners
- **Tone**: Technical but accessible; explain **why** not just **how**
- **Examples**: Every concept needs a code example
- **Visuals**: Include diagrams (SVG) for complex systems
- **Cross-references**: Link to related modules
- **Consistency**: Use same terminology across modules

### Diagrams

Diagrams should be:
- **Format**: SVG (preferred) or high-quality PNG
- **Location**: `book/static/diagrams/[module-name]/[diagram-name].svg`
- **Referenced**: Embedded in markdown with alt text
- **Clear**: Readable at small sizes, good contrast

Example:
```markdown
![ROS 2 Architecture](../static/diagrams/01-foundations/ros2-architecture.svg)
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_module1_examples.py -v

# Run with coverage report
pytest tests/ --cov --cov-report=html

# Run only code example tests
pytest tests/unit/ -v

# Run only capstone tests
pytest tests/capstone/ -v
```

### Writing Tests for Code Examples

Each code example needs a test. Place tests in:
- `tests/unit/test_module[N]_examples.py`

Example test:

```python
import pytest
from pathlib import Path

def test_ros2_publisher_example():
    """Test that minimal ROS 2 publisher runs without errors."""
    # This test would:
    # 1. Import the example module
    # 2. Mock rclpy if needed
    # 3. Call main()
    # 4. Verify output

    # For this simple example, just verify it imports
    example_path = Path(__file__).parent.parent.parent / \
                   "book/static/code-examples/minimal_publisher.py"
    assert example_path.exists()
```

### Test Requirements

- **All code examples must pass pytest** (SC-009)
- **95%+ of examples must run error-free**
- **Mocking**: Use mocks for external dependencies (ROS 2, Gazebo, LLMs)
- **Fixtures**: Use conftest.py fixtures for common setup

---

## Pull Request Process

### Before Creating a PR

1. **Verify your changes**:
   ```bash
   cd book && npm run build
   npm start  # Test locally at localhost:3000
   cd ..
   pytest tests/ -v  # Run all tests
   ```

2. **Update documentation**:
   - Add/update docstrings in code
   - Update README if needed
   - Add any new dependencies to requirements.txt

3. **Check formatting**:
   ```bash
   black [your-python-files]
   flake8 [your-python-files]
   ```

### Creating a PR

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Commit with clear messages**:
   ```bash
   git commit -m "Add [feature]: Brief description"
   ```

3. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **PR title and description**:
   - **Title**: Brief, descriptive
   - **Description**: What changed, why, and how to test

### PR Review Criteria

- ✅ Code examples run without errors
- ✅ Content is clear and accessible
- ✅ Links are correct (relative paths)
- ✅ Diagrams are referenced properly
- ✅ Tests pass
- ✅ Follows style guide
- ✅ No broken builds

---

## Style Guide

### Writing Style

- **Voice**: Technical, instructive, not promotional
- **Audience**: Advanced CS students (assume Python proficiency)
- **Clarity**: Prefer clarity over brevity
- **Consistency**: Use same terms throughout (e.g., "humanoid robot", not "humanoid" or "robot")

### Code Style

```python
# Good
def publish_sensor_data(topic_name, data):
    """Publish sensor data to ROS 2 topic."""
    publisher = self.create_publisher(PointCloud2, topic_name, 10)
    publisher.publish(data)

# Avoid
def pub(tn, d):
    p = self.create_publisher(PointCloud2, tn, 10)
    p.publish(d)
```

### Markdown Style

```markdown
# Heading 1 (Module title - use once)

## Heading 2 (Major sections)

### Heading 3 (Subsections)

**Bold**: For emphasis on key concepts
*Italic*: For references to variables/functions

- Bullet lists for multiple items
- Use consistent indentation
- Number lists when order matters

1. First item
2. Second item
3. Third item
```

### Comment Style

```python
# Good: Explain why
# Use ROS 2 actions for long-running tasks with feedback
action_client = ActionClient(self, JointTrajectory, 'arm/move')

# Avoid: Obvious comments
# Create action client
action_client = ActionClient(self, JointTrajectory, 'arm/move')
```

---

## Questions or Issues?

- **GitHub Issues**: [Open an issue](https://github.com/asad/PhysicalAI-Book/issues)
- **Discussions**: [Start a discussion](https://github.com/asad/PhysicalAI-Book/discussions)
- **Email**: Contact the maintainers

---

## Checklist Before Submitting

- [ ] Code examples execute in < 30 seconds
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Book builds locally: `cd book && npm run build`
- [ ] No broken links or missing files
- [ ] Diagrams render correctly
- [ ] Comments explain key concepts
- [ ] Following Python/Markdown style guide
- [ ] Commit messages are clear and descriptive

---

Thank you for contributing! Your work helps make Physical AI education accessible to the next generation of roboticists. 🤖

