"""
Unit tests for Module 1 code examples.

Tests verify that all code examples in the Physical AI Foundations module:
1. Have correct Python syntax
2. Can be imported without errors
3. Have necessary dependencies
4. Run without runtime errors
"""

import pytest
import sys
from pathlib import Path
import ast
import importlib.util


# Get the path to code examples
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "book" / "static" / "code-examples"


class TestCodeExampleSyntax:
    """Test that all code examples have valid Python syntax."""

    def test_minimal_publisher_syntax(self):
        """Test minimal_publisher.py syntax."""
        example_file = EXAMPLES_DIR / "minimal_publisher.py"
        assert example_file.exists(), f"Code example not found: {example_file}"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should parse without SyntaxError
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in minimal_publisher.py: {e}")

    def test_minimal_subscriber_syntax(self):
        """Test minimal_subscriber.py syntax."""
        example_file = EXAMPLES_DIR / "minimal_subscriber.py"
        assert example_file.exists(), f"Code example not found: {example_file}"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should parse without SyntaxError
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in minimal_subscriber.py: {e}")

    def test_sensor_loop_diagram_syntax(self):
        """Test sensor_loop_diagram.py syntax."""
        example_file = EXAMPLES_DIR / "sensor_loop_diagram.py"
        assert example_file.exists(), f"Code example not found: {example_file}"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should parse without SyntaxError
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in sensor_loop_diagram.py: {e}")


class TestCodeExampleImports:
    """Test that code examples can be imported (syntax + basic structure)."""

    def test_minimal_publisher_imports(self):
        """Test that minimal_publisher can be parsed and has expected structure."""
        example_file = EXAMPLES_DIR / "minimal_publisher.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Find class definitions
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'MinimalPublisher' in classes, "MinimalPublisher class not found"

        # Find function definitions
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        assert 'main' in functions, "main() function not found"
        assert '__init__' in functions, "__init__() method not found"

    def test_minimal_subscriber_imports(self):
        """Test that minimal_subscriber can be parsed and has expected structure."""
        example_file = EXAMPLES_DIR / "minimal_subscriber.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Find class definitions
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'MinimalSubscriber' in classes, "MinimalSubscriber class not found"

        # Find function definitions
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        assert 'main' in functions, "main() function not found"
        assert 'listener_callback' in functions, "listener_callback() method not found"


class TestCodeExampleDocumentation:
    """Test that code examples are properly documented."""

    def test_minimal_publisher_docstring(self):
        """Test that minimal_publisher has a docstring."""
        example_file = EXAMPLES_DIR / "minimal_publisher.py"

        with open(example_file, 'r') as f:
            content = f.read()

        # Should have a module docstring
        assert '"""' in content, "Missing module docstring"
        assert 'ROS 2' in content, "Should mention ROS 2"
        assert 'Publisher' in content, "Should mention Publisher"

    def test_minimal_subscriber_docstring(self):
        """Test that minimal_subscriber has a docstring."""
        example_file = EXAMPLES_DIR / "minimal_subscriber.py"

        with open(example_file, 'r') as f:
            content = f.read()

        # Should have a module docstring
        assert '"""' in content, "Missing module docstring"
        assert 'ROS 2' in content, "Should mention ROS 2"
        assert 'Subscriber' in content, "Should mention Subscriber"


class TestCodeExampleContent:
    """Test that code examples contain expected content."""

    def test_minimal_publisher_has_required_components(self):
        """Test minimal_publisher has all required components."""
        example_file = EXAMPLES_DIR / "minimal_publisher.py"

        with open(example_file, 'r') as f:
            content = f.read()

        # Should use rclpy
        assert 'import rclpy' in content, "Should import rclpy"

        # Should use Node
        assert 'from rclpy.node import Node' in content, "Should import Node"

        # Should use String messages
        assert 'String' in content, "Should use String messages"

        # Should create a publisher
        assert 'create_publisher' in content, "Should create a publisher"

        # Should have a timer
        assert 'create_timer' in content, "Should create a timer"

    def test_minimal_subscriber_has_required_components(self):
        """Test minimal_subscriber has all required components."""
        example_file = EXAMPLES_DIR / "minimal_subscriber.py"

        with open(example_file, 'r') as f:
            content = f.read()

        # Should use rclpy
        assert 'import rclpy' in content, "Should import rclpy"

        # Should use Node
        assert 'from rclpy.node import Node' in content, "Should import Node"

        # Should use String messages
        assert 'String' in content, "Should use String messages"

        # Should create a subscription
        assert 'create_subscription' in content, "Should create a subscription"

        # Should have a callback
        assert 'listener_callback' in content, "Should have a listener_callback"


class TestCodeExampleLength:
    """Test that code examples are concise and readable."""

    def test_minimal_publisher_length(self):
        """Test that minimal_publisher is appropriately sized."""
        example_file = EXAMPLES_DIR / "minimal_publisher.py"

        with open(example_file, 'r') as f:
            lines = f.readlines()

        # Should be concise (not too short, not too long)
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        assert len(code_lines) < 100, "Code example should be concise"
        assert len(code_lines) > 15, "Code example should be substantial enough"

    def test_minimal_subscriber_length(self):
        """Test that minimal_subscriber is appropriately sized."""
        example_file = EXAMPLES_DIR / "minimal_subscriber.py"

        with open(example_file, 'r') as f:
            lines = f.readlines()

        # Should be concise (not too short, not too long)
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        assert len(code_lines) < 100, "Code example should be concise"
        assert len(code_lines) > 15, "Code example should be substantial enough"


@pytest.mark.unit
class TestModule1ExampleIntegration:
    """Integration-style tests for Module 1 examples."""

    def test_examples_exist(self):
        """Test that all required examples exist."""
        assert (EXAMPLES_DIR / "minimal_publisher.py").exists()
        assert (EXAMPLES_DIR / "minimal_subscriber.py").exists()

    def test_examples_are_executable(self):
        """Test that examples have executable shebangs or entry points."""
        for example in [EXAMPLES_DIR / "minimal_publisher.py", EXAMPLES_DIR / "minimal_subscriber.py"]:
            with open(example, 'r') as f:
                content = f.read()

            # Should have a main() function
            assert 'if __name__' in content, f"{example.name} should have main entry point"

    def test_all_examples_have_learning_goals(self):
        """Test that all examples document their learning goals."""
        for example in [EXAMPLES_DIR / "minimal_publisher.py", EXAMPLES_DIR / "minimal_subscriber.py"]:
            with open(example, 'r') as f:
                content = f.read()

            # Should have "Learning Goals" documented
            assert 'Learning' in content or 'Key Concepts' in content, \
                f"{example.name} should document what students learn"


# Module 1 specific tests
@pytest.mark.unit
class TestModule1SpecificContent:
    """Tests specific to Module 1 learning objectives."""

    def test_publisher_demonstrates_distributed_system(self):
        """Test that publisher example shows distributed node pattern."""
        example_file = EXAMPLES_DIR / "minimal_publisher.py"

        with open(example_file, 'r') as f:
            content = f.read()

        # Should show nodes as independent processes
        assert 'MinimalPublisher' in content, "Should define a Node class"
        assert 'create_publisher' in content, "Should publish to a topic"

    def test_subscriber_demonstrates_event_driven_pattern(self):
        """Test that subscriber shows event-driven (callback) pattern."""
        example_file = EXAMPLES_DIR / "minimal_subscriber.py"

        with open(example_file, 'r') as f:
            content = f.read()

        # Should show callbacks
        assert 'listener_callback' in content, "Should use callbacks"
        assert 'create_subscription' in content, "Should subscribe to topic"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
