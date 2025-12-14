"""
Unit tests for Module 2 (Digital Twins & Gazebo) code examples.

Tests verify that:
1. All code examples have valid Python syntax
2. Required classes and functions are present
3. ROS 2 node structure is correct
4. Documentation is comprehensive
5. Examples are complete and runnable
"""

import pytest
import sys
from pathlib import Path
import ast
import subprocess

# Get the path to code examples
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "book" / "static" / "code-examples"


class TestModule2ExampleSyntax:
    """Test that Module 2 code examples have valid Python syntax."""

    def test_ros2_humanoid_nodes_syntax(self):
        """Test ros2_humanoid_nodes.py syntax."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"
        assert example_file.exists(), f"Code example not found: {example_file}"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should parse without SyntaxError
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in ros2_humanoid_nodes.py: {e}")


class TestModule2CodeStructure:
    """Test that code examples have correct structure and classes."""

    def test_humanoid_nodes_has_required_classes(self):
        """Test that humanoid_nodes.py has all required node classes."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Find class definitions
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        # Should have three node classes
        assert 'HumanoidController' in classes, "HumanoidController class not found"
        assert 'HumanoidMonitor' in classes, "HumanoidMonitor class not found"
        assert 'GripperController' in classes, "GripperController class not found"

    def test_humanoid_controller_has_required_methods(self):
        """Test HumanoidController has required methods."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Find HumanoidController class
        humanoid_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'HumanoidController':
                humanoid_class = node
                break

        assert humanoid_class is not None, "HumanoidController class not found"

        # Find methods
        methods = [n.name for n in humanoid_class.body if isinstance(n, ast.FunctionDef)]

        assert '__init__' in methods, "__init__ method not found"
        assert 'send_joint_commands' in methods, "send_joint_commands method not found"

    def test_humanoid_monitor_has_callback(self):
        """Test HumanoidMonitor has joint_state_callback."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Find HumanoidMonitor class
        monitor_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'HumanoidMonitor':
                monitor_class = node
                break

        assert monitor_class is not None, "HumanoidMonitor class not found"

        # Find methods
        methods = [n.name for n in monitor_class.body if isinstance(n, ast.FunctionDef)]

        assert '__init__' in methods, "__init__ method not found"
        assert 'joint_state_callback' in methods, "joint_state_callback method not found"


class TestModule2ROS2Integration:
    """Test that code examples properly use ROS 2."""

    def test_humanoid_nodes_imports_rclpy(self):
        """Test that humanoid_nodes imports rclpy."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should import rclpy
        assert 'import rclpy' in code, "Should import rclpy"
        assert 'from rclpy.node import Node' in code, "Should import Node from rclpy"

    def test_humanoid_nodes_uses_message_types(self):
        """Test that humanoid_nodes uses appropriate message types."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should use JointState for communication
        assert 'JointState' in code, "Should use JointState messages"
        assert 'create_publisher' in code, "Should create publishers"
        assert 'create_subscription' in code, "Should create subscriptions"

    def test_humanoid_nodes_has_main_function(self):
        """Test that humanoid_nodes has a main() entry point."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Find main function
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        assert 'main' in functions, "main() function not found"

    def test_humanoid_nodes_has_entry_point(self):
        """Test that humanoid_nodes has proper __main__ entry point."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should have if __name__ == '__main__'
        assert "if __name__ == '__main__':" in code, "Missing __main__ entry point"
        assert 'main()' in code, "main() not called in entry point"


class TestModule2Documentation:
    """Test that code examples are properly documented."""

    def test_humanoid_nodes_has_module_docstring(self):
        """Test that humanoid_nodes.py has a module docstring."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should have module docstring
        assert '"""' in code, "Missing module docstring"
        assert 'Humanoid' in code, "Docstring should mention Humanoid"
        assert 'ROS 2' in code, "Docstring should mention ROS 2"

    def test_humanoid_controller_has_docstring(self):
        """Test HumanoidController class has docstring."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Find HumanoidController
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'HumanoidController':
                docstring = ast.get_docstring(node)
                assert docstring is not None, "HumanoidController missing docstring"
                assert 'joint' in docstring.lower(), "Docstring should explain joints"
                break

    def test_send_joint_commands_has_docstring(self):
        """Test send_joint_commands method has documentation."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Find send_joint_commands method
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'send_joint_commands':
                docstring = ast.get_docstring(node)
                assert docstring is not None, "send_joint_commands missing docstring"
                break

    def test_main_function_has_docstring(self):
        """Test main() function has comprehensive docstring."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Find main function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'main':
                docstring = ast.get_docstring(node)
                assert docstring is not None, "main() missing docstring"
                assert 'Usage:' in docstring, "Should have usage examples"
                break


class TestModule2CodeContent:
    """Test that code examples contain expected content."""

    def test_humanoid_controller_has_joint_names(self):
        """Test HumanoidController defines all joint names."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should list joints
        assert 'joint_names' in code, "Should define joint_names"
        assert 'shoulder_pitch' in code, "Should include shoulder joints"
        assert 'knee_pitch' in code, "Should include knee joints"
        assert 'gripper' in code, "Should include gripper joints"

    def test_humanoid_controller_uses_timer(self):
        """Test HumanoidController uses ROS 2 timer for periodic commands."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should use create_timer
        assert 'create_timer' in code, "Should use ROS 2 timer"

    def test_humanoid_monitor_validates_feedback(self):
        """Test HumanoidMonitor validates joint feedback."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should check for anomalies
        assert 'velocity' in code.lower(), "Should check velocity"
        assert 'effort' in code.lower() or 'torque' in code.lower(), "Should check effort/torque"

    def test_gripper_controller_has_state_machine(self):
        """Test GripperController implements state machine for gripper control."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should have state machine logic
        assert 'state' in code, "Should have state variable"
        assert 'OPEN' in code or '0.1' in code, "Should specify open position"
        assert 'CLOSED' in code or '0.0' in code, "Should specify closed position"


class TestModule2CodeQuality:
    """Test code quality metrics."""

    def test_humanoid_nodes_length(self):
        """Test that humanoid_nodes.py is appropriately sized."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            lines = f.readlines()

        # Code should be substantial (~300+ lines with docs)
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        assert len(code_lines) > 150, "Code should be substantial"
        assert len(code_lines) < 500, "Code should be concise and readable"

    def test_humanoid_nodes_has_comments(self):
        """Test that code has helpful inline comments."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should have explanatory comments
        assert '#' in code, "Should have explanatory comments"

    def test_humanoid_nodes_uses_proper_naming(self):
        """Test that code uses Python naming conventions."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Check function names are lowercase_with_underscores
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        # Most function names should follow convention
        snake_case_funcs = [f for f in functions if '_' in f or f == 'main']
        assert len(snake_case_funcs) > len(functions) / 2, "Functions should use snake_case"


class TestModule2ExampleIntegration:
    """Integration tests for Module 2 examples."""

    def test_all_module2_examples_exist(self):
        """Test that all required Module 2 code examples exist."""
        assert (EXAMPLES_DIR / "ros2_humanoid_nodes.py").exists(), \
            "ros2_humanoid_nodes.py not found"

    def test_example_can_be_imported(self):
        """Test that example code can be parsed without import errors."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Remove rclpy imports for syntax checking (rclpy may not be installed in test env)
        modified_code = code.replace('import rclpy', '#import rclpy')
        modified_code = modified_code.replace('from rclpy', '#from rclpy')

        # Should still parse
        try:
            compile(modified_code, str(example_file), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Code example cannot be compiled: {e}")

    def test_example_demonstrates_core_concepts(self):
        """Test that example demonstrates key ROS 2 + Gazebo concepts."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should demonstrate:
        # 1. Node creation
        assert 'Node' in code, "Should demonstrate Node creation"

        # 2. Publishers and subscribers
        assert 'create_publisher' in code, "Should demonstrate publishers"
        assert 'create_subscription' in code, "Should demonstrate subscribers"

        # 3. Callbacks
        assert 'callback' in code, "Should demonstrate callbacks"

        # 4. Timers for periodic tasks
        assert 'create_timer' in code, "Should demonstrate timers"

        # 5. Multi-threaded execution
        assert 'MultiThreadedExecutor' in code or 'Executor' in code, \
            "Should demonstrate executor pattern"

    def test_urdf_files_exist(self):
        """Test that URDF files exist."""
        gazebo_models_dir = Path(__file__).parent.parent.parent / \
            "book" / "examples" / "humanoid-sim" / "gazebo_models"

        assert (gazebo_models_dir / "humanoid_simple.urdf").exists(), \
            "humanoid_simple.urdf not found"

    def test_world_file_exists(self):
        """Test that Gazebo world file exists."""
        gazebo_models_dir = Path(__file__).parent.parent.parent / \
            "book" / "examples" / "humanoid-sim" / "gazebo_models"

        assert (gazebo_models_dir / "simple_world.sdf").exists(), \
            "simple_world.sdf not found"


@pytest.mark.unit
class TestModule2SpecificConcepts:
    """Tests specific to Module 2 learning objectives."""

    def test_example_shows_distributed_control(self):
        """Test that example demonstrates distributed node architecture."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should have multiple nodes
        assert 'HumanoidController' in code, "Should show controller node"
        assert 'HumanoidMonitor' in code, "Should show monitor node"
        assert 'GripperController' in code, "Should show gripper node"

    def test_example_shows_feedback_loops(self):
        """Test that example demonstrates closed-loop control with feedback."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Monitor subscribes to feedback
        assert 'joint_state_callback' in code, "Should have feedback callback"
        # Controller publishes commands
        assert 'joint_command' in code, "Should publish joint commands"

    def test_example_shows_synchronization(self):
        """Test that example shows how to synchronize multiple joints."""
        example_file = EXAMPLES_DIR / "ros2_humanoid_nodes.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should show coordinated multi-joint control
        assert 'joint_names' in code, "Should list multiple joints"
        assert 'positions' in code, "Should show coordinated positions"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
