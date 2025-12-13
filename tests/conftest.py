"""
Pytest configuration and shared fixtures for Physical AI Book tests.

This module provides:
- ROS 2 node fixtures for testing
- Mock LLM fixtures for VLA testing
- Gazebo environment fixtures
- Common test utilities
"""

import pytest
import sys
from pathlib import Path

# Add source directories to path
TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent
BOOK_DIR = REPO_ROOT / "book"
EXAMPLES_DIR = REPO_ROOT / "book" / "examples"

sys.path.insert(0, str(BOOK_DIR / "static" / "code-examples"))
sys.path.insert(0, str(EXAMPLES_DIR))


# ============================================================================
# ROS 2 Fixtures (for testing ROS 2 nodes)
# ============================================================================

@pytest.fixture(scope="session")
def ros2_available():
    """Check if ROS 2 is available in the test environment."""
    try:
        import rclpy
        return True
    except ImportError:
        return False


@pytest.fixture
def ros2_node(ros2_available):
    """
    Create a minimal ROS 2 node for testing.

    Usage:
        def test_publisher(ros2_node):
            publisher = ros2_node.create_publisher(String, 'test_topic', 10)
            assert publisher is not None
    """
    if not ros2_available:
        pytest.skip("ROS 2 not available")

    import rclpy
    from rclpy.node import Node

    class TestNode(Node):
        def __init__(self):
            super().__init__('test_node')

    if not rclpy.ok():
        rclpy.init()

    node = TestNode()
    yield node

    # Cleanup
    node.destroy_node()


# ============================================================================
# LLM Fixtures (for testing VLA systems)
# ============================================================================

@pytest.fixture
def mock_llm():
    """
    Mock LLM for testing VLA systems without API calls.

    Usage:
        def test_llm_planning(mock_llm):
            response = mock_llm.generate("pick up the object")
            assert "navigate" in response.lower()
    """
    class MockLLM:
        def generate(self, prompt, max_tokens=256, temperature=0.7):
            """Generate mock LLM responses for known prompts."""
            prompt_lower = prompt.lower()

            # Mock responses for common test prompts
            if "pick up" in prompt_lower:
                return "execute action: navigate_to(object), grasp_object()"
            elif "navigate" in prompt_lower:
                return "execute action: move_to(target_location)"
            elif "fly" in prompt_lower or "moon" in prompt_lower:
                return "error: infeasible action - robot cannot fly"
            else:
                return "execute action: idle()"

        def get_embedding(self, text):
            """Mock embedding generation."""
            return [0.1] * 768  # 768-dim embedding

    return MockLLM()


@pytest.fixture
def mock_speech_recognizer():
    """
    Mock speech recognizer for testing voice interface.

    Usage:
        def test_voice_interface(mock_speech_recognizer):
            text = mock_speech_recognizer.recognize_audio(mock_audio)
            assert text == "pick up the red object"
    """
    class MockSpeechRecognizer:
        def recognize_audio(self, audio_data):
            """Mock speech-to-text."""
            # In real tests, would process actual audio
            return "pick up the red object"

        def recognize_microphone(self):
            """Mock listening from microphone."""
            return "navigate to the table"

    return MockSpeechRecognizer()


# ============================================================================
# Gazebo Fixtures (for simulation testing)
# ============================================================================

@pytest.fixture
def gazebo_available():
    """Check if Gazebo is available."""
    import subprocess
    try:
        subprocess.run(
            ["which", "gazebo"],
            capture_output=True,
            check=True,
            timeout=2
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@pytest.fixture
def gazebo_world():
    """
    Provide a minimal Gazebo world for testing.

    Returns the path to a test world file.
    """
    world_content = """<?xml version='1.0'?>
<sdf version='1.7'>
  <world name='test_world'>
    <physics type='ode'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
    </scene>
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>
  </world>
</sdf>"""

    world_file = Path(REPO_ROOT) / "tests" / "test_world.sdf"
    world_file.write_text(world_content)

    yield world_file

    # Cleanup
    world_file.unlink(missing_ok=True)


# ============================================================================
# Common Test Utilities
# ============================================================================

@pytest.fixture
def code_examples_dir():
    """Provide path to code examples directory."""
    return BOOK_DIR / "static" / "code-examples"


@pytest.fixture
def test_data_dir():
    """Provide path to test data directory."""
    test_data = REPO_ROOT / "tests" / "data"
    test_data.mkdir(exist_ok=True)
    return test_data


@pytest.fixture
def temporary_module_cache(tmp_path):
    """
    Provide a temporary directory for module caching.

    Useful for tests that need isolated Python module imports.
    """
    sys.path.insert(0, str(tmp_path))
    yield tmp_path
    sys.path.remove(str(tmp_path))


# ============================================================================
# Markers and Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "ros2: mark test as requiring ROS 2 environment"
    )
    config.addinivalue_line(
        "markers",
        "gazebo: mark test as requiring Gazebo simulator"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their location or content.

    - Tests in tests/unit/ are marked with 'unit'
    - Tests in tests/integration/ are marked with 'integration'
    - Tests in tests/capstone/ are marked with 'capstone'
    """
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "capstone" in str(item.fspath):
            item.add_marker(pytest.mark.capstone)
