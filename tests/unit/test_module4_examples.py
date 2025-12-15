"""
Unit tests for Module 4 (VLA Systems) code examples.

Tests verify that:
1. All code examples have valid Python syntax.
2. The VLA pipeline components (speech, planning, execution) are well-structured.
3. The LLM planner correctly constructs prompts and parses responses.
4. The Action Executor can process a plan.
5. Mocking is used to isolate components for unit testing.
"""

import pytest
from pathlib import Path
import ast
from unittest.mock import patch, MagicMock

# Get the path to code examples
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "book" / "static" / "code-examples"

# Import the code to be tested
# It's better to add the examples directory to the path and import directly
import sys
sys.path.insert(0, str(EXAMPLES_DIR))

import llm_task_planner
import action_executor
import speech_to_text

class TestModule4ExampleSyntax:
    """Test that Module 4 code examples have valid Python syntax."""

    @pytest.mark.parametrize("filename", [
        "speech_to_text.py",
        "llm_task_planner.py",
        "action_executor.py"
    ])
    def test_file_syntax(self, filename):
        """Generic syntax test for a given file."""
        example_file = EXAMPLES_DIR / filename
        assert example_file.exists(), f"Code example not found: {example_file}"

        with open(example_file, 'r') as f:
            code = f.read()

        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {filename}: {e}")

class TestLlmTaskPlanner:
    """Tests for the llm_task_planner.py script."""

    def test_prompt_construction(self):
        """Test that the prompt is constructed correctly."""
        action_dict = {"navigate": {"description": "go", "parameters": {"location": "string"}}}
        command = "go to the kitchen"
        prompt = llm_task_planner.get_planning_prompt(command, action_dict)
        
        assert "You are an AI assistant for a robot" in prompt
        assert json.dumps(action_dict, indent=4) in prompt
        assert command in prompt
        assert "Here are a few examples" in prompt

    def test_response_parsing(self):
        """Test that a valid JSON response is parsed correctly."""
        mock_response = """
        [
            {"action": "navigate", "parameters": {"location": "kitchen"}}
        ]
        """
        plan = llm_task_planner.parse_llm_response(mock_response)
        assert isinstance(plan, list)
        assert len(plan) == 1
        assert plan[0]["action"] == "navigate"

    def test_invalid_json_parsing(self):
        """Test that invalid JSON is handled gracefully."""
        mock_response = "this is not json"
        plan = llm_task_planner.parse_llm_response(mock_response)
        assert plan is None

    @patch('llm_task_planner.query_llm')
    def test_main_logic_with_mock_llm(self, mock_query_llm):
        """Test the main function by mocking the LLM query."""
        
        mock_response = """
        [
            {"action": "navigate", "parameters": {"location": "test_location"}}
        ]
        """
        mock_query_llm.return_value = mock_response
        
        # We can't easily test the full main(), but we can test a function that encapsulates the logic
        def run_planner():
            action_dictionary = {"navigate": {"description": "...", "parameters": {}}}
            prompt = llm_task_planner.get_planning_prompt("test command", action_dictionary)
            response = llm_task_planner.query_llm(prompt)
            return llm_task_planner.parse_llm_response(response)

        plan = run_planner()
        
        mock_query_llm.assert_called_once()
        assert len(plan) == 1
        assert plan[0]["action"] == "navigate"

class TestActionExecutor:
    """Tests for the action_executor.py script."""

    @patch('action_executor.MockActionClient')
    def test_executor_initialization(self, MockActionClient):
        """Test that the executor initializes its action clients."""
        executor = action_executor.ActionExecutor()
        assert "navigate" in executor.action_clients
        assert "grasp" in executor.action_clients
        # Check that the mock was called for each client
        assert MockActionClient.call_count == 4

    def test_plan_execution_success(self):
        """Test a successful execution of a simple plan."""
        plan = [{"action": "navigate", "parameters": {"location": "home"}}]
        
        executor = action_executor.ActionExecutor()
        
        # Mock the send_goal method of the specific client
        executor.action_clients["navigate"].send_goal = MagicMock(return_value=True)
        
        result = executor.execute_plan(plan)
        
        assert result is True
        executor.action_clients["navigate"].send_goal.assert_called_once_with({"location": "home"})

    def test_plan_execution_failure(self):
        """Test that plan execution aborts on failure."""
        plan = [
            {"action": "navigate", "parameters": {"location": "home"}},
            {"action": "grasp", "parameters": {"object_name": "ball"}}
        ]
        
        executor = action_executor.ActionExecutor()
        
        # Mock the send_goal methods
        executor.action_clients["navigate"].send_goal = MagicMock(return_value=False)
        executor.action_clients["grasp"].send_goal = MagicMock()
        
        result = executor.execute_plan(plan)
        
        assert result is False
        # The first action should be called
        executor.action_clients["navigate"].send_goal.assert_called_once()
        # But the second should not
        executor.action_clients["grasp"].send_goal.assert_not_called()

class TestSpeechToText:
    """Tests for the speech_to_text.py script."""

    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    def test_listen_and_transcribe_success(self, MockMicrophone, MockRecognizer):
        """Test the main speech-to-text function with mocked libraries."""
        
        mock_recognizer_instance = MockRecognizer.return_value
        mock_microphone_instance = MockMicrophone.return_value
        
        # Configure the mock to return a successful transcription
        mock_recognizer_instance.recognize_google.return_value = "hello world"
        
        result = speech_to_text.listen_and_transcribe(mock_recognizer_instance, mock_microphone_instance)
        
        assert result == "hello world"
        mock_recognizer_instance.adjust_for_ambient_noise.assert_called_once()
        mock_recognizer_instance.listen.assert_called_once()
        mock_recognizer_instance.recognize_google.assert_called_once()
        
if __name__ == '__main__':
    pytest.main([__file__, '-v'])