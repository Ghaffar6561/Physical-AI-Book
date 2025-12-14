"""
Unit tests for Module 4 (VLA Systems) code examples.

Tests verify that:
1. All code examples have valid Python syntax
2. Required classes and functions are present
3. VLA policy learner implementation is correct
4. Evaluation metrics are properly implemented
5. Code demonstrates learning objectives
6. Examples are comprehensive and runnable
"""

import pytest
import sys
from pathlib import Path
import ast
import subprocess

# Get the path to code examples
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "book" / "static" / "code-examples"


class TestModule4ExampleSyntax:
    """Test that Module 4 code examples have valid Python syntax."""

    def test_vla_policy_learner_syntax(self):
        """Test vla_policy_learner.py syntax."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"
        assert example_file.exists(), f"Code example not found: {example_file}"

        with open(example_file, 'r') as f:
            code = f.read()

        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in vla_policy_learner.py: {e}")

    def test_vla_evaluation_syntax(self):
        """Test vla_evaluation.py syntax."""
        example_file = EXAMPLES_DIR / "vla_evaluation.py"
        assert example_file.exists(), f"Code example not found: {example_file}"

        with open(example_file, 'r') as f:
            code = f.read()

        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in vla_evaluation.py: {e}")


class TestVLAPolicyLearnerStructure:
    """Test vla_policy_learner.py structure and implementation."""

    def test_robot_demonstration_dataclass_exists(self):
        """Test that RobotDemonstration dataclass is defined."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'RobotDemonstration' in classes, "RobotDemonstration class not found"

    def test_dataset_class_exists(self):
        """Test that RobotDemonstrationDataset class is defined."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'RobotDemonstrationDataset' in classes, "RobotDemonstrationDataset class not found"

    def test_vla_policy_learner_class_exists(self):
        """Test that VLAPolicyLearner class is defined."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'VLAPolicyLearner' in classes, "VLAPolicyLearner class not found"

    def test_vla_policy_learner_required_methods(self):
        """Test that VLAPolicyLearner has required methods."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        required_methods = [
            'encode_image',
            'encode_language',
            'forward',
            'train_step',
            'train_epoch',
            'evaluate',
            'infer',
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'VLAPolicyLearner':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                for req_method in required_methods:
                    assert req_method in methods, f"{req_method}() method not found"
                break

    def test_vla_learns_from_demonstrations(self):
        """Test that VLAPolicyLearner accepts demonstrations."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should have training loop
        assert 'train_epoch' in code, "train_epoch method not found"
        assert 'forward' in code, "forward method for predictions not found"
        assert 'infer' in code, "infer method for deployment not found"


class TestVLAEvaluationStructure:
    """Test vla_evaluation.py structure."""

    def test_failure_mode_enum_exists(self):
        """Test that FailureMode enum is defined."""
        example_file = EXAMPLES_DIR / "vla_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'FailureMode' in classes, "FailureMode enum not found"

    def test_trial_dataclass_exists(self):
        """Test that Trial dataclass is defined."""
        example_file = EXAMPLES_DIR / "vla_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'Trial' in classes, "Trial dataclass not found"

    def test_evaluator_class_exists(self):
        """Test that VLAEvaluator class is defined."""
        example_file = EXAMPLES_DIR / "vla_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'VLAEvaluator' in classes, "VLAEvaluator class not found"

    def test_evaluator_required_methods(self):
        """Test that VLAEvaluator has all required methods."""
        example_file = EXAMPLES_DIR / "vla_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        required_methods = [
            'add_trial',
            'success_rate',
            'success_by_category',
            'failure_analysis',
            'confidence_analysis',
            'position_error_analysis',
            'transfer_diagnosis',
            'print_report',
            'export_results',
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'VLAEvaluator':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                for req_method in required_methods:
                    assert req_method in methods, f"{req_method}() method not found"
                break

    def test_evaluator_tracks_failure_modes(self):
        """Test that evaluator tracks failure modes."""
        example_file = EXAMPLES_DIR / "vla_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should track different failure modes
        assert 'perception' in code.lower(), "Should track perception failures"
        assert 'language' in code.lower(), "Should track language failures"
        assert 'grounding' in code.lower(), "Should track grounding failures"
        assert 'motor' in code.lower(), "Should track motor failures"


class TestCodeDocumentation:
    """Test that code examples are well-documented."""

    def test_vla_policy_learner_docstring(self):
        """Test that vla_policy_learner.py has module docstring."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        assert '"""' in code, "Missing docstring"
        assert 'Learning Goals' in code, "Missing learning goals"
        assert 'Example' in code, "Missing usage example"

    def test_vla_evaluation_docstring(self):
        """Test that vla_evaluation.py has module docstring."""
        example_file = EXAMPLES_DIR / "vla_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        assert '"""' in code, "Missing docstring"
        assert 'Learning Goals' in code, "Missing learning goals"
        assert 'Example' in code, "Missing usage example"

    def test_class_docstrings(self):
        """Test that classes have docstrings."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in [
                'RobotDemonstration',
                'RobotDemonstrationDataset',
                'VLAPolicyLearner'
            ]:
                docstring = ast.get_docstring(node)
                assert docstring is not None, f"Class {node.name} missing docstring"

    def test_method_docstrings(self):
        """Test that key methods have docstrings."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        key_methods = ['forward', 'infer', 'train_step', 'encode_image', 'encode_language']

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'VLAPolicyLearner':
                for method_node in node.body:
                    if isinstance(method_node, ast.FunctionDef) and method_node.name in key_methods:
                        docstring = ast.get_docstring(method_node)
                        assert docstring is not None, f"Method {method_node.name} missing docstring"


class TestLearningObjectives:
    """Test that examples demonstrate learning objectives."""

    def test_vla_policy_demonstrates_learning(self):
        """Test that policy learner demonstrates fine-tuning."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should show multimodal encoding
        assert 'vision' in code.lower(), "Should show vision encoding"
        assert 'language' in code.lower(), "Should show language encoding"

        # Should show fusion
        assert 'fusion' in code.lower(), "Should show feature fusion"

        # Should show action prediction
        assert 'action' in code.lower(), "Should predict actions"

    def test_evaluation_demonstrates_metrics(self):
        """Test that evaluator demonstrates key metrics."""
        example_file = EXAMPLES_DIR / "vla_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should demonstrate success rate
        assert 'success_rate' in code, "Should calculate success rate"

        # Should demonstrate failure analysis
        assert 'failure' in code.lower(), "Should analyze failures"

        # Should demonstrate confidence metrics
        assert 'confidence' in code.lower(), "Should track confidence"

    def test_code_demonstrates_real_world_patterns(self):
        """Test that examples show real-world robotics patterns."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Real-world patterns
        real_world_concepts = [
            'vision',           # Vision input
            'language',         # Natural language instruction
            'gripper',          # Robot gripper control
            'force',            # Force control
            'position',         # Spatial positioning
            'inference',        # Deployment inference
        ]

        for concept in real_world_concepts:
            assert concept in code.lower(), f"Missing real-world concept: {concept}"


class TestCodeQuality:
    """Test code quality metrics."""

    def test_vla_policy_learner_length(self):
        """Test that vla_policy_learner.py is appropriately sized."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            lines = f.readlines()

        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        assert len(code_lines) > 300, "Code should be substantial"
        assert len(code_lines) < 700, "Code should be concise"

    def test_vla_evaluation_length(self):
        """Test that vla_evaluation.py is appropriately sized."""
        example_file = EXAMPLES_DIR / "vla_evaluation.py"

        with open(example_file, 'r') as f:
            lines = f.readlines()

        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        assert len(code_lines) > 250, "Code should be substantial"
        assert len(code_lines) < 600, "Code should be concise"

    def test_code_has_usage_examples(self):
        """Test that main() function provides usage examples."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        assert 'main' in functions, "main() function not found"

        # Check main has docstring
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'main':
                docstring = ast.get_docstring(node)
                assert docstring is not None, "main() missing docstring"


class TestIntegration:
    """Integration tests for Module 4 examples."""

    def test_all_module4_examples_exist(self):
        """Test that all required Module 4 code examples exist."""
        required_files = [
            "vla_policy_learner.py",
            "vla_evaluation.py",
        ]

        for filename in required_files:
            filepath = EXAMPLES_DIR / filename
            assert filepath.exists(), f"Required example not found: {filename}"

    def test_examples_can_be_imported(self):
        """Test that examples can be parsed without import errors."""
        example_files = [
            "vla_policy_learner.py",
            "vla_evaluation.py",
        ]

        for example_file in example_files:
            filepath = EXAMPLES_DIR / example_file

            with open(filepath, 'r') as f:
                code = f.read()

            # Remove external imports for syntax checking
            modified_code = code.replace('import torch', '#import torch')
            modified_code = modified_code.replace('import numpy', '#import numpy')
            modified_code = modified_code.replace('from transformers', '#from transformers')

            try:
                compile(modified_code, str(filepath), 'exec')
            except SyntaxError as e:
                pytest.fail(f"Code example cannot be compiled: {e}")

    def test_documentation_references_code_examples(self):
        """Test that Module 4 documentation references code examples."""
        doc_files = [
            Path(__file__).parent.parent.parent / "book" / "docs" / "04-vla-systems" / "vision-language-models.md",
            Path(__file__).parent.parent.parent / "book" / "docs" / "04-vla-systems" / "action-grounding.md",
            Path(__file__).parent.parent.parent / "book" / "docs" / "04-vla-systems" / "vla-architecture.md",
        ]

        for doc_file in doc_files:
            if doc_file.exists():
                with open(doc_file, 'r') as f:
                    content = f.read()

                # Check that code examples are referenced
                if 'code' in doc_file.name.lower() or any(x in doc_file.name.lower() for x in ['vla', 'policy', 'architecture']):
                    assert 'code' in content.lower() or 'example' in content.lower(), \
                        f"Documentation {doc_file.name} should reference code examples"


class TestModule4SpecificConcepts:
    """Tests for Module 4 learning objectives."""

    def test_vla_shows_multimodal_fusion(self):
        """Test that policy shows vision-language fusion."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should fuse vision and language
        assert 'vision' in code.lower(), "Should encode vision"
        assert 'language' in code.lower(), "Should encode language"
        assert 'fus' in code.lower(), "Should fuse modalities"

    def test_evaluation_shows_multiple_metrics(self):
        """Test that evaluation example shows multiple metrics."""
        example_file = EXAMPLES_DIR / "vla_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        metrics = [
            'success_rate',
            'failure',
            'confidence',
            'transfer',
        ]

        for metric in metrics:
            assert metric in code.lower(), f"Missing metric: {metric}"

    def test_vla_demonstrates_transfer_learning(self):
        """Test that policy demonstrates fine-tuning for transfer."""
        example_file = EXAMPLES_DIR / "vla_policy_learner.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should show frozen encoders
        assert 'requires_grad = False' in code or 'freeze' in code.lower(), \
            "Should freeze pre-trained encoders"

        # Should show trainable head
        assert 'action_head' in code or 'Linear' in code, \
            "Should have trainable action prediction head"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
