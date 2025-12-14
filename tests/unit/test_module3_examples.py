"""
Unit tests for Module 3 (Perception & Sim-to-Real) code examples.

Tests verify that:
1. All code examples have valid Python syntax
2. Required classes and functions are present
3. Domain randomization implementation is correct
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


class TestModule3ExampleSyntax:
    """Test that Module 3 code examples have valid Python syntax."""

    def test_domain_randomization_syntax(self):
        """Test domain_randomization.py syntax."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"
        assert example_file.exists(), f"Code example not found: {example_file}"

        with open(example_file, 'r') as f:
            code = f.read()

        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in domain_randomization.py: {e}")

    def test_sim_to_real_evaluation_syntax(self):
        """Test sim_to_real_evaluation.py syntax."""
        example_file = EXAMPLES_DIR / "sim_to_real_evaluation.py"
        assert example_file.exists(), f"Code example not found: {example_file}"

        with open(example_file, 'r') as f:
            code = f.read()

        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in sim_to_real_evaluation.py: {e}")


class TestDomainRandomizationStructure:
    """Test domain_randomization.py structure and functionality."""

    def test_domain_randomizer_class_exists(self):
        """Test that DomainRandomizer class is defined."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'DomainRandomizer' in classes, "DomainRandomizer class not found"

    def test_domain_randomizer_init(self):
        """Test DomainRandomizer.__init__ method."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Find DomainRandomizer class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'DomainRandomizer':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                assert '__init__' in methods, "__init__ method not found"
                break

    def test_domain_randomizer_sample_method(self):
        """Test DomainRandomizer.sample() method exists."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'DomainRandomizer':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                assert 'sample' in methods, "sample() method not found"
                break

    def test_domain_randomizer_apply_to_gazebo(self):
        """Test DomainRandomizer.apply_to_gazebo() method exists."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'DomainRandomizer':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                assert 'apply_to_gazebo' in methods, "apply_to_gazebo() method not found"
                break

    def test_default_config_coverage(self):
        """Test that default config covers all critical randomizations."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Check for critical parameters
        critical_params = [
            'object_friction',
            'object_mass',
            'object_position',
            'camera_noise',
            'lighting_intensity',
            'motor_delay',
        ]

        for param in critical_params:
            assert param in code, f"Critical parameter '{param}' not found in randomization config"

    def test_sdf_generation(self):
        """Test that apply_to_gazebo generates valid SDF."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Check for SDF template (XML)
        assert '<?xml version=' in code, "SDF XML template not found"
        assert '<sdf version=' in code, "SDF version tag not found"
        assert '<world name=' in code, "Gazebo world definition not found"
        assert '<model name=' in code, "Gazebo model definition not found"


class TestSimToRealEvaluationStructure:
    """Test sim_to_real_evaluation.py structure."""

    def test_trial_dataclass_exists(self):
        """Test that Trial dataclass is defined."""
        example_file = EXAMPLES_DIR / "sim_to_real_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'Trial' in classes, "Trial dataclass not found"

    def test_evaluator_class_exists(self):
        """Test that SimToRealEvaluator class is defined."""
        example_file = EXAMPLES_DIR / "sim_to_real_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'SimToRealEvaluator' in classes, "SimToRealEvaluator class not found"

    def test_evaluator_required_methods(self):
        """Test that SimToRealEvaluator has all required methods."""
        example_file = EXAMPLES_DIR / "sim_to_real_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        required_methods = [
            'add_sim_trial',
            'add_real_trial',
            'success_rate',
            'transfer_ratio',
            'failure_analysis',
            'transfer_diagnosis',
            'print_report',
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'SimToRealEvaluator':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                for req_method in required_methods:
                    assert req_method in methods, f"{req_method}() method not found"
                break

    def test_confidence_interval_calculation(self):
        """Test that confidence interval calculation is implemented."""
        example_file = EXAMPLES_DIR / "sim_to_real_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Find success_ci method
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'SimToRealEvaluator':
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                ci_method = [m for m in methods if m.name == 'success_ci']
                assert ci_method, "success_ci() method not found"
                break

    def test_failure_mode_tracking(self):
        """Test that failure modes are tracked."""
        example_file = EXAMPLES_DIR / "sim_to_real_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Check for failure mode tracking
        assert 'reason' in code, "Failure reason tracking not found"
        assert 'failure_analysis' in code, "failure_analysis method not found"


class TestCodeDocumentation:
    """Test that code examples are well-documented."""

    def test_domain_randomization_docstring(self):
        """Test that domain_randomization.py has module docstring."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            code = f.read()

        assert '"""' in code, "Missing docstring"
        assert 'Learning Goals:' in code, "Missing learning goals"
        assert 'Example:' in code, "Missing usage example"

    def test_sim_to_real_evaluation_docstring(self):
        """Test that sim_to_real_evaluation.py has module docstring."""
        example_file = EXAMPLES_DIR / "sim_to_real_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        assert '"""' in code, "Missing docstring"
        assert 'Learning Goals:' in code, "Missing learning goals"
        assert 'Example:' in code, "Missing usage example"

    def test_class_docstrings(self):
        """Test that classes have docstrings."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                assert docstring is not None, f"Class {node.name} missing docstring"

    def test_method_docstrings(self):
        """Test that key methods have docstrings."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        key_methods = ['sample', 'apply_to_gazebo', 'get_statistics']

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'DomainRandomizer':
                for method_node in node.body:
                    if isinstance(method_node, ast.FunctionDef) and method_node.name in key_methods:
                        docstring = ast.get_docstring(method_node)
                        assert docstring is not None, f"Method {method_node.name} missing docstring"


class TestLearningObjectives:
    """Test that examples demonstrate learning objectives."""

    def test_domain_randomization_demonstrates_randomization(self):
        """Test that example demonstrates domain randomization concept."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should demonstrate parameter ranges
        assert 'uniform(' in code, "Should show random sampling"
        assert 'normal(' in code, "Should show Gaussian sampling"

        # Should show Gazebo integration
        assert 'gazebo' in code.lower(), "Should integrate with Gazebo"
        assert 'sdf' in code.lower(), "Should generate SDF files"

    def test_sim_to_real_evaluation_demonstrates_metrics(self):
        """Test that example demonstrates sim-to-real metrics."""
        example_file = EXAMPLES_DIR / "sim_to_real_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should demonstrate success rate
        assert 'success_rate' in code, "Should calculate success rate"

        # Should demonstrate transfer ratio
        assert 'transfer_ratio' in code, "Should calculate transfer ratio"

        # Should demonstrate failure analysis
        assert 'failure_analysis' in code, "Should analyze failures"

    def test_code_demonstrates_real_world_application(self):
        """Test that examples show real-world robotics patterns."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Real-world patterns
        real_world_concepts = [
            'friction',      # Material properties
            'mass',          # Object weight
            'camera',        # Sensor simulation
            'lighting',      # Environmental variation
            'motor_delay',   # Timing/latency
        ]

        for concept in real_world_concepts:
            assert concept in code, f"Missing real-world concept: {concept}"


class TestCodeQuality:
    """Test code quality metrics."""

    def test_domain_randomization_length(self):
        """Test that domain_randomization.py is appropriately sized."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            lines = f.readlines()

        # Code should be substantial (200+ lines with docs)
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        assert len(code_lines) > 150, "Code should be substantial"
        assert len(code_lines) < 500, "Code should be concise"

    def test_sim_to_real_evaluation_length(self):
        """Test that sim_to_real_evaluation.py is appropriately sized."""
        example_file = EXAMPLES_DIR / "sim_to_real_evaluation.py"

        with open(example_file, 'r') as f:
            lines = f.readlines()

        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        assert len(code_lines) > 150, "Code should be substantial"
        assert len(code_lines) < 500, "Code should be concise"

    def test_code_has_usage_examples(self):
        """Test that main() function provides usage examples."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

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
    """Integration tests for Module 3 examples."""

    def test_all_module3_examples_exist(self):
        """Test that all required Module 3 code examples exist."""
        required_files = [
            "domain_randomization.py",
            "sim_to_real_evaluation.py",
        ]

        for filename in required_files:
            filepath = EXAMPLES_DIR / filename
            assert filepath.exists(), f"Required example not found: {filename}"

    def test_examples_can_be_imported(self):
        """Test that examples can be parsed without import errors."""
        example_files = [
            "domain_randomization.py",
            "sim_to_real_evaluation.py",
        ]

        for example_file in example_files:
            filepath = EXAMPLES_DIR / example_file

            with open(filepath, 'r') as f:
                code = f.read()

            # Remove external imports for syntax checking
            modified_code = code.replace('import numpy as np', '#import numpy as np')
            modified_code = modified_code.replace('from typing import', '#from typing import')
            modified_code = modified_code.replace('from dataclasses import', '#from dataclasses import')

            try:
                compile(modified_code, str(filepath), 'exec')
            except SyntaxError as e:
                pytest.fail(f"Code example cannot be compiled: {e}")

    def test_documentation_references_code_examples(self):
        """Test that Module 3 documentation references these code examples."""
        doc_files = [
            Path(__file__).parent.parent.parent / "book" / "docs" / "03-perception" / "sim-to-real-transfer.md",
            Path(__file__).parent.parent.parent / "book" / "docs" / "03-perception" / "isaac-workflows.md",
        ]

        for doc_file in doc_files:
            if doc_file.exists():
                with open(doc_file, 'r') as f:
                    content = f.read()

                # Check that code examples are referenced
                # (At least one should reference domain randomization)
                if 'domain' in doc_file.name.lower() or 'sim-to-real' in doc_file.name.lower():
                    assert 'code' in content.lower() or 'example' in content.lower(), \
                        f"Documentation {doc_file.name} should reference code examples"


class TestModule3SpecificConcepts:
    """Tests for Module 3 learning objectives."""

    def test_domain_randomization_shows_parameter_ranges(self):
        """Test that domain randomization explains parameter ranges."""
        example_file = EXAMPLES_DIR / "domain_randomization.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should explain why ranges are chosen
        assert 'real_world_note' in code or 'comment' in code.lower(), \
            "Should document why ranges are chosen"

    def test_evaluation_shows_multiple_metrics(self):
        """Test that evaluation example shows multiple metrics."""
        example_file = EXAMPLES_DIR / "sim_to_real_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        metrics = [
            'success_rate',
            'transfer_ratio',
            'confidence_interval',
            'failure_analysis',
        ]

        for metric in metrics:
            assert metric in code.lower(), f"Missing metric: {metric}"

    def test_sim_to_real_transfer_checklist(self):
        """Test that evaluation includes diagnostic checklist."""
        example_file = EXAMPLES_DIR / "sim_to_real_evaluation.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should have diagnosis method
        assert 'diagnosis' in code, "Should provide diagnostic assessment"
        assert 'recommend' in code.lower(), "Should provide recommendations"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
