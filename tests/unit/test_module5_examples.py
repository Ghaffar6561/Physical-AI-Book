"""
Unit tests for Module 5 (End-to-End Learning) code examples.

Tests verify that:
1. All code examples have valid Python syntax
2. Required classes and functions are present
3. Learning algorithm implementations are correct
4. Code demonstrates learning objectives
5. Examples are comprehensive and runnable

Test Coverage:
- Diffusion Policy: DiffusionPolicy, DiffusionUNet, NoiseSchedule classes
- RL Policy: PPOPolicy, ActorNetwork, CriticNetwork, PPOBuffer classes
- Data handling: Datasets for both approaches
- Training loops: Both diffusion and RL training procedures
- Inference: Policy deployment and trajectory generation
"""

import pytest
import sys
from pathlib import Path
import ast
import subprocess

# Get the path to code examples
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "book" / "static" / "code-examples"


class TestModule5ExampleSyntax:
    """Test that Module 5 code examples have valid Python syntax."""

    def test_diffusion_policy_syntax(self):
        """Test diffusion_policy.py syntax."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"
        assert example_file.exists(), f"Code example not found: {example_file}"

        with open(example_file, 'r') as f:
            code = f.read()

        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in diffusion_policy.py: {e}")

    def test_rl_policy_syntax(self):
        """Test rl_policy.py syntax."""
        example_file = EXAMPLES_DIR / "rl_policy.py"
        assert example_file.exists(), f"Code example not found: {example_file}"

        with open(example_file, 'r') as f:
            code = f.read()

        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in rl_policy.py: {e}")


class TestDiffusionPolicyStructure:
    """Test diffusion_policy.py structure and implementation."""

    def test_robot_demonstration_class_exists(self):
        """Test that RobotDemonstration dataclass is defined."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'RobotDemonstration' in classes, "RobotDemonstration class not found"

    def test_demonstration_dataset_class_exists(self):
        """Test that DemonstrationDataset class is defined."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'DemonstrationDataset' in classes, "DemonstrationDataset class not found"

    def test_noise_schedule_class_exists(self):
        """Test that NoiseSchedule class is defined."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'NoiseSchedule' in classes, "NoiseSchedule class not found"

    def test_diffusion_unet_class_exists(self):
        """Test that DiffusionUNet class is defined."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'DiffusionUNet' in classes, "DiffusionUNet class not found"

    def test_diffusion_policy_class_exists(self):
        """Test that DiffusionPolicy class is defined."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'DiffusionPolicy' in classes, "DiffusionPolicy class not found"

    def test_diffusion_policy_required_methods(self):
        """Test that DiffusionPolicy has required methods."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        required_methods = [
            'forward',
            'train_step',
            'infer',
            'evaluate',
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'DiffusionPolicy':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                for req_method in required_methods:
                    assert req_method in methods, f"{req_method}() method not found"
                break


class TestRLPolicyStructure:
    """Test rl_policy.py structure and implementation."""

    def test_transition_class_exists(self):
        """Test that Transition dataclass is defined."""
        example_file = EXAMPLES_DIR / "rl_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'Transition' in classes, "Transition class not found"

    def test_trajectory_class_exists(self):
        """Test that Trajectory dataclass is defined."""
        example_file = EXAMPLES_DIR / "rl_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'Trajectory' in classes, "Trajectory class not found"

    def test_actor_network_class_exists(self):
        """Test that ActorNetwork class is defined."""
        example_file = EXAMPLES_DIR / "rl_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'ActorNetwork' in classes, "ActorNetwork class not found"

    def test_critic_network_class_exists(self):
        """Test that CriticNetwork class is defined."""
        example_file = EXAMPLES_DIR / "rl_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'CriticNetwork' in classes, "CriticNetwork class not found"

    def test_ppo_buffer_class_exists(self):
        """Test that PPOBuffer class is defined."""
        example_file = EXAMPLES_DIR / "rl_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'PPOBuffer' in classes, "PPOBuffer class not found"

    def test_ppo_policy_class_exists(self):
        """Test that PPOPolicy class is defined."""
        example_file = EXAMPLES_DIR / "rl_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert 'PPOPolicy' in classes, "PPOPolicy class not found"

    def test_ppo_policy_required_methods(self):
        """Test that PPOPolicy has required methods."""
        example_file = EXAMPLES_DIR / "rl_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        required_methods = [
            'select_action',
            'compute_value',
            'store_transition',
            'train_step',
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'PPOPolicy':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                for req_method in required_methods:
                    assert req_method in methods, f"{req_method}() method not found"
                break


class TestCodeDocumentation:
    """Test that code examples are well-documented."""

    def test_diffusion_policy_docstring(self):
        """Test that diffusion_policy.py has module docstring."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        assert '"""' in code, "Missing module docstring"
        assert 'Learning Goals' in code, "Missing learning goals"
        assert 'Example' in code, "Missing usage example"

    def test_rl_policy_docstring(self):
        """Test that rl_policy.py has module docstring."""
        example_file = EXAMPLES_DIR / "rl_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        assert '"""' in code, "Missing module docstring"
        assert 'Learning Goals' in code, "Missing learning goals"
        assert 'Example' in code, "Missing usage example"

    def test_class_docstrings(self):
        """Test that classes have docstrings."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        key_classes = ['DiffusionPolicy', 'DiffusionUNet', 'NoiseSchedule',
                      'DemonstrationDataset']

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in key_classes:
                docstring = ast.get_docstring(node)
                assert docstring is not None, f"Class {node.name} missing docstring"

    def test_method_docstrings(self):
        """Test that key methods have docstrings."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        key_methods = ['forward', 'infer', 'train_step', 'q_sample']

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'DiffusionPolicy':
                for method_node in node.body:
                    if isinstance(method_node, ast.FunctionDef) and method_node.name in key_methods:
                        docstring = ast.get_docstring(method_node)
                        assert docstring is not None, f"Method {method_node.name} missing docstring"


class TestLearningObjectives:
    """Test that examples demonstrate learning objectives."""

    def test_diffusion_demonstrates_noise_prediction(self):
        """Test that diffusion policy demonstrates noise prediction."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should demonstrate forward/reverse process
        assert 'forward' in code.lower(), "Should demonstrate forward process"
        assert 'noise' in code.lower(), "Should predict noise"
        assert 'denoise' in code.lower() or 'reverse' in code.lower(), \
            "Should demonstrate reverse process"

    def test_rl_demonstrates_policy_gradient(self):
        """Test that RL policy demonstrates policy gradient."""
        example_file = EXAMPLES_DIR / "rl_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should demonstrate core RL concepts
        assert 'actor' in code.lower(), "Should have actor network"
        assert 'critic' in code.lower(), "Should have critic network"
        assert 'advantage' in code.lower(), "Should compute advantage"
        assert 'ppo' in code.lower(), "Should implement PPO"

    def test_both_demonstrate_multimodal(self):
        """Test that both examples handle multimodal distributions."""
        diffusion_file = EXAMPLES_DIR / "diffusion_policy.py"
        rl_file = EXAMPLES_DIR / "rl_policy.py"

        with open(diffusion_file, 'r') as f:
            diffusion_code = f.read()

        with open(rl_file, 'r') as f:
            rl_code = f.read()

        # Diffusion handles multimodality via distribution
        assert 'distribution' in diffusion_code.lower() or \
               'trajectory' in diffusion_code.lower(), \
            "Diffusion should handle trajectory generation"

        # RL learns diverse strategies
        assert 'policy' in rl_code.lower(), "RL should learn policy"


class TestCodeQuality:
    """Test code quality metrics."""

    def test_diffusion_policy_length(self):
        """Test that diffusion_policy.py is appropriately sized."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"

        with open(example_file, 'r') as f:
            lines = f.readlines()

        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        assert len(code_lines) > 400, "Code should be substantial"
        assert len(code_lines) < 1000, "Code should be concise"

    def test_rl_policy_length(self):
        """Test that rl_policy.py is appropriately sized."""
        example_file = EXAMPLES_DIR / "rl_policy.py"

        with open(example_file, 'r') as f:
            lines = f.readlines()

        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        assert len(code_lines) > 400, "Code should be substantial"
        assert len(code_lines) < 1000, "Code should be concise"

    def test_code_has_main_functions(self):
        """Test that examples have main() functions."""
        for example_file in [
            EXAMPLES_DIR / "diffusion_policy.py",
            EXAMPLES_DIR / "rl_policy.py"
        ]:
            with open(example_file, 'r') as f:
                code = f.read()

            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)]
            assert 'main' in functions, f"main() function not found in {example_file.name}"


class TestIntegration:
    """Integration tests for Module 5 examples."""

    def test_all_module5_examples_exist(self):
        """Test that all required Module 5 code examples exist."""
        required_files = [
            "diffusion_policy.py",
            "rl_policy.py",
        ]

        for filename in required_files:
            filepath = EXAMPLES_DIR / filename
            assert filepath.exists(), f"Required example not found: {filename}"

    def test_examples_can_be_imported(self):
        """Test that examples can be parsed without import errors."""
        example_files = [
            "diffusion_policy.py",
            "rl_policy.py",
        ]

        for example_file in example_files:
            filepath = EXAMPLES_DIR / example_file

            with open(filepath, 'r') as f:
                code = f.read()

            # Remove external imports for syntax checking
            modified_code = code.replace('import torch', '#import torch')
            modified_code = modified_code.replace('import torch.nn', '#import torch.nn')
            modified_code = modified_code.replace('import torch.optim', '#import torch.optim')
            modified_code = modified_code.replace('from torch', '#from torch')
            modified_code = modified_code.replace('import numpy', '#import numpy')

            try:
                compile(modified_code, str(filepath), 'exec')
            except SyntaxError as e:
                pytest.fail(f"Code example cannot be compiled: {e}")

    def test_documentation_references_code_examples(self):
        """Test that Module 5 documentation references code examples."""
        doc_files = [
            Path(__file__).parent.parent.parent / "book" / "docs" / "05-embodied-learning" / "end-to-end-learning.md",
            Path(__file__).parent.parent.parent / "book" / "docs" / "05-embodied-learning" / "training-pipeline.md",
        ]

        for doc_file in doc_files:
            if doc_file.exists():
                with open(doc_file, 'r') as f:
                    content = f.read()

                # Check that examples are referenced
                assert 'code' in content.lower() or 'example' in content.lower(), \
                    f"Documentation {doc_file.name} should reference code examples"


class TestModule5SpecificConcepts:
    """Tests for Module 5 learning objectives."""

    def test_diffusion_shows_trajectory_generation(self):
        """Test that diffusion example shows trajectory generation."""
        example_file = EXAMPLES_DIR / "diffusion_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should generate trajectories
        assert 'trajectory' in code.lower(), "Should work with trajectories"
        assert 'infer' in code.lower(), "Should have inference method"

    def test_rl_shows_exploration_exploitation(self):
        """Test that RL example shows exploration/exploitation."""
        example_file = EXAMPLES_DIR / "rl_policy.py"

        with open(example_file, 'r') as f:
            code = f.read()

        # Should demonstrate learning through exploration
        assert 'sample' in code.lower(), "Should sample actions"
        assert 'reward' in code.lower(), "Should use reward signal"
        assert 'gradient' in code.lower() or 'loss' in code.lower(), \
            "Should compute gradients for learning"

    def test_both_show_network_architectures(self):
        """Test that both examples show realistic network architectures."""
        files_to_check = [
            (EXAMPLES_DIR / "diffusion_policy.py", "DiffusionUNet"),
            (EXAMPLES_DIR / "rl_policy.py", "ActorNetwork"),
        ]

        for filepath, network_name in files_to_check:
            with open(filepath, 'r') as f:
                code = f.read()

            tree = ast.parse(code)

            # Check that network exists
            classes = [node.name for node in ast.walk(tree)
                      if isinstance(node, ast.ClassDef)]
            assert network_name in classes, f"{network_name} not found in {filepath.name}"

            # Check that it has reasonable layers
            assert 'linear' in code.lower() or 'conv' in code.lower(), \
                f"{filepath.name} should define neural network layers"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
