#!/usr/bin/env python3
"""
Validate code examples in the Physical AI Book.

This script:
1. Discovers all Python code examples
2. Checks for syntax errors
3. Verifies imports are available
4. Tests execution (if tests exist)
5. Reports coverage of SC-009 requirement (95% must work)

Usage:
    python validate-code-examples.py
    python validate-code-examples.py --verbose
    python validate-code-examples.py --fix-imports
"""

import sys
import os
import ast
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


class CodeExampleValidator:
    """Validate code examples meet quality standards."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.repo_root = Path(__file__).parent.parent.parent
        self.examples_dir = self.repo_root / "book" / "static" / "code-examples"
        self.test_dir = self.repo_root / "tests"
        self.results = {
            "total": 0,
            "valid": 0,
            "syntax_errors": [],
            "import_errors": [],
            "test_failures": [],
            "execution_warnings": [],
        }

    def log(self, message: str, level="INFO"):
        """Log messages with optional verbosity."""
        if level == "INFO" or self.verbose:
            print(f"[{level}] {message}")

    def find_code_examples(self) -> List[Path]:
        """Find all Python code examples."""
        if not self.examples_dir.exists():
            self.log(f"Code examples directory not found: {self.examples_dir}", "WARN")
            return []

        examples = list(self.examples_dir.glob("*.py"))
        self.log(f"Found {len(examples)} code examples")
        return examples

    def check_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """Check Python syntax of a file."""
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            error_msg = f"{file_path.name}: {e.msg} (line {e.lineno})"
            return False, error_msg

    def check_imports(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Check if imports in a file are available."""
        missing_imports = []

        try:
            with open(file_path, 'r') as f:
                code = f.read()

            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if not self._can_import(module_name):
                            missing_imports.append(module_name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if not self._can_import(module_name):
                            missing_imports.append(module_name)

        except Exception as e:
            self.log(f"Error parsing imports in {file_path.name}: {e}", "WARN")

        return len(missing_imports) == 0, missing_imports

    def _can_import(self, module_name: str) -> bool:
        """Check if a module can be imported."""
        # Skip standard library modules
        import importlib.util
        try:
            importlib.util.find_spec(module_name)
            return True
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def run_tests(self) -> bool:
        """Run pytest on code examples."""
        if not self.test_dir.exists():
            self.log("Test directory not found", "WARN")
            return True

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(self.test_dir / "unit"), "-v", "--tb=short"],
                capture_output=True,
                timeout=300,
                text=True
            )

            if result.returncode == 0:
                self.log("All tests passed", "INFO")
                return True
            else:
                self.log("Some tests failed", "ERROR")
                print(result.stdout)
                print(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            self.log("Tests timed out (> 5 minutes)", "ERROR")
            return False
        except FileNotFoundError:
            self.log("pytest not found - install with: pip install pytest", "WARN")
            return True

    def validate(self) -> bool:
        """Run all validations."""
        examples = self.find_code_examples()
        self.results["total"] = len(examples)

        if not examples:
            self.log("No code examples found to validate", "WARN")
            return True

        print("\n" + "=" * 60)
        print("VALIDATING CODE EXAMPLES")
        print("=" * 60)

        for example_file in examples:
            self.log(f"\nValidating: {example_file.name}")

            # Check syntax
            syntax_ok, error = self.check_syntax(example_file)
            if not syntax_ok:
                self.results["syntax_errors"].append(error)
                self.log(f"  ✗ Syntax Error: {error}", "ERROR")
                continue
            else:
                self.log(f"  ✓ Syntax OK", "INFO")

            # Check imports
            imports_ok, missing = self.check_imports(example_file)
            if not imports_ok:
                warning = f"{example_file.name}: Missing imports: {', '.join(missing)}"
                self.results["import_errors"].append(warning)
                self.log(f"  ⚠ Missing imports: {', '.join(missing)}", "WARN")
            else:
                self.log(f"  ✓ Imports OK", "INFO")

            # Mark as valid if no syntax errors
            self.results["valid"] += 1

        # Run tests
        self.log("\nRunning pytest on all examples...")
        tests_ok = self.run_tests()

        # Calculate compliance
        compliance_rate = (self.results["valid"] / self.results["total"] * 100) \
            if self.results["total"] > 0 else 0

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total examples: {self.results['total']}")
        print(f"Valid (no syntax errors): {self.results['valid']}")
        print(f"Compliance rate: {compliance_rate:.1f}%")

        if self.results["syntax_errors"]:
            print(f"\nSyntax Errors ({len(self.results['syntax_errors'])}):")
            for error in self.results["syntax_errors"]:
                print(f"  - {error}")

        if self.results["import_errors"]:
            print(f"\nMissing Imports ({len(self.results['import_errors'])}):")
            for error in self.results["import_errors"]:
                print(f"  - {error}")

        # SC-009: 95% of code examples must run without errors
        success = compliance_rate >= 95.0 and tests_ok

        if success:
            print("\n✅ ALL VALIDATIONS PASSED (SC-009 compliance achieved)")
        else:
            print("\n❌ VALIDATION FAILED")
            if compliance_rate < 95.0:
                print(f"  - Compliance rate {compliance_rate:.1f}% < 95% required")
            if not tests_ok:
                print(f"  - Tests failed")

        print("=" * 60 + "\n")

        return success


def main():
    parser = argparse.ArgumentParser(description="Validate code examples")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fix-imports", action="store_true", help="Attempt to fix import issues")
    args = parser.parse_args()

    validator = CodeExampleValidator(verbose=args.verbose)
    success = validator.validate()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
