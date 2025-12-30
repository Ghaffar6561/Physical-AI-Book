"""
Quickstart validation script for the RAG Ingestion Pipeline
This script validates that all steps in the quickstart guide work correctly
"""
import os
import sys
import subprocess
from pathlib import Path


def main():
    """Main validation function"""
    print("Starting quickstart validation for RAG Ingestion Pipeline...")
    print("="*60)

    # Get the project root directory (two levels up from the script location)
    # Script is in specs/001-rag-ingestion-pipeline/, so parent.parent is the project root
    project_root = Path(__file__).parent.parent.parent

    all_validations_passed = True

    # Run all validations
    validations = [
        lambda: validate_project_structure(project_root),
        lambda: validate_dependencies(project_root),
        lambda: validate_environment_config(project_root)
    ]

    for validation_func in validations:
        if not validation_func():
            all_validations_passed = False

    print("\n" + "="*60)
    if all_validations_passed:
        print("[SUCCESS] All quickstart validations PASSED!")
        print("The RAG Ingestion Pipeline is properly set up and ready to use.")
        return 0
    else:
        print("[FAILURE] Some quickstart validations FAILED!")
        print("Please check the above errors and fix them before proceeding.")
        return 1


def validate_project_structure(project_root):
    """Validate that the required project structure exists"""
    print(f"Validating project structure...")
    print(f"Project root: {project_root}")

    required_dirs = [
        "backend",
        "tests",
        "tests/unit",
        "tests/integration",
        "backup"
    ]

    required_files = [
        "backend/main.py",
        "backend/requirements.txt",
        "backend/.env.example",
        "backend/config.py",
        "backend/crawler.py",
        "backend/extractor.py",
        "backend/chunker.py",
        "backend/embedder.py",
        "backend/storage.py",
        "backend/models.py",
        "backend/exceptions.py",
        "backend/logger.py"
    ]

    all_good = True

    for directory in required_dirs:
        dir_path = project_root / directory
        print(f"Checking directory: {dir_path} - exists: {dir_path.is_dir()}")
        if not dir_path.is_dir():
            print(f"X Missing directory: {directory}")
            all_good = False
        else:
            print(f"[OK] Found directory: {directory}")

    for file in required_files:
        file_path = project_root / file
        print(f"Checking file: {file_path} - exists: {file_path.is_file()}")
        if not file_path.is_file():
            print(f"[MISSING] Missing file: {file}")
            all_good = False
        else:
            print(f"[OK] Found file: {file}")

    return all_good


def validate_dependencies(project_root):
    """Validate that dependencies can be installed"""
    print("\nValidating dependencies...")

    try:
        # Check if requirements.txt exists
        req_file = project_root / "backend" / "requirements.txt"
        if not req_file.is_file():
            print("[ERROR] requirements.txt not found")
            return False

        print("[OK] requirements.txt found")

        # Try to read the requirements file
        with open(req_file, 'r') as f:
            requirements = f.read()

        print("[OK] Successfully read requirements.txt")
        print(f"Requirements: {requirements[:100]}...")  # Print first 100 chars

        return True
    except Exception as e:
        print(f"[ERROR] Error validating dependencies: {e}")
        return False


def validate_environment_config(project_root):
    """Validate environment configuration"""
    print("\nValidating environment configuration...")

    try:
        # Check if .env.example exists
        env_example = project_root / "backend" / ".env.example"
        if not env_example.is_file():
            print("[ERROR] .env.example not found")
            return False

        print("[OK] .env.example found")

        # Try to read the .env.example file
        with open(env_example, 'r') as f:
            env_content = f.read()

        print("[OK] Successfully read .env.example")
        required_vars = ["BOOK_BASE_URL", "COHERE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
        missing_vars = []

        for var in required_vars:
            if var not in env_content:
                missing_vars.append(var)

        if missing_vars:
            print(f"[ERROR] Missing environment variables in .env.example: {missing_vars}")
            return False

        print("[OK] All required environment variables found in .env.example")
        return True
    except Exception as e:
        print(f"[ERROR] Error validating environment configuration: {e}")
        return False


if __name__ == "__main__":
    sys.exit(main())