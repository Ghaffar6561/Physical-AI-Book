#!/usr/bin/env python3
"""
Validate diagrams in the Physical AI Book.

This script:
1. Finds all diagram files (SVG, PNG)
2. Verifies diagrams are referenced in markdown
3. Checks for broken image links
4. Validates SVG syntax (if applicable)
5. Reports missing or orphaned diagrams

Usage:
    python validate-diagrams.py
    python validate-diagrams.py --verbose
    python validate-diagrams.py --fix-links
"""

import sys
import re
from pathlib import Path
from typing import List, Set, Tuple
import argparse


class DiagramValidator:
    """Validate diagram files and references."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.repo_root = Path(__file__).parent.parent.parent
        self.diagrams_dir = self.repo_root / "book" / "static" / "diagrams"
        self.docs_dir = self.repo_root / "book" / "docs"
        self.results = {
            "total_diagrams": 0,
            "referenced_diagrams": 0,
            "orphaned_diagrams": [],
            "missing_references": [],
            "broken_links": [],
            "svg_errors": [],
        }

    def log(self, message: str, level="INFO"):
        """Log messages with optional verbosity."""
        if level == "INFO" or self.verbose:
            print(f"[{level}] {message}")

    def find_diagrams(self) -> Set[Path]:
        """Find all diagram files."""
        if not self.diagrams_dir.exists():
            self.log(f"Diagrams directory not found: {self.diagrams_dir}", "WARN")
            return set()

        # Find SVG and PNG files
        diagrams = set()
        diagrams.update(self.diagrams_dir.glob("**/*.svg"))
        diagrams.update(self.diagrams_dir.glob("**/*.png"))

        self.log(f"Found {len(diagrams)} diagram files")
        return diagrams

    def find_markdown_files(self) -> List[Path]:
        """Find all markdown files."""
        if not self.docs_dir.exists():
            self.log(f"Docs directory not found: {self.docs_dir}", "WARN")
            return []

        markdown_files = list(self.docs_dir.glob("**/*.md"))
        self.log(f"Found {len(markdown_files)} markdown files")
        return markdown_files

    def extract_image_references(self, markdown_content: str) -> Set[str]:
        """Extract image references from markdown content."""
        # Match ![alt](path) pattern
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(pattern, markdown_content)
        return {match[1] for match in matches}

    def find_referenced_diagrams(self) -> Tuple[Set[Path], Set[str]]:
        """Find all diagrams referenced in markdown files."""
        referenced_paths = set()
        broken_links = set()

        markdown_files = self.find_markdown_files()

        for md_file in markdown_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                references = self.extract_image_references(content)

                for ref in references:
                    # Resolve relative path
                    resolved_path = (md_file.parent / ref).resolve()

                    # Check if it's a diagram
                    if resolved_path.suffix in ['.svg', '.png']:
                        if resolved_path.exists():
                            referenced_paths.add(resolved_path)
                        else:
                            broken_links.add(f"{md_file.name}: {ref}")

            except Exception as e:
                self.log(f"Error reading {md_file.name}: {e}", "WARN")

        return referenced_paths, broken_links

    def validate_svg(self, file_path: Path) -> Tuple[bool, str]:
        """Validate SVG file syntax."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Basic SVG validation: check for SVG tags
            if '<svg' not in content or '</svg>' not in content:
                return False, "Missing SVG tags"

            # Check if it's valid XML (simplified)
            if content.count('<') != content.count('>'):
                return False, "Mismatched XML tags"

            return True, ""

        except Exception as e:
            return False, str(e)

    def validate(self) -> bool:
        """Run all validations."""
        all_diagrams = self.find_diagrams()
        self.results["total_diagrams"] = len(all_diagrams)

        if not all_diagrams:
            self.log("No diagrams found to validate", "WARN")
            return True

        print("\n" + "=" * 60)
        print("VALIDATING DIAGRAMS")
        print("=" * 60)

        # Find referenced diagrams
        referenced_diagrams, broken_links = self.find_referenced_diagrams()
        self.results["referenced_diagrams"] = len(referenced_diagrams)
        self.results["broken_links"] = list(broken_links)

        # Find orphaned diagrams
        orphaned = all_diagrams - referenced_diagrams
        self.results["orphaned_diagrams"] = [str(d.relative_to(self.repo_root))
                                             for d in orphaned]

        # Validate SVG files
        for diagram in all_diagrams:
            if diagram.suffix == '.svg':
                is_valid, error = self.validate_svg(diagram)
                if not is_valid:
                    self.results["svg_errors"].append(
                        f"{diagram.name}: {error}"
                    )

        # Print results
        print(f"\nTotal diagrams: {self.results['total_diagrams']}")
        print(f"Referenced diagrams: {self.results['referenced_diagrams']}")
        print(f"Orphaned diagrams: {len(self.results['orphaned_diagrams'])}")

        if self.results["broken_links"]:
            print(f"\n❌ Broken Links ({len(self.results['broken_links'])}):")
            for link in self.results["broken_links"]:
                print(f"  - {link}")

        if self.results["orphaned_diagrams"]:
            print(f"\n⚠️  Orphaned Diagrams ({len(self.results['orphaned_diagrams'])}):")
            for diagram in self.results["orphaned_diagrams"]:
                print(f"  - {diagram}")
            print("\n  Consider removing unused diagrams or adding references in markdown")

        if self.results["svg_errors"]:
            print(f"\n❌ SVG Errors ({len(self.results['svg_errors'])}):")
            for error in self.results["svg_errors"]:
                print(f"  - {error}")

        success = len(self.results["broken_links"]) == 0 and \
                  len(self.results["svg_errors"]) == 0

        if success:
            print("\n✅ ALL DIAGRAM VALIDATIONS PASSED")
        else:
            print("\n❌ DIAGRAM VALIDATION FAILED")

        print("=" * 60 + "\n")

        return success


def main():
    parser = argparse.ArgumentParser(description="Validate diagrams")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fix-links", action="store_true", help="Attempt to fix broken links")
    args = parser.parse_args()

    validator = DiagramValidator(verbose=args.verbose)
    success = validator.validate()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
