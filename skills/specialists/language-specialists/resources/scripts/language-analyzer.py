#!/usr/bin/env python3
"""
Language Analyzer - Analyze project language distribution and recommend tooling.

Scans directory for file types, calculates language distribution,
recommends tooling based on stack, and generates dependency graph.

Usage:
    python language-analyzer.py --path /path/to/project
    python language-analyzer.py --path . --report
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


class LanguageAnalyzer:
    """Analyze programming languages used in a project."""

    EXTENSIONS = {
        ".py": "Python",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".go": "Go",
        ".rs": "Rust",
        ".java": "Java",
        ".cpp": "C++",
        ".c": "C",
        ".cs": "C#",
        ".rb": "Ruby",
        ".php": "PHP",
    }

    IGNORE_DIRS = {
        "node_modules",
        ".venv",
        "venv",
        ".git",
        "__pycache__",
        "dist",
        "build",
        ".next",
        ".cache",
    }

    def __init__(self, path: Path):
        """Initialize analyzer with project path."""
        self.path = path
        self.file_counts: Dict[str, int] = defaultdict(int)
        self.line_counts: Dict[str, int] = defaultdict(int)
        self.total_files = 0
        self.total_lines = 0

    def should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        return any(ignore_dir in path.parts for ignore_dir in self.IGNORE_DIRS)

    def count_lines(self, file_path: Path) -> int:
        """Count lines in a file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def analyze(self) -> None:
        """Analyze all files in the project."""
        for file_path in self.path.rglob("*"):
            if file_path.is_file() and not self.should_ignore(file_path):
                ext = file_path.suffix.lower()
                if ext in self.EXTENSIONS:
                    language = self.EXTENSIONS[ext]
                    self.file_counts[language] += 1
                    lines = self.count_lines(file_path)
                    self.line_counts[language] += lines
                    self.total_files += 1
                    self.total_lines += lines

    def get_distribution(self) -> Dict[str, float]:
        """Get language distribution by lines of code."""
        if self.total_lines == 0:
            return {}
        return {
            lang: (lines / self.total_lines) * 100
            for lang, lines in self.line_counts.items()
        }

    def recommend_tooling(self) -> List[str]:
        """Recommend tooling based on language distribution."""
        recommendations = []
        distribution = self.get_distribution()

        # Python tooling
        if "Python" in distribution and distribution["Python"] > 10:
            recommendations.append("🐍 Python Tooling:")
            recommendations.append("  - black (formatter)")
            recommendations.append("  - ruff (linter)")
            recommendations.append("  - mypy (type checker)")
            recommendations.append("  - pytest (testing)")

        # TypeScript tooling
        if "TypeScript" in distribution and distribution["TypeScript"] > 10:
            recommendations.append("🔷 TypeScript Tooling:")
            recommendations.append("  - Prettier (formatter)")
            recommendations.append("  - ESLint (linter)")
            recommendations.append("  - tsc (type checker)")
            recommendations.append("  - Jest/Vitest (testing)")

        # Multi-language projects
        python_pct = distribution.get("Python", 0)
        ts_pct = distribution.get("TypeScript", 0)
        if python_pct > 10 and ts_pct > 10:
            recommendations.append("🔗 Multi-Language Recommendations:")
            recommendations.append("  - Docker Compose for service orchestration")
            recommendations.append("  - Turborepo/nx for monorepo management")
            recommendations.append("  - Shared API contracts (OpenAPI/gRPC)")

        return recommendations

    def generate_report(self, output_file: Path = None) -> None:
        """Generate analysis report."""
        self.analyze()

        report = {
            "summary": {
                "total_files": self.total_files,
                "total_lines": self.total_lines,
            },
            "file_counts": dict(self.file_counts),
            "line_counts": dict(self.line_counts),
            "distribution": self.get_distribution(),
        }

        print("=" * 60)
        print("📊 Language Analysis Report")
        print("=" * 60)
        print(f"\nProject Path: {self.path.absolute()}")
        print(f"Total Files: {self.total_files}")
        print(f"Total Lines: {self.total_lines:,}")

        print("\n📈 Language Distribution (by lines):")
        distribution = sorted(
            self.get_distribution().items(), key=lambda x: x[1], reverse=True
        )
        for language, percentage in distribution:
            lines = self.line_counts[language]
            files = self.file_counts[language]
            bar = "█" * int(percentage / 2)
            print(f"  {language:15} {percentage:5.1f}% {bar}")
            print(f"  {'':15} {lines:,} lines in {files} files")

        print("\n💡 Recommended Tooling:")
        for rec in self.recommend_tooling():
            print(f"  {rec}")

        print("\n" + "=" * 60)

        if output_file:
            output_file.write_text(json.dumps(report, indent=2))
            print(f"📄 Report saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze project language distribution")
    parser.add_argument("--path", type=str, default=".", help="Project path to analyze")
    parser.add_argument(
        "--report", action="store_true", help="Generate JSON report"
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"❌ Error: Path {path} does not exist")
        return 1

    analyzer = LanguageAnalyzer(path)
    output_file = path / "language-analysis.json" if args.report else None
    analyzer.generate_report(output_file)

    return 0


if __name__ == "__main__":
    exit(main())
