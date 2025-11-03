#!/usr/bin/env python3
"""
Style Audit Script for Code Review Assistant
Part of Gold tier enhancement - Style Reviewer Agent
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import re


@dataclass
class StyleIssue:
    """Code style issue"""
    severity: str  # high, medium, low, info
    category: str
    message: str
    file: str
    line: int
    suggestion: str = ""


class StyleAuditor:
    """Audits code for style, naming conventions, and best practices"""

    NAMING_PATTERNS = {
        'python': {
            'class': r'^[A-Z][a-zA-Z0-9]*$',  # PascalCase
            'function': r'^[a-z_][a-z0-9_]*$',  # snake_case
            'constant': r'^[A-Z_][A-Z0-9_]*$',  # UPPER_SNAKE
        },
        'javascript': {
            'class': r'^[A-Z][a-zA-Z0-9]*$',  # PascalCase
            'function': r'^[a-z][a-zA-Z0-9]*$',  # camelCase
            'constant': r'^[A-Z_][A-Z0-9_]*$',  # UPPER_SNAKE
        }
    }

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.issues: List[StyleIssue] = []

    def audit_all(self) -> Dict[str, Any]:
        """Run all style audits"""
        print("[Style Reviewer] Starting comprehensive style audit...")

        # Audit Python files
        for py_file in self.directory.rglob("*.py"):
            if "node_modules" not in str(py_file) and ".git" not in str(py_file):
                self.audit_python_file(py_file)

        # Audit JavaScript/TypeScript files
        for js_file in self.directory.rglob("*.js"):
            if "node_modules" not in str(js_file) and ".git" not in str(js_file):
                self.audit_js_file(js_file)

        for ts_file in self.directory.rglob("*.ts"):
            if "node_modules" not in str(ts_file) and ".git" not in str(ts_file):
                self.audit_js_file(ts_file)

        return self.generate_report()

    def audit_python_file(self, file_path: Path):
        """Audit Python file for style issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                # Check line length
                if len(line.rstrip()) > 88:  # Black's default
                    self.issues.append(StyleIssue(
                        severity="low",
                        category="line_length",
                        message=f"Line exceeds 88 characters ({len(line.rstrip())} chars)",
                        file=str(file_path),
                        line=i,
                        suggestion="Break into multiple lines or refactor"
                    ))

                # Check trailing whitespace
                if line.endswith(' \n') or line.endswith('\t\n'):
                    self.issues.append(StyleIssue(
                        severity="info",
                        category="whitespace",
                        message="Trailing whitespace",
                        file=str(file_path),
                        line=i,
                        suggestion="Remove trailing whitespace"
                    ))

                # Check naming conventions
                self._check_python_naming(line, file_path, i)

                # Check for common anti-patterns
                self._check_python_antipatterns(line, file_path, i)

                # Check documentation
                if line.strip().startswith('def ') or line.strip().startswith('class '):
                    # Check if next non-empty line is a docstring
                    if i < len(lines):
                        next_line = lines[i].strip() if i < len(lines) else ""
                        if not next_line.startswith('"""') and not next_line.startswith("'''"):
                            self.issues.append(StyleIssue(
                                severity="medium",
                                category="documentation",
                                message="Missing docstring",
                                file=str(file_path),
                                line=i,
                                suggestion="Add docstring describing purpose and parameters"
                            ))

        except Exception as e:
            print(f"Warning: Could not audit {file_path}: {e}")

    def audit_js_file(self, file_path: Path):
        """Audit JavaScript/TypeScript file for style issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                # Check line length
                if len(line.rstrip()) > 100:
                    self.issues.append(StyleIssue(
                        severity="low",
                        category="line_length",
                        message=f"Line exceeds 100 characters ({len(line.rstrip())} chars)",
                        file=str(file_path),
                        line=i,
                        suggestion="Break into multiple lines"
                    ))

                # Check for var usage (should use let/const)
                if re.search(r'\bvar\s+', line):
                    self.issues.append(StyleIssue(
                        severity="medium",
                        category="best_practices",
                        message="Using 'var' instead of 'let' or 'const'",
                        file=str(file_path),
                        line=i,
                        suggestion="Use 'const' for immutable variables, 'let' for mutable"
                    ))

                # Check for console.log in production code
                if 'console.log' in line and 'debug' not in file_path.name.lower():
                    self.issues.append(StyleIssue(
                        severity="low",
                        category="debugging",
                        message="console.log() statement in production code",
                        file=str(file_path),
                        line=i,
                        suggestion="Remove or replace with proper logging"
                    ))

                # Check naming conventions
                self._check_js_naming(line, file_path, i)

                # Check for common anti-patterns
                self._check_js_antipatterns(line, file_path, i)

        except Exception as e:
            print(f"Warning: Could not audit {file_path}: {e}")

    def _check_python_naming(self, line: str, file_path: Path, line_num: int):
        """Check Python naming conventions"""
        # Check class names
        class_match = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
        if class_match:
            class_name = class_match.group(1)
            if not re.match(self.NAMING_PATTERNS['python']['class'], class_name):
                self.issues.append(StyleIssue(
                    severity="medium",
                    category="naming",
                    message=f"Class name '{class_name}' should be PascalCase",
                    file=str(file_path),
                    line=line_num,
                    suggestion="Use PascalCase for class names"
                ))

        # Check function names
        func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
        if func_match:
            func_name = func_match.group(1)
            if not re.match(self.NAMING_PATTERNS['python']['function'], func_name):
                if not func_name.startswith('_'):  # Allow private methods
                    self.issues.append(StyleIssue(
                        severity="medium",
                        category="naming",
                        message=f"Function name '{func_name}' should be snake_case",
                        file=str(file_path),
                        line=line_num,
                        suggestion="Use snake_case for function names"
                    ))

    def _check_python_antipatterns(self, line: str, file_path: Path, line_num: int):
        """Check for Python anti-patterns"""
        # Check for mutable default arguments
        if re.search(r'def\s+\w+\([^)]*=\s*(\[\]|\{\})', line):
            self.issues.append(StyleIssue(
                severity="high",
                category="anti_pattern",
                message="Mutable default argument",
                file=str(file_path),
                line=line_num,
                suggestion="Use None and initialize inside function"
            ))

        # Check for bare except
        if line.strip() == 'except:':
            self.issues.append(StyleIssue(
                severity="medium",
                category="best_practices",
                message="Bare 'except:' clause",
                file=str(file_path),
                line=line_num,
                suggestion="Catch specific exceptions"
            ))

        # Check for comparison with True/False
        if re.search(r'==\s*(True|False)\b', line):
            self.issues.append(StyleIssue(
                severity="low",
                category="best_practices",
                message="Explicit comparison with True/False",
                file=str(file_path),
                line=line_num,
                suggestion="Use 'if variable:' or 'if not variable:'"
            ))

    def _check_js_naming(self, line: str, file_path: Path, line_num: int):
        """Check JavaScript naming conventions"""
        # Check class names
        class_match = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
        if class_match:
            class_name = class_match.group(1)
            if not re.match(self.NAMING_PATTERNS['javascript']['class'], class_name):
                self.issues.append(StyleIssue(
                    severity="medium",
                    category="naming",
                    message=f"Class name '{class_name}' should be PascalCase",
                    file=str(file_path),
                    line=line_num,
                    suggestion="Use PascalCase for class names"
                ))

        # Check function names
        func_match = re.search(r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
        if func_match:
            func_name = func_match.group(1)
            if not re.match(self.NAMING_PATTERNS['javascript']['function'], func_name):
                self.issues.append(StyleIssue(
                    severity="medium",
                    category="naming",
                    message=f"Function name '{func_name}' should be camelCase",
                    file=str(file_path),
                    line=line_num,
                    suggestion="Use camelCase for function names"
                ))

    def _check_js_antipatterns(self, line: str, file_path: Path, line_num: int):
        """Check for JavaScript anti-patterns"""
        # Check for == instead of ===
        if re.search(r'[^=!><]={2}[^=]', line) and '===' not in line:
            self.issues.append(StyleIssue(
                severity="medium",
                category="best_practices",
                message="Using '==' instead of '==='",
                file=str(file_path),
                line=line_num,
                suggestion="Use '===' for strict equality"
            ))

        # Check for callback hell (many nested callbacks)
        indent = len(line) - len(line.lstrip())
        if indent > 24 and ('function' in line or '=>' in line):
            self.issues.append(StyleIssue(
                severity="medium",
                category="code_complexity",
                message="Deeply nested callback",
                file=str(file_path),
                line=line_num,
                suggestion="Refactor using async/await or promises"
            ))

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive style audit report"""

        # Group issues by severity and category
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)

        for issue in self.issues:
            severity_counts[issue.severity] += 1
            category_counts[issue.category] += 1

        # Calculate quality score
        score = 100 - (
            severity_counts['high'] * 10 +
            severity_counts['medium'] * 5 +
            severity_counts['low'] * 2 +
            severity_counts['info'] * 1
        )
        score = max(0, min(100, score))

        # Generate top issues
        top_issues = sorted(self.issues, key=lambda x:
                          {'high': 0, 'medium': 1, 'low': 2, 'info': 3}[x.severity])[:15]

        report = {
            'agent': 'Style Reviewer',
            'quality_score': score,
            'summary': {
                'total_issues': len(self.issues),
                'high_severity': severity_counts['high'],
                'medium_severity': severity_counts['medium'],
                'low_severity': severity_counts['low'],
                'info': severity_counts['info'],
                'categories': dict(category_counts)
            },
            'issues': [asdict(issue) for issue in top_issues],
            'recommendations': self._generate_recommendations(category_counts)
        }

        return report

    def _generate_recommendations(self, category_counts: Dict[str, int]) -> List[str]:
        """Generate actionable style recommendations"""
        recommendations = []

        if category_counts['naming'] > 0:
            recommendations.append(
                "Enforce consistent naming conventions using linters"
            )

        if category_counts['documentation'] > 0:
            recommendations.append(
                "Add docstrings/JSDoc for public functions and classes"
            )

        if category_counts['best_practices'] > 0:
            recommendations.append(
                "Follow language-specific best practices and idioms"
            )

        if category_counts['line_length'] > 0:
            recommendations.append(
                "Configure auto-formatting with Black (Python) or Prettier (JS)"
            )

        recommendations.append(
            "Set up pre-commit hooks for automated style checking"
        )

        return recommendations


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: style_audit.py <directory> [output_file]")
        sys.exit(1)

    directory = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "style-review.json"

    auditor = StyleAuditor(directory)
    report = auditor.audit_all()

    # Print summary
    print("\n" + "="*60)
    print("Style Audit Complete")
    print("="*60)
    print(f"Quality Score: {report['quality_score']}/100")
    print(f"Total Issues: {report['summary']['total_issues']}")
    print(f"  High: {report['summary']['high_severity']}")
    print(f"  Medium: {report['summary']['medium_severity']}")
    print(f"  Low: {report['summary']['low_severity']}")
    print(f"  Info: {report['summary']['info']}")
    print("="*60)

    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
