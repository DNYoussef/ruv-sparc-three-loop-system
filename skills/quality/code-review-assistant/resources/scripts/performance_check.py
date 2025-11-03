#!/usr/bin/env python3
"""
Performance Analysis Script for Code Review Assistant
Part of Gold tier enhancement - Performance Analyst Agent
"""

import ast
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class PerformanceIssue:
    """Performance issue finding"""
    severity: str  # high, medium, low
    category: str
    message: str
    file: str
    line: int
    complexity: Optional[int] = None
    suggestion: str = ""


class PerformanceAnalyzer:
    """Analyzes code for performance bottlenecks"""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.issues: List[PerformanceIssue] = []

    def analyze_all(self) -> Dict[str, Any]:
        """Run all performance checks"""
        print("[Performance Analyst] Starting comprehensive analysis...")

        # Analyze Python files
        for py_file in self.directory.rglob("*.py"):
            if "node_modules" not in str(py_file) and ".git" not in str(py_file):
                self.analyze_python_file(py_file)

        # Analyze JavaScript/TypeScript files
        for js_file in self.directory.rglob("*.js"):
            if "node_modules" not in str(js_file) and ".git" not in str(js_file):
                self.analyze_js_file(js_file)

        for ts_file in self.directory.rglob("*.ts"):
            if "node_modules" not in str(ts_file) and ".git" not in str(ts_file):
                self.analyze_js_file(ts_file)

        return self.generate_report()

    def analyze_python_file(self, file_path: Path):
        """Analyze Python file for performance issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)

            # Check for inefficient patterns
            self._check_nested_loops(tree, file_path)
            self._check_list_comprehension_complexity(tree, file_path)
            self._check_repeated_operations(tree, file_path)
            self._check_global_variables(tree, file_path)

        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")

    def analyze_js_file(self, file_path: Path):
        """Analyze JavaScript/TypeScript file for performance issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple pattern matching (more sophisticated with proper JS parser)
            self._check_js_nested_loops(content, file_path)
            self._check_js_dom_manipulation(content, file_path)
            self._check_js_memory_leaks(content, file_path)

        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")

    def _check_nested_loops(self, tree: ast.AST, file_path: Path):
        """Detect nested loops that may cause O(n²) or worse complexity"""
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check if there's a nested loop
                for child in ast.walk(node):
                    if child != node and isinstance(child, ast.For):
                        self.issues.append(PerformanceIssue(
                            severity="medium",
                            category="algorithmic_complexity",
                            message="Nested loop detected - potential O(n²) complexity",
                            file=str(file_path),
                            line=node.lineno,
                            complexity=2,
                            suggestion="Consider using hash maps, sets, or more efficient algorithms"
                        ))
                        break

    def _check_list_comprehension_complexity(self, tree: ast.AST, file_path: Path):
        """Check for overly complex list comprehensions"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ListComp):
                # Count number of generators (nested comprehensions)
                num_generators = len(node.generators)
                if num_generators > 2:
                    self.issues.append(PerformanceIssue(
                        severity="low",
                        category="code_complexity",
                        message=f"Complex list comprehension with {num_generators} levels",
                        file=str(file_path),
                        line=node.lineno,
                        suggestion="Consider breaking into multiple steps for readability and performance"
                    ))

    def _check_repeated_operations(self, tree: ast.AST, file_path: Path):
        """Check for operations repeated in loops"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Check for function calls or attribute access in loop
                for child in ast.walk(node.body[0] if node.body else node):
                    if isinstance(child, ast.Call):
                        # Check if calling len() in loop condition
                        if isinstance(child.func, ast.Name) and child.func.id == 'len':
                            self.issues.append(PerformanceIssue(
                                severity="low",
                                category="repeated_computation",
                                message="len() called repeatedly in loop",
                                file=str(file_path),
                                line=node.lineno,
                                suggestion="Cache len() result before loop"
                            ))

    def _check_global_variables(self, tree: ast.AST, file_path: Path):
        """Check for excessive use of global variables (slower access)"""
        global_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                global_count += len(node.names)

        if global_count > 5:
            self.issues.append(PerformanceIssue(
                severity="low",
                category="performance",
                message=f"Excessive global variables ({global_count})",
                file=str(file_path),
                line=1,
                suggestion="Consider using class attributes or local variables"
            ))

    def _check_js_nested_loops(self, content: str, file_path: Path):
        """Check for nested loops in JavaScript"""
        lines = content.split('\n')
        loop_stack = []

        for i, line in enumerate(lines, 1):
            # Simple pattern matching
            if any(keyword in line for keyword in ['for (', 'for(', 'while (', 'while(']):
                loop_stack.append(i)
            elif '}' in line and loop_stack:
                loop_stack.pop()

            # If we have 2+ nested loops
            if len(loop_stack) >= 2:
                self.issues.append(PerformanceIssue(
                    severity="medium",
                    category="algorithmic_complexity",
                    message="Nested loop detected",
                    file=str(file_path),
                    line=i,
                    suggestion="Consider using Map/Set or alternative algorithms"
                ))
                loop_stack = []  # Reset to avoid duplicates

    def _check_js_dom_manipulation(self, content: str, file_path: Path):
        """Check for inefficient DOM manipulation"""
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for DOM manipulation in loops
            if 'for' in line or 'forEach' in line:
                # Look ahead for DOM operations
                for j in range(i, min(i + 10, len(lines))):
                    if any(dom_op in lines[j] for dom_op in [
                        'appendChild', 'innerHTML +=', 'document.createElement'
                    ]):
                        self.issues.append(PerformanceIssue(
                            severity="medium",
                            category="dom_performance",
                            message="DOM manipulation inside loop",
                            file=str(file_path),
                            line=i,
                            suggestion="Use DocumentFragment or batch DOM updates"
                        ))
                        break

    def _check_js_memory_leaks(self, content: str, file_path: Path):
        """Check for potential memory leaks"""
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for event listeners without removal
            if 'addEventListener' in line:
                # Simple heuristic: check if there's no corresponding removeEventListener
                if 'removeEventListener' not in content[content.find(line):]:
                    self.issues.append(PerformanceIssue(
                        severity="low",
                        category="memory_leak",
                        message="Event listener without removal",
                        file=str(file_path),
                        line=i,
                        suggestion="Add removeEventListener in cleanup/unmount"
                    ))

            # Check for unclosed intervals/timeouts
            if 'setInterval' in line and 'clearInterval' not in content:
                self.issues.append(PerformanceIssue(
                    severity="medium",
                    category="memory_leak",
                    message="setInterval without clearInterval",
                    file=str(file_path),
                    line=i,
                    suggestion="Store interval ID and clear in cleanup"
                ))

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""

        # Group issues by severity
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)

        for issue in self.issues:
            severity_counts[issue.severity] += 1
            category_counts[issue.category] += 1

        # Calculate performance score
        score = 100 - (
            severity_counts['high'] * 15 +
            severity_counts['medium'] * 10 +
            severity_counts['low'] * 5
        )
        score = max(0, min(100, score))

        # Generate bottleneck summary
        bottlenecks = []
        for issue in sorted(self.issues, key=lambda x:
                          {'high': 0, 'medium': 1, 'low': 2}[x.severity]):
            bottlenecks.append({
                'severity': issue.severity,
                'category': issue.category,
                'message': issue.message,
                'file': issue.file,
                'line': issue.line,
                'suggestion': issue.suggestion
            })

        report = {
            'agent': 'Performance Analyst',
            'score': score,
            'summary': {
                'total_issues': len(self.issues),
                'high_severity': severity_counts['high'],
                'medium_severity': severity_counts['medium'],
                'low_severity': severity_counts['low'],
                'categories': dict(category_counts)
            },
            'bottlenecks': bottlenecks[:10],  # Top 10 issues
            'recommendations': self._generate_recommendations(category_counts)
        }

        return report

    def _generate_recommendations(self, category_counts: Dict[str, int]) -> List[str]:
        """Generate actionable performance recommendations"""
        recommendations = []

        if category_counts['algorithmic_complexity'] > 0:
            recommendations.append(
                "Optimize nested loops using hash-based data structures (Maps/Sets)"
            )

        if category_counts['dom_performance'] > 0:
            recommendations.append(
                "Batch DOM updates using DocumentFragment or virtual DOM"
            )

        if category_counts['memory_leak'] > 0:
            recommendations.append(
                "Implement proper cleanup for event listeners and intervals"
            )

        if category_counts['repeated_computation'] > 0:
            recommendations.append(
                "Cache frequently computed values outside loops"
            )

        recommendations.append(
            "Profile code with Chrome DevTools or Python cProfile for precise bottlenecks"
        )

        return recommendations


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: performance_check.py <directory> [output_file]")
        sys.exit(1)

    directory = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "performance-review.json"

    analyzer = PerformanceAnalyzer(directory)
    report = analyzer.analyze_all()

    # Print summary
    print("\n" + "="*60)
    print("Performance Analysis Complete")
    print("="*60)
    print(f"Performance Score: {report['score']}/100")
    print(f"Total Issues: {report['summary']['total_issues']}")
    print(f"  High: {report['summary']['high_severity']}")
    print(f"  Medium: {report['summary']['medium_severity']}")
    print(f"  Low: {report['summary']['low_severity']}")
    print("="*60)

    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
