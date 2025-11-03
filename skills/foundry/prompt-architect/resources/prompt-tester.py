#!/usr/bin/env python3
"""
Prompt Tester - Test prompt effectiveness through simulated execution.

This tool validates prompts by checking:
1. Clarity and parsability
2. Completeness of instructions
3. Handling of edge cases
4. Output format specification
5. Potential ambiguities

Usage:
    python prompt-tester.py <prompt_file>
    python prompt-tester.py --text "Your prompt" --test-cases cases.json
    python prompt-tester.py --interactive
"""

import re
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class Severity(Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class TestResult:
    """Result from a single test."""
    test_name: str
    passed: bool
    severity: Severity
    message: str
    details: Dict[str, Any] = None


@dataclass
class PromptTestReport:
    """Complete test report for a prompt."""
    prompt: str
    tests_run: int
    tests_passed: int
    tests_failed: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    results: List[TestResult]
    overall_score: float
    recommendation: str


class PromptTester:
    """Test prompts for effectiveness and completeness."""

    def __init__(self):
        self.results = []

    def test(self, prompt: str) -> PromptTestReport:
        """Run comprehensive tests on a prompt."""
        self.results = []

        # Run all test categories
        self._test_clarity(prompt)
        self._test_structure(prompt)
        self._test_completeness(prompt)
        self._test_edge_cases(prompt)
        self._test_output_spec(prompt)
        self._test_ambiguity(prompt)
        self._test_context(prompt)
        self._test_techniques(prompt)

        # Calculate metrics
        tests_run = len(self.results)
        tests_passed = sum(1 for r in self.results if r.passed)
        tests_failed = tests_run - tests_passed

        # Count by severity
        critical_issues = sum(1 for r in self.results if not r.passed and r.severity == Severity.CRITICAL)
        high_issues = sum(1 for r in self.results if not r.passed and r.severity == Severity.HIGH)
        medium_issues = sum(1 for r in self.results if not r.passed and r.severity == Severity.MEDIUM)
        low_issues = sum(1 for r in self.results if not r.passed and r.severity == Severity.LOW)

        # Calculate score (weighted by severity)
        max_score = tests_run * 10
        deductions = (
            critical_issues * 10 +
            high_issues * 7 +
            medium_issues * 4 +
            low_issues * 2
        )
        score = max(0, (max_score - deductions) / max_score * 100)

        # Generate recommendation
        recommendation = self._generate_recommendation(score, critical_issues, high_issues)

        return PromptTestReport(
            prompt=prompt,
            tests_run=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            low_issues=low_issues,
            results=self.results,
            overall_score=score,
            recommendation=recommendation,
        )

    def _test_clarity(self, prompt: str):
        """Test clarity of instructions."""
        # Test: Has clear opening statement
        first_sentence = prompt.split('.')[0] if '.' in prompt else prompt
        action_verbs = ['analyze', 'create', 'build', 'implement', 'design', 'evaluate', 'generate', 'write']

        has_action = any(verb in first_sentence.lower() for verb in action_verbs)
        self.results.append(TestResult(
            test_name="Clear Action Verb",
            passed=has_action,
            severity=Severity.HIGH,
            message="Prompt starts with clear action verb" if has_action else "Prompt lacks clear action verb at start"
        ))

        # Test: Defines success criteria
        has_criteria = bool(re.search(r'\b(should|must|will|ensure|verify|success)\b', prompt, re.IGNORECASE))
        self.results.append(TestResult(
            test_name="Success Criteria",
            passed=has_criteria,
            severity=Severity.MEDIUM,
            message="Success criteria defined" if has_criteria else "No explicit success criteria"
        ))

        # Test: Avoids ambiguous pronouns
        ambiguous = len(re.findall(r'\b(it|this|that|they)\b', prompt))
        passed = ambiguous < 5
        self.results.append(TestResult(
            test_name="Pronoun Clarity",
            passed=passed,
            severity=Severity.LOW,
            message=f"Appropriate pronoun usage ({ambiguous})" if passed else f"Too many ambiguous pronouns ({ambiguous})"
        ))

    def _test_structure(self, prompt: str):
        """Test structural organization."""
        word_count = len(prompt.split())

        # Test: Has structure for longer prompts
        if word_count > 200:
            has_structure = bool(re.search(r'(^|\n)#{1,6}\s+|\d+\.\s+|[-*]\s+', prompt))
            self.results.append(TestResult(
                test_name="Hierarchical Structure",
                passed=has_structure,
                severity=Severity.MEDIUM,
                message="Has hierarchical structure" if has_structure else "Long prompt lacks structure (headers/lists)"
            ))

        # Test: Has delimiters for data/code
        has_delimiters = bool(re.search(r'```|<[^>]+>|---', prompt))
        self.results.append(TestResult(
            test_name="Delimiters Present",
            passed=has_delimiters or word_count < 100,
            severity=Severity.LOW,
            message="Uses delimiters appropriately" if has_delimiters else "Consider adding delimiters for clarity"
        ))

        # Test: Logical flow
        has_flow = bool(re.search(r'\b(first|second|then|next|finally)\b', prompt, re.IGNORECASE))
        self.results.append(TestResult(
            test_name="Logical Flow",
            passed=has_flow or word_count < 150,
            severity=Severity.LOW,
            message="Has logical flow indicators" if has_flow else "Could benefit from flow indicators (first, then, finally)"
        ))

    def _test_completeness(self, prompt: str):
        """Test completeness of instructions."""
        # Test: Has context
        has_context = bool(re.search(r'\b(context|background|purpose)\b', prompt, re.IGNORECASE))
        self.results.append(TestResult(
            test_name="Context Provided",
            passed=has_context or len(prompt.split()) < 100,
            severity=Severity.MEDIUM,
            message="Context provided" if has_context else "Missing context/background section"
        ))

        # Test: Has constraints
        has_constraints = bool(re.search(r'\b(must|should|cannot|require)\b', prompt, re.IGNORECASE))
        self.results.append(TestResult(
            test_name="Constraints Defined",
            passed=has_constraints,
            severity=Severity.MEDIUM,
            message="Constraints defined" if has_constraints else "No explicit constraints (must/should/cannot)"
        ))

        # Test: Specifies input
        has_input = bool(re.search(r'\b(input|given|provided|data)\b', prompt, re.IGNORECASE))
        self.results.append(TestResult(
            test_name="Input Specification",
            passed=has_input,
            severity=Severity.LOW,
            message="Input specified" if has_input else "Input not explicitly specified"
        ))

    def _test_edge_cases(self, prompt: str):
        """Test edge case handling."""
        # Test: Mentions edge cases
        has_edge_cases = bool(re.search(r'\b(edge\s+case|boundary|exception|if.*then)\b', prompt, re.IGNORECASE))
        self.results.append(TestResult(
            test_name="Edge Case Handling",
            passed=has_edge_cases,
            severity=Severity.MEDIUM,
            message="Edge cases addressed" if has_edge_cases else "No explicit edge case handling"
        ))

        # Test: Error handling
        has_error_handling = bool(re.search(r'\b(error|fail|invalid|incorrect)\b', prompt, re.IGNORECASE))
        self.results.append(TestResult(
            test_name="Error Handling",
            passed=has_error_handling,
            severity=Severity.LOW,
            message="Error handling mentioned" if has_error_handling else "No error handling specified"
        ))

    def _test_output_spec(self, prompt: str):
        """Test output specification."""
        # Test: Specifies format
        has_format = bool(re.search(r'\b(format|json|structure|template)\b', prompt, re.IGNORECASE))
        self.results.append(TestResult(
            test_name="Output Format",
            passed=has_format,
            severity=Severity.HIGH,
            message="Output format specified" if has_format else "Output format not specified"
        ))

        # Test: Specifies length
        has_length = bool(re.search(r'\b(\d+\s+words?|\d+\s+sentences?|brief|detailed|comprehensive)\b', prompt, re.IGNORECASE))
        self.results.append(TestResult(
            test_name="Output Length",
            passed=has_length,
            severity=Severity.LOW,
            message="Output length specified" if has_length else "Output length not specified"
        ))

    def _test_ambiguity(self, prompt: str):
        """Test for ambiguities."""
        # Test: Avoids vague modifiers
        vague = re.findall(r'\b(quickly|simply|easily|obviously|clearly)\b', prompt, re.IGNORECASE)
        passed = len(vague) == 0
        self.results.append(TestResult(
            test_name="Vague Modifiers",
            passed=passed,
            severity=Severity.LOW,
            message="No vague modifiers" if passed else f"Contains vague modifiers: {', '.join(vague)}"
        ))

        # Test: Avoids contradictions
        contradictions = bool(re.search(r'(comprehensive|detailed).*(brief|concise|short)', prompt, re.IGNORECASE))
        self.results.append(TestResult(
            test_name="No Contradictions",
            passed=not contradictions,
            severity=Severity.MEDIUM,
            message="No contradictory requirements" if not contradictions else "Contains contradictory requirements"
        ))

    def _test_context(self, prompt: str):
        """Test context adequacy."""
        # Test: Defines audience
        has_audience = bool(re.search(r'\b(audience|reader|user|for\s+\w+)\b', prompt, re.IGNORECASE))
        self.results.append(TestResult(
            test_name="Audience Defined",
            passed=has_audience or len(prompt.split()) < 100,
            severity=Severity.LOW,
            message="Audience defined" if has_audience else "Audience not specified"
        ))

        # Test: States purpose
        has_purpose = bool(re.search(r'\b(purpose|goal|objective|aim)\b', prompt, re.IGNORECASE))
        self.results.append(TestResult(
            test_name="Purpose Stated",
            passed=has_purpose,
            severity=Severity.MEDIUM,
            message="Purpose stated" if has_purpose else "Purpose not explicitly stated"
        ))

    def _test_techniques(self, prompt: str):
        """Test for evidence-based techniques."""
        techniques = {
            'Chain-of-Thought': r'step\s+by\s+step|reasoning|think\s+through',
            'Self-Consistency': r'validate|verify|cross-check|alternative',
            'Few-Shot': r'example|demonstration|input:.*output:',
        }

        found = []
        for name, pattern in techniques.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                found.append(name)

        passed = len(found) > 0
        self.results.append(TestResult(
            test_name="Evidence-Based Techniques",
            passed=passed,
            severity=Severity.INFO,
            message=f"Uses techniques: {', '.join(found)}" if found else "No evidence-based techniques detected",
            details={'techniques': found}
        ))

    def _generate_recommendation(self, score: float, critical: int, high: int) -> str:
        """Generate overall recommendation."""
        if critical > 0:
            return "NOT READY - Critical issues must be addressed before use"
        elif high > 2:
            return "NEEDS IMPROVEMENT - Multiple high-priority issues detected"
        elif score < 60:
            return "FAIR - Consider refinement for better results"
        elif score < 80:
            return "GOOD - Ready for use with minor refinements"
        else:
            return "EXCELLENT - Well-structured and complete"


def print_report(report: PromptTestReport):
    """Print test report in readable format."""
    print(f"\n{'='*70}")
    print(f"PROMPT TEST REPORT")
    print(f"{'='*70}\n")

    # Summary
    print(f"Tests Run:     {report.tests_run}")
    print(f"Passed:        {report.tests_passed} ({report.tests_passed/report.tests_run*100:.1f}%)")
    print(f"Failed:        {report.tests_failed}")

    # Issues by severity
    print(f"\nIssues by Severity:")
    print(f"  Critical:    {report.critical_issues}")
    print(f"  High:        {report.high_issues}")
    print(f"  Medium:      {report.medium_issues}")
    print(f"  Low:         {report.low_issues}")

    # Overall score
    print(f"\nOverall Score: {report.overall_score:.1f}/100")
    print(f"Recommendation: {report.recommendation}")

    # Detailed results
    print(f"\n{'='*70}")
    print("DETAILED TEST RESULTS")
    print(f"{'='*70}\n")

    # Group by severity
    for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
        severity_results = [r for r in report.results if r.severity == severity]
        if not severity_results:
            continue

        severity_symbol = {
            Severity.CRITICAL: "🔴",
            Severity.HIGH: "🟠",
            Severity.MEDIUM: "🟡",
            Severity.LOW: "🔵",
            Severity.INFO: "ℹ️",
        }

        print(f"\n{severity_symbol[severity]} {severity.value.upper()} Priority:")
        for result in severity_results:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.test_name}: {result.message}")

    print(f"\n{'='*70}\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Test prompts for effectiveness')
    parser.add_argument('prompt', nargs='?', help='Prompt file to test')
    parser.add_argument('--text', help='Test text directly')
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    parser.add_argument('--output', help='Output file path')

    args = parser.parse_args()

    tester = PromptTester()

    # Get prompt
    if args.text:
        prompt = args.text
    elif args.prompt:
        with open(args.prompt, 'r', encoding='utf-8') as f:
            prompt = f.read()
    else:
        parser.print_help()
        sys.exit(1)

    # Run tests
    report = tester.test(prompt)

    # Output
    if args.json:
        output = {
            'tests_run': report.tests_run,
            'tests_passed': report.tests_passed,
            'tests_failed': report.tests_failed,
            'issues': {
                'critical': report.critical_issues,
                'high': report.high_issues,
                'medium': report.medium_issues,
                'low': report.low_issues,
            },
            'overall_score': report.overall_score,
            'recommendation': report.recommendation,
            'results': [
                {
                    'test': r.test_name,
                    'passed': r.passed,
                    'severity': r.severity.value,
                    'message': r.message,
                }
                for r in report.results
            ]
        }

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2)
        else:
            print(json.dumps(output, indent=2))
    else:
        print_report(report)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"Prompt Test Report\n")
                f.write(f"Overall Score: {report.overall_score:.1f}/100\n")
                f.write(f"Recommendation: {report.recommendation}\n")


if __name__ == '__main__':
    main()
