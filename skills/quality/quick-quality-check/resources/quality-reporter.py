#!/usr/bin/env python3
"""
Quality Reporter - Aggregate and format quality check results
Part of quick-quality-check Enhanced tier resources
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class QualityReporter:
    """Aggregate and format quality check results from multiple sources"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.lint_data = {}
        self.security_data = {}
        self.test_data = {}
        self.theater_data = {}

    def load_lint_results(self, file_path: str) -> None:
        """Load linting results"""
        try:
            with open(file_path, 'r') as f:
                self.lint_data = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load lint results: {e}", file=sys.stderr)
            self.lint_data = {}

    def load_security_results(self, file_path: str) -> None:
        """Load security scan results"""
        try:
            with open(file_path, 'r') as f:
                self.security_data = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load security results: {e}", file=sys.stderr)
            self.security_data = {}

    def load_test_results(self, file_path: str) -> None:
        """Load test execution results"""
        try:
            with open(file_path, 'r') as f:
                self.test_data = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load test results: {e}", file=sys.stderr)
            self.test_data = {}

    def load_theater_results(self, file_path: str) -> None:
        """Load theater detection results"""
        try:
            with open(file_path, 'r') as f:
                self.theater_data = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load theater results: {e}", file=sys.stderr)
            self.theater_data = {}

    def calculate_quality_score(self) -> int:
        """Calculate overall quality score (0-100)"""
        scores = []

        # Linting score (25 points max)
        if self.lint_data:
            summary = self.lint_data.get('summary', {})
            total_issues = summary.get('total_issues', 0)
            lint_score = max(0, 25 - (total_issues * 2))
            scores.append(lint_score)

        # Security score (30 points max)
        if self.security_data:
            scan_summary = self.security_data.get('scan_summary', {})
            critical = scan_summary.get('critical', 0)
            high = scan_summary.get('high', 0)

            security_score = 30 - (critical * 10) - (high * 5)
            security_score = max(0, security_score)
            scores.append(security_score)

        # Test score (25 points max)
        if self.test_data:
            total = self.test_data.get('tests_run', 0)
            passed = self.test_data.get('tests_passed', 0)

            if total > 0:
                test_score = int((passed / total) * 25)
            else:
                test_score = 0
            scores.append(test_score)

        # Theater score (20 points max)
        if self.theater_data:
            theater_issues = self.theater_data.get('total_patterns', 0)
            theater_score = max(0, 20 - (theater_issues * 5))
            scores.append(theater_score)

        # Calculate weighted average
        if scores:
            return int(sum(scores))
        else:
            return 0

    def categorize_issues(self) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize all issues by severity"""
        issues = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
        }

        # Add security issues
        if self.security_data:
            sec_issues = self.security_data.get('issues', {})
            for severity in issues.keys():
                issues[severity].extend([
                    {**issue, 'source': 'security'}
                    for issue in sec_issues.get(severity, [])
                ])

        # Add lint issues
        if self.lint_data:
            summary = self.lint_data.get('summary', {})
            eslint_errors = summary.get('eslint_errors', 0)

            if eslint_errors > 0:
                issues['high'].append({
                    'source': 'lint',
                    'description': f'{eslint_errors} ESLint errors found',
                })

        # Add test failures
        if self.test_data:
            failed = self.test_data.get('tests_failed', 0)
            if failed > 0:
                issues['high'].append({
                    'source': 'tests',
                    'description': f'{failed} tests failing',
                    'failures': self.test_data.get('failures', []),
                })

        # Add theater issues
        if self.theater_data:
            patterns = self.theater_data.get('patterns_found', [])
            for pattern in patterns:
                issues['medium'].append({
                    'source': 'theater',
                    'description': f"Theater pattern: {pattern.get('type')}",
                    'file': pattern.get('file'),
                })

        return issues

    def generate_recommendations(self, issues: Dict[str, List]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Critical recommendations
        if issues['critical']:
            recommendations.append("🚨 CRITICAL: Address all critical security issues immediately")
            recommendations.append("   - Review hardcoded secrets and move to environment variables")
            recommendations.append("   - Fix SQL injection vulnerabilities")

        # High priority recommendations
        if issues['high']:
            recommendations.append("⚠️  HIGH: Fix high-severity issues before deployment")

            has_lint_errors = any(i['source'] == 'lint' for i in issues['high'])
            if has_lint_errors:
                recommendations.append("   - Fix all ESLint errors")

            has_test_failures = any(i['source'] == 'tests' for i in issues['high'])
            if has_test_failures:
                recommendations.append("   - Investigate and fix failing tests")

        # Medium priority recommendations
        if issues['medium']:
            recommendations.append("ℹ️  MEDIUM: Address medium-priority issues in next sprint")

            has_theater = any(i['source'] == 'theater' for i in issues['medium'])
            if has_theater:
                recommendations.append("   - Replace theater patterns with real implementations")

        # General recommendations
        if self.test_data:
            passed = self.test_data.get('tests_passed', 0)
            total = self.test_data.get('tests_run', 0)

            if total > 0 and (passed / total) < 0.9:
                recommendations.append("📊 Improve test coverage to at least 90%")

        return recommendations

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        quality_score = self.calculate_quality_score()
        issues = self.categorize_issues()
        recommendations = self.generate_recommendations(issues)

        # Determine overall status
        if quality_score >= 90:
            status = "EXCELLENT"
        elif quality_score >= 75:
            status = "GOOD"
        elif quality_score >= 60:
            status = "FAIR"
        elif quality_score >= 40:
            status = "POOR"
        else:
            status = "CRITICAL"

        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'quality_score': quality_score,
            'status': status,
            'summary': {
                'lint': self.lint_data.get('summary', {}),
                'security': self.security_data.get('scan_summary', {}),
                'tests': {
                    'framework': self.test_data.get('framework'),
                    'total': self.test_data.get('tests_run', 0),
                    'passed': self.test_data.get('tests_passed', 0),
                    'failed': self.test_data.get('tests_failed', 0),
                    'execution_time': self.test_data.get('execution_time', 0),
                },
                'theater': {
                    'patterns_found': len(self.theater_data.get('patterns_found', [])),
                },
            },
            'issues': issues,
            'recommendations': recommendations,
            'pass_criteria': {
                'quality_score_min': 75,
                'critical_issues_max': 0,
                'high_issues_max': 5,
                'test_failures_max': 0,
            },
            'passed': (
                quality_score >= 75 and
                len(issues['critical']) == 0 and
                len(issues['high']) <= 5 and
                self.test_data.get('tests_failed', 0) == 0
            ),
        }

        return report

    def format_console_output(self, report: Dict[str, Any]) -> str:
        """Format report for console display"""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("QUICK QUALITY CHECK REPORT".center(80))
        lines.append("=" * 80)
        lines.append("")

        # Overall score
        score = report['quality_score']
        status = report['status']
        lines.append(f"Overall Quality Score: {score}/100 ({status})")
        lines.append("")

        # Summary
        lines.append("Summary:")
        lines.append(f"  Lint Issues: {report['summary']['lint'].get('total_issues', 0)}")
        lines.append(f"  Security Issues: {report['summary']['security'].get('total_issues', 0)}")
        lines.append(f"    - Critical: {report['summary']['security'].get('critical', 0)}")
        lines.append(f"    - High: {report['summary']['security'].get('high', 0)}")
        lines.append(f"  Tests: {report['summary']['tests']['passed']}/{report['summary']['tests']['total']} passed")
        lines.append(f"  Theater Patterns: {report['summary']['theater']['patterns_found']}")
        lines.append("")

        # Recommendations
        if report['recommendations']:
            lines.append("Recommendations:")
            for rec in report['recommendations']:
                lines.append(f"  {rec}")
            lines.append("")

        # Pass/Fail
        lines.append("=" * 80)
        if report['passed']:
            lines.append("✅ QUALITY CHECK PASSED".center(80))
        else:
            lines.append("❌ QUALITY CHECK FAILED".center(80))
        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Quality check reporter')
    parser.add_argument('--lint', help='Lint results JSON file')
    parser.add_argument('--security', help='Security scan results JSON file')
    parser.add_argument('--tests', help='Test results JSON file')
    parser.add_argument('--theater', help='Theater detection results JSON file')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('--format', choices=['json', 'console'], default='json', help='Output format')

    args = parser.parse_args()

    # Create reporter
    reporter = QualityReporter()

    # Load results
    if args.lint:
        reporter.load_lint_results(args.lint)
    if args.security:
        reporter.load_security_results(args.security)
    if args.tests:
        reporter.load_test_results(args.tests)
    if args.theater:
        reporter.load_theater_results(args.theater)

    # Generate report
    report = reporter.generate_report()

    # Format output
    if args.format == 'console':
        output = reporter.format_console_output(report)
    else:
        output = json.dumps(report, indent=2)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"[INFO] Report written to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Exit code based on pass/fail
    sys.exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
