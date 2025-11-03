#!/usr/bin/env python3
"""
Test suite for quality-reporter.py
Part of quick-quality-check Enhanced tier tests
"""

import unittest
import tempfile
import shutil
import json
import sys
from pathlib import Path

# Add resources directory to path
SCRIPT_DIR = Path(__file__).parent
RESOURCES_DIR = SCRIPT_DIR.parent / 'resources'
sys.path.insert(0, str(RESOURCES_DIR))

# Import the reporter
try:
    from quality_reporter import QualityReporter
    REPORTER_AVAILABLE = True
except ImportError:
    REPORTER_AVAILABLE = False
    print("[WARN] Quality reporter not available for import testing")


class TestQualityReporter(unittest.TestCase):
    """Test cases for quality reporter"""

    def setUp(self):
        """Create temporary test directory and sample data"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        # Create sample lint results
        self.lint_data = {
            'summary': {
                'total_issues': 10,
                'eslint_errors': 2,
                'eslint_warnings': 5,
                'prettier_violations': 3,
            }
        }

        # Create sample security results
        self.security_data = {
            'scan_summary': {
                'total_issues': 5,
                'critical': 1,
                'high': 2,
                'medium': 2,
                'low': 0,
            },
            'issues': {
                'critical': [{'description': 'Hardcoded password'}],
                'high': [
                    {'description': 'Unsafe eval()'},
                    {'description': 'SQL injection'},
                ],
                'medium': [],
                'low': [],
            }
        }

        # Create sample test results
        self.test_data = {
            'framework': 'jest',
            'tests_run': 50,
            'tests_passed': 48,
            'tests_failed': 2,
            'execution_time': 5.2,
        }

    def tearDown(self):
        """Clean up temporary test directory"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def write_json_file(self, filename: str, data: dict) -> Path:
        """Helper to write JSON file"""
        file_path = self.test_path / filename
        with open(file_path, 'w') as f:
            json.dump(data, f)
        return file_path

    def test_reporter_loads_lint_results(self):
        """Test loading lint results"""
        if not REPORTER_AVAILABLE:
            self.skipTest("Reporter not available")

        lint_file = self.write_json_file('lint.json', self.lint_data)

        reporter = QualityReporter()
        reporter.load_lint_results(str(lint_file))

        self.assertEqual(reporter.lint_data, self.lint_data)

    def test_reporter_loads_security_results(self):
        """Test loading security results"""
        if not REPORTER_AVAILABLE:
            self.skipTest("Reporter not available")

        security_file = self.write_json_file('security.json', self.security_data)

        reporter = QualityReporter()
        reporter.load_security_results(str(security_file))

        self.assertEqual(reporter.security_data, self.security_data)

    def test_reporter_calculates_quality_score(self):
        """Test quality score calculation"""
        if not REPORTER_AVAILABLE:
            self.skipTest("Reporter not available")

        reporter = QualityReporter()
        reporter.lint_data = self.lint_data
        reporter.security_data = self.security_data
        reporter.test_data = self.test_data

        score = reporter.calculate_quality_score()

        # Score should be 0-100
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

        # Score should be reduced due to critical security issue
        self.assertLess(score, 90)

    def test_reporter_categorizes_issues(self):
        """Test issue categorization"""
        if not REPORTER_AVAILABLE:
            self.skipTest("Reporter not available")

        reporter = QualityReporter()
        reporter.security_data = self.security_data
        reporter.test_data = self.test_data

        issues = reporter.categorize_issues()

        # Check structure
        self.assertIn('critical', issues)
        self.assertIn('high', issues)
        self.assertIn('medium', issues)
        self.assertIn('low', issues)

        # Should have critical security issue
        self.assertGreater(len(issues['critical']), 0)

        # Should have test failures in high
        test_failures = [i for i in issues['high'] if i.get('source') == 'tests']
        self.assertGreater(len(test_failures), 0)

    def test_reporter_generates_recommendations(self):
        """Test recommendation generation"""
        if not REPORTER_AVAILABLE:
            self.skipTest("Reporter not available")

        reporter = QualityReporter()
        reporter.security_data = self.security_data

        issues = reporter.categorize_issues()
        recommendations = reporter.generate_recommendations(issues)

        # Should have recommendations due to critical issues
        self.assertGreater(len(recommendations), 0)

        # Should mention critical issues
        critical_mentioned = any(
            'critical' in rec.lower() for rec in recommendations
        )
        self.assertTrue(critical_mentioned)

    def test_reporter_generates_complete_report(self):
        """Test complete report generation"""
        if not REPORTER_AVAILABLE:
            self.skipTest("Reporter not available")

        # Write all result files
        lint_file = self.write_json_file('lint.json', self.lint_data)
        security_file = self.write_json_file('security.json', self.security_data)
        test_file = self.write_json_file('tests.json', self.test_data)

        reporter = QualityReporter()
        reporter.load_lint_results(str(lint_file))
        reporter.load_security_results(str(security_file))
        reporter.load_test_results(str(test_file))

        report = reporter.generate_report()

        # Check required sections
        self.assertIn('timestamp', report)
        self.assertIn('quality_score', report)
        self.assertIn('status', report)
        self.assertIn('summary', report)
        self.assertIn('issues', report)
        self.assertIn('recommendations', report)
        self.assertIn('pass_criteria', report)
        self.assertIn('passed', report)

    def test_reporter_pass_fail_criteria(self):
        """Test pass/fail determination"""
        if not REPORTER_AVAILABLE:
            self.skipTest("Reporter not available")

        reporter = QualityReporter()

        # Test with critical issues (should fail)
        reporter.security_data = self.security_data  # Has 1 critical
        reporter.test_data = self.test_data
        report = reporter.generate_report()

        self.assertFalse(report['passed'], "Report should fail with critical issues")

        # Test with no critical issues
        reporter.security_data = {
            'scan_summary': {'total_issues': 0, 'critical': 0, 'high': 0},
            'issues': {'critical': [], 'high': [], 'medium': [], 'low': []},
        }
        reporter.test_data = {
            'tests_run': 10,
            'tests_passed': 10,
            'tests_failed': 0,
        }
        report = reporter.generate_report()

        # May still fail due to lint issues, but should not fail on critical/tests
        critical_issues = len(report['issues']['critical'])
        test_failures = report['summary']['tests']['failed']

        self.assertEqual(critical_issues, 0)
        self.assertEqual(test_failures, 0)

    def test_reporter_console_output_format(self):
        """Test console output formatting"""
        if not REPORTER_AVAILABLE:
            self.skipTest("Reporter not available")

        reporter = QualityReporter()
        reporter.security_data = self.security_data
        reporter.test_data = self.test_data

        report = reporter.generate_report()
        console_output = reporter.format_console_output(report)

        # Check for key sections
        self.assertIn('QUALITY CHECK REPORT', console_output)
        self.assertIn('Overall Quality Score', console_output)
        self.assertIn('Summary:', console_output)
        self.assertIn('Recommendations:', console_output)

        # Should indicate failure due to critical issues
        self.assertIn('FAILED', console_output)


class TestReporterEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_reporter_handles_missing_files(self):
        """Test reporter handles missing input files gracefully"""
        if not REPORTER_AVAILABLE:
            self.skipTest("Reporter not available")

        reporter = QualityReporter()

        # Should not raise exception
        reporter.load_lint_results('nonexistent.json')
        reporter.load_security_results('nonexistent.json')

        # Should still generate report
        report = reporter.generate_report()
        self.assertIsInstance(report, dict)

    def test_reporter_handles_empty_data(self):
        """Test reporter with no data"""
        if not REPORTER_AVAILABLE:
            self.skipTest("Reporter not available")

        reporter = QualityReporter()
        report = reporter.generate_report()

        # Should have default structure
        self.assertIn('quality_score', report)
        self.assertIn('passed', report)


def main():
    """Run test suite"""
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
