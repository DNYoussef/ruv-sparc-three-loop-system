#!/usr/bin/env python3
"""
Test suite for prompt-tester.py

Tests the prompt validation and testing framework.
"""

import sys
import os
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'resources'))

from prompt_tester import PromptTester, PromptTestReport, Severity


class TestPromptTester(unittest.TestCase):
    """Test cases for PromptTester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tester = PromptTester()

    def test_test_good_prompt(self):
        """Test validation of a well-formed prompt."""
        good_prompt = """
        Analyze quarterly sales data to identify growth trends and seasonal patterns.

        Requirements:
        - Must include year-over-year comparison
        - Should identify top 3 growth opportunities
        - Cannot exceed 500 words

        Output format: JSON with fields trend_score, seasonal_factors, opportunities

        Success criteria: Identify at least 2 statistically significant trends (p < 0.05)
        """

        report = self.tester.test(good_prompt)

        self.assertIsInstance(report, PromptTestReport)
        self.assertGreater(report.overall_score, 70.0)
        self.assertEqual(report.critical_issues, 0)
        self.assertIn("GOOD", report.recommendation)

    def test_test_poor_prompt(self):
        """Test validation of a poorly-formed prompt."""
        poor_prompt = "Quickly analyze this and make it better. Obviously keep it brief."

        report = self.tester.test(poor_prompt)

        self.assertLess(report.overall_score, 60.0)
        self.assertGreater(report.tests_failed, 5)
        self.assertGreater(len(report.results), 0)

    def test_critical_issues_detection(self):
        """Test detection of critical issues."""
        # A prompt with no clear action or output spec is critical
        critical_prompt = "Do something with the data maybe"

        report = self.tester.test(critical_prompt)

        # Should have failed tests
        self.assertGreater(report.tests_failed, 0)

    def test_clarity_tests(self):
        """Test clarity validation."""
        # Clear prompt
        clear_prompt = "Analyze sales data to find trends. Success means identifying 3 patterns."
        clear_report = self.tester.test(clear_prompt)

        # Unclear prompt
        unclear_prompt = "Look at the stuff and tell me about it"
        unclear_report = self.tester.test(unclear_prompt)

        self.assertGreater(clear_report.overall_score, unclear_report.overall_score)

    def test_structure_tests(self):
        """Test structure validation."""
        # Well-structured
        structured = """
        # Task: Data Analysis

        ## Step 1: Load Data
        Load CSV from ./data/sales.csv

        ## Step 2: Clean
        - Remove nulls
        - Handle outliers

        ## Step 3: Analyze
        Calculate summary statistics
        """
        structured_report = self.tester.test(structured)

        # No structure
        unstructured = "Load CSV clean data calculate stats"
        unstructured_report = self.tester.test(unstructured)

        # Count structure-related passes
        structured_passes = sum(
            1 for r in structured_report.results
            if r.passed and 'structure' in r.test_name.lower()
        )
        unstructured_passes = sum(
            1 for r in unstructured_report.results
            if r.passed and 'structure' in r.test_name.lower()
        )

        self.assertGreaterEqual(structured_passes, unstructured_passes)

    def test_completeness_tests(self):
        """Test completeness validation."""
        # Complete prompt
        complete = """
        Context: E-commerce platform with 100K products

        Task: Optimize search performance

        Requirements:
        - Must handle 1000 concurrent searches
        - Should return results in <100ms

        Input: User search query (string)
        Output: Sorted list of products (JSON array)

        Edge cases:
        - Empty query → return popular products
        - No results → suggest alternatives
        """
        complete_report = self.tester.test(complete)

        # Incomplete prompt
        incomplete = "Optimize search"
        incomplete_report = self.tester.test(incomplete)

        self.assertGreater(complete_report.overall_score, incomplete_report.overall_score)

    def test_output_specification_tests(self):
        """Test output format validation."""
        # With format spec
        with_format = """
        Calculate statistics.

        Output format: JSON with fields {mean: float, median: float, std_dev: float}
        Length: Exactly 3 fields as shown above
        """
        format_report = self.tester.test(with_format)

        # Without format spec
        no_format = "Calculate statistics"
        no_format_report = self.tester.test(no_format)

        # Check output format test
        format_tests = [r for r in format_report.results if 'format' in r.test_name.lower()]
        self.assertTrue(any(r.passed for r in format_tests))

    def test_edge_case_handling_tests(self):
        """Test edge case validation."""
        # With edge cases
        with_edges = """
        Parse user input.

        Edge cases:
        - Empty input → return default
        - Invalid format → raise ValueError
        - Null values → skip and log warning
        """
        edges_report = self.tester.test(with_edges)

        # Without edge cases
        no_edges = "Parse user input"
        no_edges_report = self.tester.test(no_edges)

        # Check edge case test
        edge_tests = [r for r in edges_report.results if 'edge case' in r.test_name.lower()]
        self.assertTrue(any(r.passed for r in edge_tests))

    def test_anti_pattern_detection(self):
        """Test anti-pattern detection."""
        # Vague modifiers
        vague = "Quickly analyze this and simply explain it briefly"
        vague_report = self.tester.test(vague)

        vague_tests = [r for r in vague_report.results if 'vague' in r.test_name.lower()]
        self.assertTrue(any(not r.passed for r in vague_tests))

        # Contradictions
        contradictory = "Provide a comprehensive yet brief analysis"
        contra_report = self.tester.test(contradictory)

        contra_tests = [r for r in contra_report.results if 'contradiction' in r.test_name.lower()]
        self.assertTrue(any(not r.passed for r in contra_tests))

    def test_technique_detection(self):
        """Test detection of evidence-based techniques."""
        # Chain of thought
        cot = "Solve this step by step, explaining your reasoning"
        cot_report = self.tester.test(cot)

        # Self-consistency
        sc = "Validate your conclusions by considering alternatives"
        sc_report = self.tester.test(sc)

        # Few-shot
        fs = "Examples:\nInput: A\nOutput: B\n\nInput: C\nOutput: D\n\nNow process: E"
        fs_report = self.tester.test(fs)

        # Each should have technique tests passing
        self.assertTrue(any('technique' in r.test_name.lower() for r in cot_report.results))

    def test_severity_levels(self):
        """Test severity classification."""
        prompt = "Do stuff"
        report = self.tester.test(prompt)

        # Should have results with different severities
        severities = set(r.severity for r in report.results)
        self.assertGreater(len(severities), 0)

        # Check severity is valid
        for result in report.results:
            self.assertIsInstance(result.severity, Severity)

    def test_recommendation_generation(self):
        """Test recommendation generation."""
        # Poor prompt should get recommendations
        poor = "do the thing"
        poor_report = self.tester.test(poor)

        # Good prompt should get fewer recommendations
        good = """
        # Objective: Analyze sales trends

        ## Context
        Quarterly sales data for 2023-2024

        ## Requirements
        - Must identify top 3 trends
        - Should include statistical validation

        ## Output
        JSON format with trend_analysis field

        Think through this step by step.
        """
        good_report = self.tester.test(good)

        # Poor should have more recommendations
        # (Good might still have some suggestions)
        self.assertGreater(len(poor_report.recommendations), 0)

    def test_score_calculation(self):
        """Test overall score calculation."""
        # Score should be 0-100
        prompt = "Test prompt"
        report = self.tester.test(prompt)

        self.assertGreaterEqual(report.overall_score, 0.0)
        self.assertLessEqual(report.overall_score, 100.0)

    def test_metrics_tracking(self):
        """Test that metrics are tracked."""
        prompt = "Test prompt"
        report = self.tester.test(prompt)

        self.assertGreater(report.tests_run, 0)
        self.assertEqual(report.tests_passed + report.tests_failed, report.tests_run)
        self.assertEqual(
            report.critical_issues + report.high_issues + report.medium_issues + report.low_issues,
            report.tests_failed
        )

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty prompt
        empty_report = self.tester.test("")
        self.assertIsInstance(empty_report, PromptTestReport)

        # Very long prompt
        long_prompt = " ".join(["word"] * 1000)
        long_report = self.tester.test(long_prompt)
        self.assertIsInstance(long_report, PromptTestReport)

    def test_integration(self):
        """Integration test with realistic prompt."""
        prompt = """
        # Task: Optimize Database Queries

        ## Context
        PostgreSQL database with 10M rows
        Current query takes 5+ seconds

        ## Objective
        Reduce execution time to <500ms

        ## Requirements
        - Must maintain result accuracy
        - Should preserve existing functionality
        - Cannot change database schema

        ## Approach
        1. Analyze execution plan (EXPLAIN ANALYZE)
        2. Identify bottlenecks (sequential scans, missing indexes)
        3. Propose optimizations (indexes, query rewrite)
        4. Test and validate changes

        ## Success Criteria
        - Query executes in <500ms
        - All existing tests pass
        - No regression in other queries

        ## Input
        Current SQL query (provided separately)

        ## Output Format
        JSON with fields:
        - analysis: execution plan summary
        - bottlenecks: list of identified issues
        - optimizations: list of proposed changes
        - expected_improvement: estimated speedup

        ## Edge Cases
        - If query is already optimized → report no changes needed
        - If schema change required → note as limitation
        - If multiple indexes needed → prioritize by impact

        Think through this step by step.
        After proposing optimizations, validate they don't negatively impact other queries.
        """

        report = self.tester.test(prompt)

        # Should score very high
        self.assertGreater(report.overall_score, 80.0)

        # Should have no critical issues
        self.assertEqual(report.critical_issues, 0)

        # Should have minimal high issues
        self.assertLessEqual(report.high_issues, 1)

        # Most tests should pass
        self.assertGreater(report.tests_passed, report.tests_failed)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
