#!/usr/bin/env python3
"""
Test suite for prompt-analyzer.py

Tests the comprehensive prompt analysis and evaluation framework.
"""

import sys
import os
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'resources'))

from prompt_analyzer import PromptAnalyzer, AnalysisResult


class TestPromptAnalyzer(unittest.TestCase):
    """Test cases for PromptAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = PromptAnalyzer()

    def test_analyze_simple_prompt(self):
        """Test analysis of a simple, well-formed prompt."""
        prompt = """
        Analyze the quarterly sales data to identify trends.

        Focus on:
        1. Growth patterns
        2. Seasonal variations
        3. Regional differences

        Provide results in a structured report format.
        """

        result = self.analyzer.analyze(prompt)

        self.assertIsInstance(result, AnalysisResult)
        self.assertGreater(result.overall_score, 50.0)
        self.assertGreater(result.clarity_score, 70.0)
        self.assertGreater(result.word_count, 0)

    def test_analyze_complex_prompt(self):
        """Test analysis of a complex prompt with multiple elements."""
        prompt = """
        # Objective
        Create a comprehensive analysis of machine learning model performance.

        ## Context
        We have trained three models (Random Forest, XGBoost, Neural Network)
        on the same dataset. Need to compare their performance across multiple metrics.

        ## Requirements
        - Must include accuracy, precision, recall, F1-score
        - Should visualize results with confusion matrices
        - Cannot exceed 1000 words

        ## Approach
        First, calculate all metrics for each model.
        Then, create comparison tables and visualizations.
        Finally, provide recommendations based on the results.

        Validate your conclusions by considering trade-offs between metrics.
        """

        result = self.analyzer.analyze(prompt)

        # Should score high on structure
        self.assertGreater(result.structure_score, 80.0)

        # Should detect techniques (chain of thought, self-consistency)
        self.assertGreater(len(result.detected_patterns), 0)

        # Should have context
        self.assertGreater(result.context_score, 70.0)

    def test_detect_anti_patterns(self):
        """Test detection of common anti-patterns."""
        bad_prompt = """
        Quickly analyze this data and make it better.
        Obviously the results should be comprehensive but brief.
        Simply check the trends and clearly explain what you find.
        """

        result = self.analyzer.analyze(bad_prompt)

        # Should detect anti-patterns
        self.assertGreater(len(result.anti_patterns), 0)

        # Should have recommendations
        self.assertGreater(len(result.recommendations), 0)

        # Should score lower
        self.assertLess(result.overall_score, 60.0)

    def test_clarity_assessment(self):
        """Test clarity scoring."""
        # Clear prompt
        clear_prompt = "Analyze sales data to identify growth trends. Success means finding 3 key patterns."
        clear_result = self.analyzer.analyze(clear_prompt)

        # Unclear prompt
        unclear_prompt = "Look at the data and see what you think about it and stuff."
        unclear_result = self.analyzer.analyze(unclear_prompt)

        self.assertGreater(clear_result.clarity_score, unclear_result.clarity_score)

    def test_structure_scoring(self):
        """Test structure scoring."""
        # Structured prompt
        structured_prompt = """
        # Task: Data Analysis

        ## Step 1: Load Data
        Load the CSV file from the data directory.

        ## Step 2: Clean Data
        - Remove null values
        - Handle outliers
        - Normalize columns

        ## Step 3: Analyze
        Calculate summary statistics.
        """
        structured_result = self.analyzer.analyze(structured_prompt)

        # Unstructured prompt
        unstructured_prompt = "Load the CSV clean the data remove nulls handle outliers normalize then calculate stats"
        unstructured_result = self.analyzer.analyze(unstructured_prompt)

        self.assertGreater(structured_result.structure_score, unstructured_result.structure_score)

    def test_technique_detection(self):
        """Test detection of evidence-based techniques."""
        # Chain of thought
        cot_prompt = "Solve this problem step by step, explaining your reasoning at each stage."
        cot_result = self.analyzer.analyze(cot_prompt)
        self.assertIn('chain_of_thought', cot_result.detected_patterns)

        # Self-consistency
        sc_prompt = "After reaching your conclusion, validate it by considering alternative perspectives."
        sc_result = self.analyzer.analyze(sc_prompt)
        self.assertIn('self_consistency', sc_result.detected_patterns)

        # Few-shot
        fs_prompt = """
        Here are examples:
        Input: hello
        Output: HELLO

        Now process: world
        """
        fs_result = self.analyzer.analyze(fs_prompt)
        self.assertIn('few_shot', fs_result.detected_patterns)

    def test_complexity_assessment(self):
        """Test complexity level assessment."""
        # Simple
        simple = "Calculate 2 + 2"
        simple_result = self.analyzer.analyze(simple)
        self.assertEqual(simple_result.complexity_level, "Simple")

        # Complex
        complex_text = " ".join(["word"] * 600)
        complex_result = self.analyzer.analyze(complex_text)
        self.assertIn(complex_result.complexity_level, ["Complex", "Very Complex"])

    def test_recommendations_generation(self):
        """Test that recommendations are generated for low scores."""
        poor_prompt = "do the thing"
        result = self.analyzer.analyze(poor_prompt)

        # Should have multiple recommendations
        self.assertGreater(len(result.recommendations), 2)

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty prompt
        empty_result = self.analyzer.analyze("")
        self.assertIsInstance(empty_result, AnalysisResult)
        self.assertEqual(empty_result.word_count, 0)

        # Very long prompt
        long_prompt = " ".join(["word"] * 1000)
        long_result = self.analyzer.analyze(long_prompt)
        self.assertEqual(long_result.word_count, 1000)
        self.assertEqual(long_result.complexity_level, "Very Complex")

    def test_context_sufficiency(self):
        """Test context scoring."""
        # Good context
        with_context = """
        Context: We're building an e-commerce platform.
        Audience: Technical team members.
        Purpose: Design a scalable architecture.

        Requirements: Handle 10K concurrent users...
        """
        context_result = self.analyzer.analyze(with_context)
        self.assertGreater(context_result.context_score, 70.0)

        # No context
        no_context = "Build a system"
        no_context_result = self.analyzer.analyze(no_context)
        self.assertLess(no_context_result.context_score, context_result.context_score)

    def test_failure_mode_detection(self):
        """Test failure mode and anti-pattern detection."""
        # Contradictory requirements
        contradictory = "Provide a comprehensive yet brief analysis"
        result = self.analyzer.analyze(contradictory)
        self.assertLess(result.failure_score, 100.0)

        # Should have anti-patterns
        self.assertGreater(len(result.anti_patterns), 0)


class TestAnalysisResult(unittest.TestCase):
    """Test AnalysisResult dataclass."""

    def test_result_creation(self):
        """Test creating an AnalysisResult."""
        result = AnalysisResult(
            clarity_score=85.0,
            structure_score=75.0,
            context_score=80.0,
            technique_score=60.0,
            failure_score=90.0,
            formatting_score=70.0,
            overall_score=77.5,
            recommendations=["Test recommendation"],
            detected_patterns=["chain_of_thought"],
            anti_patterns=[],
            word_count=100,
            sentence_count=5,
            complexity_level="Medium"
        )

        self.assertEqual(result.clarity_score, 85.0)
        self.assertEqual(result.word_count, 100)
        self.assertEqual(result.complexity_level, "Medium")
        self.assertIsInstance(result.recommendations, list)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_full_analysis_pipeline(self):
        """Test complete analysis workflow."""
        analyzer = PromptAnalyzer()

        prompt = """
        # Task: Optimize Database Query Performance

        ## Context
        Our main application query is taking 5+ seconds to execute.
        Database: PostgreSQL 14, Table has 10M rows.

        ## Objective
        Reduce query execution time to under 500ms.

        ## Approach
        1. Analyze current query execution plan
        2. Identify bottlenecks
        3. Propose optimization strategies
        4. Implement and test solutions

        ## Constraints
        - Cannot change database schema
        - Must maintain result accuracy
        - Should preserve existing functionality

        ## Success Criteria
        - Query executes in <500ms
        - All test cases pass
        - No regression in other queries

        Think through this step by step.
        After proposing solutions, validate them against the constraints.
        """

        result = analyzer.analyze(prompt)

        # Comprehensive checks
        self.assertGreater(result.overall_score, 75.0)
        self.assertGreater(result.clarity_score, 80.0)
        self.assertGreater(result.structure_score, 80.0)
        self.assertGreater(result.context_score, 80.0)

        # Should detect techniques
        self.assertIn('chain_of_thought', result.detected_patterns)
        self.assertIn('self_consistency', result.detected_patterns)

        # Should have minimal anti-patterns
        self.assertEqual(len(result.anti_patterns), 0)

        # Should have few or no recommendations (it's already good)
        self.assertLessEqual(len(result.recommendations), 2)


def run_tests():
    """Run all tests."""
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
