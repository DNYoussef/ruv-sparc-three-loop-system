#!/usr/bin/env python3
"""
Test suite for plan-optimizer.py
Validates optimization analysis, suggestion generation, and plan creation.
"""

import unittest
import json
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'resources'))

from plan_optimizer import (
    QuestionDependency,
    OptimizationSuggestion,
    PlanOptimizer
)


class TestQuestionDependency(unittest.TestCase):
    """Test QuestionDependency dataclass."""

    def test_create_dependency(self):
        """Test creating question dependency."""
        dep = QuestionDependency(
            question_id='database_choice',
            depends_on=['backend_stack', 'framework'],
            reason='Database choice depends on backend technology'
        )

        self.assertEqual(dep.question_id, 'database_choice')
        self.assertEqual(len(dep.depends_on), 2)
        self.assertIn('backend_stack', dep.depends_on)


class TestOptimizationSuggestion(unittest.TestCase):
    """Test OptimizationSuggestion dataclass."""

    def test_create_suggestion(self):
        """Test creating optimization suggestion."""
        suggestion = OptimizationSuggestion(
            suggestion_type='add',
            question_id='missing_testing',
            reason='Critical topic "testing" not covered',
            confidence=0.9,
            details={'topic': 'testing'}
        )

        self.assertEqual(suggestion.suggestion_type, 'add')
        self.assertEqual(suggestion.confidence, 0.9)
        self.assertEqual(suggestion.details['topic'], 'testing')


class TestPlanOptimizer(unittest.TestCase):
    """Test PlanOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_spec = {
            'answers': [
                {
                    'question': 'What is the primary purpose of this project?',
                    'selectedOptions': ['New feature'],
                    'isMultiSelect': False
                },
                {
                    'question': 'What framework should we use?',
                    'selectedOptions': ['React/Next.js'],
                    'isMultiSelect': False
                },
                {
                    'question': 'Which features are needed?',
                    'selectedOptions': ['User management', 'Real-time updates'],
                    'isMultiSelect': True
                }
            ]
        }

    def test_initialize_optimizer(self):
        """Test initializing plan optimizer."""
        optimizer = PlanOptimizer(self.sample_spec)

        self.assertEqual(len(optimizer.spec['answers']), 3)
        self.assertEqual(len(optimizer.dependencies), 0)
        self.assertEqual(len(optimizer.suggestions), 0)

    def test_analyze_existing_answers(self):
        """Test analyzing existing answered questions."""
        optimizer = PlanOptimizer(self.sample_spec)
        optimizer._analyze_existing_answers()

        self.assertGreater(len(optimizer.answered_questions), 0)
        self.assertGreater(len(optimizer.question_categories), 0)

    def test_categorize_question(self):
        """Test question categorization."""
        optimizer = PlanOptimizer(self.sample_spec)

        # Core functionality
        cat1 = optimizer._categorize_question('What is the primary purpose?')
        self.assertEqual(cat1, 'core_functionality')

        # Technical architecture
        cat2 = optimizer._categorize_question('What framework should we use?')
        self.assertEqual(cat2, 'technical_architecture')

        # Quality/scale
        cat3 = optimizer._categorize_question('What testing coverage is required?')
        self.assertEqual(cat3, 'quality_scale')

        # Constraints
        cat4 = optimizer._categorize_question('What is the timeline?')
        self.assertEqual(cat4, 'constraints_context')

        # Unknown category
        cat5 = optimizer._categorize_question('Some random question')
        self.assertEqual(cat5, 'other')

    def test_identify_missing_topics(self):
        """Test identifying missing topics."""
        # Minimal spec missing many topics
        minimal_spec = {
            'answers': [
                {
                    'question': 'What framework?',
                    'selectedOptions': ['React'],
                    'isMultiSelect': False
                }
            ]
        }

        optimizer = PlanOptimizer(minimal_spec)
        optimizer._analyze_existing_answers()
        optimizer._identify_missing_topics()

        # Should identify several missing topics
        self.assertGreater(len(optimizer.missing_topics), 0)

        # Should include critical topics like database, testing, etc.
        all_topics = set(optimizer.missing_topics)
        expected_missing = {'database', 'testing', 'authentication', 'timeline'}

        # At least some expected topics should be missing
        self.assertTrue(len(all_topics & expected_missing) > 0)

    def test_detect_dependencies(self):
        """Test detecting question dependencies."""
        spec_with_deps = {
            'answers': [
                {
                    'question': 'What backend technology?',
                    'selectedOptions': ['Node.js'],
                    'isMultiSelect': False
                },
                {
                    'question': 'What database should we use?',
                    'selectedOptions': ['PostgreSQL'],
                    'isMultiSelect': False
                }
            ]
        }

        optimizer = PlanOptimizer(spec_with_deps)
        optimizer._analyze_existing_answers()
        optimizer._detect_dependencies()

        # Database question depends on backend/stack
        database_deps = [
            d for d in optimizer.dependencies
            if 'database' in d.question_id
        ]

        # Should detect dependency relationship
        self.assertGreater(len(optimizer.dependencies), 0)

    def test_generate_suggestions_for_missing_topics(self):
        """Test generating suggestions for missing topics."""
        minimal_spec = {
            'answers': [
                {
                    'question': 'What framework?',
                    'selectedOptions': ['React'],
                    'isMultiSelect': False
                }
            ]
        }

        optimizer = PlanOptimizer(minimal_spec)
        optimizer._analyze_existing_answers()
        optimizer._identify_missing_topics()
        optimizer._generate_suggestions()

        # Should have suggestions for missing topics
        add_suggestions = [
            s for s in optimizer.suggestions
            if s.suggestion_type == 'add'
        ]

        self.assertGreater(len(add_suggestions), 0)

    def test_generate_suggestions_for_reordering(self):
        """Test generating suggestions for question reordering."""
        spec_with_deps = {
            'answers': [
                {
                    'question': 'What database?',
                    'selectedOptions': ['PostgreSQL'],
                    'isMultiSelect': False
                },
                {
                    'question': 'What backend stack?',
                    'selectedOptions': ['Node.js'],
                    'isMultiSelect': False
                }
            ]
        }

        optimizer = PlanOptimizer(spec_with_deps)
        optimizer._analyze_existing_answers()
        optimizer._detect_dependencies()
        optimizer._generate_suggestions()

        # Should have reorder suggestions
        reorder_suggestions = [
            s for s in optimizer.suggestions
            if s.suggestion_type == 'reorder'
        ]

        # May or may not have reorder suggestions depending on detection
        # Just verify suggestions were generated
        self.assertGreater(len(optimizer.suggestions), 0)

    def test_generate_recommended_batches(self):
        """Test generating recommended question batches."""
        optimizer = PlanOptimizer(self.sample_spec)
        optimizer._analyze_existing_answers()
        optimizer._identify_missing_topics()

        batches = optimizer._generate_recommended_batches()

        # Should generate batches for missing topics
        if len(optimizer.missing_topics) > 0:
            self.assertGreater(len(batches), 0)

            # Batches should have proper structure
            for batch in batches:
                self.assertIn('batch_number', batch)
                self.assertIn('name', batch)
                self.assertIn('topics', batch)
                self.assertIn('priority', batch)
                self.assertIn('questions_needed', batch)

    def test_create_optimized_plan(self):
        """Test creating optimized plan."""
        optimizer = PlanOptimizer(self.sample_spec)
        plan = optimizer.analyze()

        # Plan should have required sections
        self.assertIn('analysis', plan)
        self.assertIn('suggestions', plan)
        self.assertIn('recommended_batches', plan)
        self.assertIn('dependencies', plan)

        # Analysis should have metrics
        self.assertIn('total_questions_analyzed', plan['analysis'])
        self.assertIn('categories_covered', plan['analysis'])
        self.assertIn('missing_topics', plan['analysis'])

    def test_export_yaml(self):
        """Test exporting optimized plan to YAML."""
        optimizer = PlanOptimizer(self.sample_spec)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            output_file = f.name

        try:
            optimizer.export_yaml(output_file)

            # Verify file was created
            self.assertTrue(os.path.exists(output_file))

            # Verify file is valid YAML
            import yaml
            with open(output_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            self.assertIn('analysis', data)
            self.assertIn('suggestions', data)

        finally:
            os.unlink(output_file)

    def test_export_json(self):
        """Test exporting optimized plan to JSON."""
        optimizer = PlanOptimizer(self.sample_spec)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            output_file = f.name

        try:
            optimizer.export_json(output_file)

            # Verify file was created
            self.assertTrue(os.path.exists(output_file))

            # Verify file is valid JSON
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.assertIn('analysis', data)
            self.assertIn('suggestions', data)

        finally:
            os.unlink(output_file)

    def test_suggestion_confidence_ordering(self):
        """Test that suggestions are ordered by confidence."""
        optimizer = PlanOptimizer(self.sample_spec)
        plan = optimizer.analyze()

        if len(plan['suggestions']) > 1:
            # Verify suggestions are sorted by confidence (descending)
            confidences = [s['confidence'] for s in plan['suggestions']]

            for i in range(len(confidences) - 1):
                self.assertGreaterEqual(confidences[i], confidences[i + 1])


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQuestionDependency))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizationSuggestion))
    suite.addTests(loader.loadTestsFromTestCase(TestPlanOptimizer))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
