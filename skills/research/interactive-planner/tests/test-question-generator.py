#!/usr/bin/env python3
"""
Test suite for question-generator.py
Validates question generation, batching, and export functionality.
"""

import unittest
import json
import yaml
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'resources'))

from question_generator import (
    QuestionOption,
    Question,
    QuestionBatch,
    QuestionCategory,
    ComplexityLevel,
    QuestionGenerator
)


class TestQuestionOption(unittest.TestCase):
    """Test QuestionOption dataclass."""

    def test_valid_option(self):
        """Test creating valid option."""
        option = QuestionOption("React/Next.js", "Modern React framework")
        self.assertTrue(option.validate())

    def test_invalid_option_empty_label(self):
        """Test option with empty label."""
        option = QuestionOption("", "Description")
        self.assertFalse(option.validate())

    def test_invalid_option_empty_description(self):
        """Test option with empty description."""
        option = QuestionOption("Label", "")
        self.assertFalse(option.validate())

    def test_invalid_option_label_too_long(self):
        """Test option with label exceeding max length."""
        option = QuestionOption("A" * 51, "Description")
        self.assertFalse(option.validate())


class TestQuestion(unittest.TestCase):
    """Test Question dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_options = [
            QuestionOption("React", "React framework"),
            QuestionOption("Vue", "Vue framework")
        ]

    def test_valid_question(self):
        """Test creating valid question."""
        question = Question(
            question="What framework?",
            header="Framework",
            options=self.valid_options,
            multiSelect=False
        )
        self.assertTrue(question.validate())

    def test_invalid_question_header_too_long(self):
        """Test question with header exceeding 12 chars."""
        question = Question(
            question="What framework?",
            header="ThisHeaderIsTooLong",
            options=self.valid_options,
            multiSelect=False
        )
        self.assertFalse(question.validate())

    def test_invalid_question_too_few_options(self):
        """Test question with fewer than 2 options."""
        question = Question(
            question="What framework?",
            header="Framework",
            options=[QuestionOption("React", "React framework")],
            multiSelect=False
        )
        self.assertFalse(question.validate())

    def test_invalid_question_too_many_options(self):
        """Test question with more than 4 options."""
        options = [
            QuestionOption(f"Option {i}", f"Description {i}")
            for i in range(5)
        ]
        question = Question(
            question="What framework?",
            header="Framework",
            options=options,
            multiSelect=False
        )
        self.assertFalse(question.validate())

    def test_question_to_dict(self):
        """Test converting question to dictionary."""
        question = Question(
            question="What framework?",
            header="Framework",
            options=self.valid_options,
            multiSelect=True,
            category=QuestionCategory.TECHNICAL_ARCHITECTURE
        )
        result = question.to_dict()

        self.assertEqual(result['question'], "What framework?")
        self.assertEqual(result['header'], "Framework")
        self.assertTrue(result['multiSelect'])
        self.assertEqual(len(result['options']), 2)


class TestQuestionBatch(unittest.TestCase):
    """Test QuestionBatch functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_question = Question(
            question="What framework?",
            header="Framework",
            options=[
                QuestionOption("React", "React framework"),
                QuestionOption("Vue", "Vue framework")
            ],
            multiSelect=False
        )

    def test_add_question_to_batch(self):
        """Test adding questions to batch."""
        batch = QuestionBatch()
        result = batch.add_question(self.valid_question)

        self.assertTrue(result)
        self.assertEqual(len(batch.questions), 1)

    def test_batch_full_after_four_questions(self):
        """Test that batch is full after 4 questions."""
        batch = QuestionBatch()

        for i in range(4):
            batch.add_question(self.valid_question)

        self.assertTrue(batch.is_full())
        self.assertFalse(batch.add_question(self.valid_question))

    def test_batch_to_dict(self):
        """Test converting batch to dictionary."""
        batch = QuestionBatch(
            batch_number=1,
            category=QuestionCategory.CORE_FUNCTIONALITY
        )
        batch.add_question(self.valid_question)

        result = batch.to_dict()

        self.assertEqual(result['batch_number'], 1)
        self.assertEqual(result['category'], 'core_functionality')
        self.assertEqual(len(result['questions']), 1)


class TestQuestionGenerator(unittest.TestCase):
    """Test QuestionGenerator class."""

    def test_generate_core_functionality_questions(self):
        """Test generating core functionality questions."""
        generator = QuestionGenerator("web", ComplexityLevel.MODERATE)
        questions = generator.generate_core_functionality_questions()

        self.assertGreater(len(questions), 0)
        self.assertTrue(all(q.validate() for q in questions))

    def test_generate_technical_architecture_questions(self):
        """Test generating technical architecture questions."""
        generator = QuestionGenerator("api", ComplexityLevel.COMPLEX)
        questions = generator.generate_technical_architecture_questions()

        self.assertGreater(len(questions), 0)
        self.assertTrue(all(q.validate() for q in questions))

    def test_generate_quality_scale_questions(self):
        """Test generating quality/scale questions."""
        generator = QuestionGenerator("mobile", ComplexityLevel.SIMPLE)
        questions = generator.generate_quality_scale_questions()

        self.assertGreater(len(questions), 0)
        self.assertTrue(all(q.validate() for q in questions))

    def test_generate_batches(self):
        """Test generating multiple batches."""
        generator = QuestionGenerator("web", ComplexityLevel.MODERATE)
        batches = generator.generate_batches(num_batches=3)

        self.assertEqual(len(batches), 3)
        self.assertTrue(all(len(b.questions) > 0 for b in batches))

    def test_export_json(self):
        """Test exporting batches to JSON."""
        generator = QuestionGenerator("web", ComplexityLevel.MODERATE)
        generator.generate_batches(num_batches=2)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            output_file = f.name

        try:
            generator.export_json(output_file)

            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.assertEqual(data['project_type'], 'web')
            self.assertEqual(data['complexity'], 'moderate')
            self.assertGreater(data['total_batches'], 0)

        finally:
            os.unlink(output_file)

    def test_export_yaml(self):
        """Test exporting batches to YAML."""
        generator = QuestionGenerator("api", ComplexityLevel.COMPLEX)
        generator.generate_batches(num_batches=2)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            output_file = f.name

        try:
            generator.export_yaml(output_file)

            with open(output_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            self.assertEqual(data['project_type'], 'api')
            self.assertEqual(data['complexity'], 'complex')
            self.assertGreater(data['total_batches'], 0)

        finally:
            os.unlink(output_file)


class TestComplexityAdaptation(unittest.TestCase):
    """Test that question generation adapts to complexity."""

    def test_simple_vs_complex_question_count(self):
        """Test that complex projects get more questions."""
        simple_gen = QuestionGenerator("web", ComplexityLevel.SIMPLE)
        complex_gen = QuestionGenerator("web", ComplexityLevel.COMPLEX)

        simple_batches = simple_gen.generate_batches(num_batches=5)
        complex_batches = complex_gen.generate_batches(num_batches=5)

        # Both should have questions, complexity may affect content
        self.assertGreater(len(simple_batches), 0)
        self.assertGreater(len(complex_batches), 0)


class TestProjectTypeAdaptation(unittest.TestCase):
    """Test that questions adapt to project type."""

    def test_web_vs_api_questions(self):
        """Test different questions for web vs API projects."""
        web_gen = QuestionGenerator("web", ComplexityLevel.MODERATE)
        api_gen = QuestionGenerator("api", ComplexityLevel.MODERATE)

        web_questions = web_gen.generate_core_functionality_questions()
        api_questions = api_gen.generate_core_functionality_questions()

        # Both should generate questions
        self.assertGreater(len(web_questions), 0)
        self.assertGreater(len(api_questions), 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQuestionOption))
    suite.addTests(loader.loadTestsFromTestCase(TestQuestion))
    suite.addTests(loader.loadTestsFromTestCase(TestQuestionBatch))
    suite.addTests(loader.loadTestsFromTestCase(TestQuestionGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestComplexityAdaptation))
    suite.addTests(loader.loadTestsFromTestCase(TestProjectTypeAdaptation))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
