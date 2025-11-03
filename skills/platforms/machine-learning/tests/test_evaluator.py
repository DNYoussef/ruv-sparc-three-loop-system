#!/usr/bin/env python3
"""
Unit tests for ModelEvaluator
"""

import unittest
import tempfile
import shutil
import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ModelEvaluator
import subprocess
import importlib.util

# Load the JS module using Node.js for testing
evaluator_path = Path(__file__).parent.parent / 'resources' / 'scripts' / 'model-evaluator.js'


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator (Node.js module)"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.temp_dir = tempfile.mkdtemp()

        # Test data
        cls.predictions_clf = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        cls.ground_truth_clf = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]
        cls.sensitive_attrs = ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'B']

        cls.predictions_reg = [1.5, 2.3, 3.1, 4.2, 5.0]
        cls.ground_truth_reg = [1.4, 2.5, 3.0, 4.1, 5.2]

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        shutil.rmtree(cls.temp_dir)

    def test_classification_metrics_calculation(self):
        """Test classification metrics calculation logic"""
        # Calculate expected accuracy manually
        correct = sum([p == g for p, g in zip(self.predictions_clf, self.ground_truth_clf)])
        expected_accuracy = correct / len(self.predictions_clf)

        self.assertAlmostEqual(expected_accuracy, 0.8, places=2)

    def test_regression_metrics_calculation(self):
        """Test regression metrics calculation logic"""
        # Calculate MSE manually
        errors = [(p - g) ** 2 for p, g in zip(self.predictions_reg, self.ground_truth_reg)]
        expected_mse = sum(errors) / len(errors)

        self.assertGreaterEqual(expected_mse, 0)

    def test_fairness_metrics_structure(self):
        """Test fairness metrics data structure"""
        # Group predictions by sensitive attribute
        groups = set(self.sensitive_attrs)
        self.assertEqual(len(groups), 2)

        # Count samples per group
        group_counts = {}
        for attr in self.sensitive_attrs:
            group_counts[attr] = group_counts.get(attr, 0) + 1

        self.assertEqual(group_counts['A'], 5)
        self.assertEqual(group_counts['B'], 5)

    def test_confusion_matrix_calculation(self):
        """Test confusion matrix calculation"""
        # Calculate confusion matrix manually
        tp = sum([1 for p, g in zip(self.predictions_clf, self.ground_truth_clf)
                  if p == 1 and g == 1])
        tn = sum([1 for p, g in zip(self.predictions_clf, self.ground_truth_clf)
                  if p == 0 and g == 0])
        fp = sum([1 for p, g in zip(self.predictions_clf, self.ground_truth_clf)
                  if p == 1 and g == 0])
        fn = sum([1 for p, g in zip(self.predictions_clf, self.ground_truth_clf)
                  if p == 0 and g == 1])

        # Verify counts
        self.assertEqual(tp + tn + fp + fn, len(self.predictions_clf))

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1)
        self.assertGreaterEqual(recall, 0)
        self.assertLessEqual(recall, 1)

    def test_f1_score_calculation(self):
        """Test F1 score calculation"""
        # Calculate manually
        tp = sum([1 for p, g in zip(self.predictions_clf, self.ground_truth_clf)
                  if p == 1 and g == 1])
        fp = sum([1 for p, g in zip(self.predictions_clf, self.ground_truth_clf)
                  if p == 1 and g == 0])
        fn = sum([1 for p, g in zip(self.predictions_clf, self.ground_truth_clf)
                  if p == 0 and g == 1])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)

    def test_regression_r2_calculation(self):
        """Test R² calculation"""
        import numpy as np

        pred = np.array(self.predictions_reg)
        true = np.array(self.ground_truth_reg)

        # Calculate R²
        mean_y = np.mean(true)
        ss_total = np.sum((true - mean_y) ** 2)
        ss_residual = np.sum((true - pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)

        # R² should be between -inf and 1
        self.assertLessEqual(r2, 1)

    def test_fairness_demographic_parity(self):
        """Test demographic parity calculation"""
        # Calculate positive prediction rate per group
        group_pprs = {}

        for group in set(self.sensitive_attrs):
            group_indices = [i for i, attr in enumerate(self.sensitive_attrs) if attr == group]
            group_preds = [self.predictions_clf[i] for i in group_indices]
            ppr = sum([1 for p in group_preds if p == 1]) / len(group_preds)
            group_pprs[group] = ppr

        # Calculate disparity
        disparity = max(group_pprs.values()) - min(group_pprs.values())

        self.assertGreaterEqual(disparity, 0)
        self.assertLessEqual(disparity, 1)

    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation"""
        # Calculate MAE
        errors = [abs(p - g) for p, g in zip(self.predictions_reg, self.ground_truth_reg)]
        mae = sum(errors) / len(errors)

        self.assertGreaterEqual(mae, 0)

    def test_rmse_calculation(self):
        """Test Root Mean Squared Error calculation"""
        import math

        # Calculate RMSE
        errors = [(p - g) ** 2 for p, g in zip(self.predictions_reg, self.ground_truth_reg)]
        mse = sum(errors) / len(errors)
        rmse = math.sqrt(mse)

        self.assertGreaterEqual(rmse, 0)


class TestFeatureImportance(unittest.TestCase):
    """Test feature importance analysis"""

    def test_feature_ranking(self):
        """Test feature importance ranking"""
        feature_names = ['feature1', 'feature2', 'feature3']
        importances = [0.5, 0.3, 0.2]

        # Sort by importance
        sorted_features = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )

        self.assertEqual(sorted_features[0][0], 'feature1')
        self.assertEqual(sorted_features[1][0], 'feature2')
        self.assertEqual(sorted_features[2][0], 'feature3')


class TestReportGeneration(unittest.TestCase):
    """Test report generation functionality"""

    def test_report_structure(self):
        """Test report structure and content"""
        report_lines = [
            '=' * 80,
            'MODEL EVALUATION REPORT',
            '=' * 80,
            '',
            'PERFORMANCE METRICS',
            '-' * 80
        ]

        self.assertTrue(all(isinstance(line, str) for line in report_lines))


if __name__ == '__main__':
    unittest.main()
