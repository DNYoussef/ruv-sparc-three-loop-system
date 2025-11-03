#!/usr/bin/env python3
"""
Unit tests for DataPreprocessor
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from resources.scripts.data_preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.temp_dir = tempfile.mkdtemp()

        # Create test dataset
        np.random.seed(42)
        cls.test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'feature4': np.random.randn(100),
            'label': np.random.randint(0, 2, 100)
        })

        # Add missing values
        cls.test_data.loc[0:5, 'feature1'] = np.nan
        cls.test_data.loc[10:15, 'feature2'] = np.nan

        # Add outliers
        cls.test_data.loc[20:25, 'feature4'] = 100

        cls.test_csv = os.path.join(cls.temp_dir, 'test_data.csv')
        cls.test_data.to_csv(cls.test_csv, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up before each test"""
        self.preprocessor = DataPreprocessor(
            scaling_method='standard',
            imputation_strategy='mean',
            handle_outliers=True,
            random_state=42
        )

    def test_init(self):
        """Test preprocessor initialization"""
        self.assertIsNotNone(self.preprocessor)
        self.assertEqual(self.preprocessor.scaling_method, 'standard')
        self.assertEqual(self.preprocessor.imputation_strategy, 'mean')
        self.assertTrue(self.preprocessor.handle_outliers)

    def test_load_data(self):
        """Test data loading"""
        df = self.preprocessor.load_data(self.test_csv)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertEqual(len(df.columns), 5)

    def test_analyze_data(self):
        """Test data analysis"""
        df = self.preprocessor.load_data(self.test_csv)
        analysis = self.preprocessor.analyze_data(df)

        self.assertIn('shape', analysis)
        self.assertIn('missing_values', analysis)
        self.assertIn('outliers', analysis)
        self.assertIn('categorical_columns', analysis)
        self.assertIn('numeric_columns', analysis)

        # Check missing values detected
        self.assertTrue(analysis['missing_values']['feature1'] > 0)
        self.assertTrue(analysis['missing_values']['feature2'] > 0)

        # Check outliers detected
        self.assertTrue(analysis['outliers']['feature4'] > 0)

    def test_handle_missing_values(self):
        """Test missing value imputation"""
        df = self.preprocessor.load_data(self.test_csv)
        df_imputed = self.preprocessor.handle_missing_values(df)

        # Check no missing values remain
        self.assertEqual(df_imputed.isnull().sum().sum(), 0)

    def test_handle_outliers(self):
        """Test outlier handling"""
        df = self.preprocessor.load_data(self.test_csv)
        df_no_outliers = self.preprocessor.handle_outliers_iqr(df)

        # Check outliers capped
        Q3 = self.test_data['feature4'].quantile(0.75)
        IQR = Q3 - self.test_data['feature4'].quantile(0.25)
        upper_bound = Q3 + 1.5 * IQR

        self.assertTrue(df_no_outliers['feature4'].max() <= upper_bound)

    def test_encode_categorical(self):
        """Test categorical encoding"""
        df = self.preprocessor.load_data(self.test_csv)
        df_encoded = self.preprocessor.encode_categorical(df)

        # Check categorical column is now numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(df_encoded['feature3']))

    def test_scale_features(self):
        """Test feature scaling"""
        df = self.preprocessor.load_data(self.test_csv)
        df = self.preprocessor.handle_missing_values(df)
        df = self.preprocessor.encode_categorical(df)
        df_scaled = self.preprocessor.scale_features(df)

        # Check scaling applied (mean ~ 0, std ~ 1 for standard scaler)
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'label':  # Exclude target
                self.assertAlmostEqual(df_scaled[col].mean(), 0, places=1)
                self.assertAlmostEqual(df_scaled[col].std(), 1, places=1)

    def test_engineer_features(self):
        """Test feature engineering"""
        df = self.preprocessor.load_data(self.test_csv)
        df = self.preprocessor.handle_missing_values(df)
        df_engineered = self.preprocessor.engineer_features(df)

        # Check new features created
        self.assertTrue('feature1_squared' in df_engineered.columns)
        self.assertTrue('feature1_log' in df_engineered.columns)

    def test_prepare_data(self):
        """Test complete data preparation pipeline"""
        train_df, val_df, test_df = self.preprocessor.prepare_data(
            self.test_csv,
            target='label',
            test_size=0.2,
            val_size=0.1
        )

        # Check splits
        total_samples = len(self.test_data)
        expected_test = int(total_samples * 0.2)
        expected_val = int((total_samples - expected_test) * 0.1)
        expected_train = total_samples - expected_test - expected_val

        self.assertAlmostEqual(len(train_df), expected_train, delta=2)
        self.assertAlmostEqual(len(val_df), expected_val, delta=2)
        self.assertAlmostEqual(len(test_df), expected_test, delta=2)

        # Check no missing values
        self.assertEqual(train_df.isnull().sum().sum(), 0)
        self.assertEqual(val_df.isnull().sum().sum(), 0)
        self.assertEqual(test_df.isnull().sum().sum(), 0)

    def test_save_load_preprocessor(self):
        """Test preprocessor persistence"""
        # Fit preprocessor
        df = self.preprocessor.load_data(self.test_csv)
        df = self.preprocessor.handle_missing_values(df)
        df = self.preprocessor.encode_categorical(df)
        self.preprocessor.scale_features(df)

        # Save
        save_path = os.path.join(self.temp_dir, 'preprocessor.pkl')
        self.preprocessor.save_preprocessor(save_path)
        self.assertTrue(os.path.exists(save_path))

        # Load
        loaded = DataPreprocessor.load_preprocessor(save_path)
        self.assertIsNotNone(loaded.scaler)
        self.assertIsNotNone(loaded.imputer)

    def test_different_scalers(self):
        """Test different scaling methods"""
        scalers = ['standard', 'minmax']

        for scaler in scalers:
            preprocessor = DataPreprocessor(scaling_method=scaler)
            df = preprocessor.load_data(self.test_csv)
            df = preprocessor.handle_missing_values(df)
            df = preprocessor.encode_categorical(df)
            df_scaled = preprocessor.scale_features(df)

            self.assertIsNotNone(df_scaled)


class TestImputationStrategies(unittest.TestCase):
    """Test different imputation strategies"""

    def test_mean_imputation(self):
        """Test mean imputation"""
        preprocessor = DataPreprocessor(imputation_strategy='mean')
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [5, 6, 7, 8]
        })

        df_imputed = preprocessor.handle_missing_values(df)
        expected_value = df['col1'].mean()
        self.assertAlmostEqual(df_imputed['col1'].iloc[2], expected_value, places=5)

    def test_median_imputation(self):
        """Test median imputation"""
        preprocessor = DataPreprocessor(imputation_strategy='median')
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [5, 6, 7, 8]
        })

        df_imputed = preprocessor.handle_missing_values(df)
        expected_value = df['col1'].median()
        self.assertEqual(df_imputed['col1'].iloc[2], expected_value)


if __name__ == '__main__':
    unittest.main()
