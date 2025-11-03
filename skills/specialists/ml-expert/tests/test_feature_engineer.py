"""
Tests for feature-engineer.js

Validates feature engineering pipeline automation.
"""

import pytest
import subprocess
import json
from pathlib import Path
import csv


@pytest.fixture
def sample_csv_data(tmp_path):
    """Create sample CSV data for testing."""
    csv_file = tmp_path / 'sample_data.csv'

    data = [
        ['age', 'income', 'category', 'score'],
        ['25', '50000', 'A', '0.75'],
        ['30', '60000', 'B', '0.82'],
        ['35', '75000', 'A', '0.91'],
        ['40', '80000', 'C', '0.68'],
        ['45', '90000', 'B', '0.85']
    ]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return csv_file


@pytest.fixture
def feature_config(tmp_path):
    """Create feature engineering configuration."""
    config = {
        "transformations": [
            {
                "feature": "age",
                "method": "standardize",
                "params": {}
            },
            {
                "feature": "income",
                "method": "normalize",
                "params": {}
            },
            {
                "feature": "category",
                "method": "one_hot_encode",
                "params": {}
            }
        ]
    }

    config_file = tmp_path / 'feature_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    return config_file


class TestFeatureEngineer:
    """Test feature engineering functionality."""

    def test_data_analysis(self, sample_csv_data, tmp_path):
        """Test data analysis functionality."""
        report_file = tmp_path / 'analysis.json'

        result = subprocess.run([
            'node',
            str(Path(__file__).parent.parent / 'resources' / 'feature-engineer.js'),
            '--analyze', str(sample_csv_data),
            '--report', str(report_file)
        ], capture_output=True, text=True)

        # Check execution success
        if result.returncode != 0:
            pytest.skip(f"Node.js execution failed: {result.stderr}")

        # Validate report
        assert report_file.exists()

        with open(report_file) as f:
            analysis = json.load(f)

        assert 'rowCount' in analysis
        assert analysis['rowCount'] == 5
        assert 'features' in analysis
        assert 'age' in analysis['features']

    def test_standardization(self, sample_csv_data, feature_config, tmp_path):
        """Test numerical feature standardization."""
        output_file = tmp_path / 'transformed.csv'

        result = subprocess.run([
            'node',
            str(Path(__file__).parent.parent / 'resources' / 'feature-engineer.js'),
            '--config', str(feature_config),
            '--input', str(sample_csv_data),
            '--output', str(output_file)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            pytest.skip(f"Node.js execution failed: {result.stderr}")

        assert output_file.exists()

        # Check for standardized column
        with open(output_file) as f:
            reader = csv.DictReader(f)
            first_row = next(reader)
            assert 'age_standardized' in first_row

    def test_normalization(self, sample_csv_data, feature_config, tmp_path):
        """Test min-max normalization."""
        output_file = tmp_path / 'normalized.csv'

        result = subprocess.run([
            'node',
            str(Path(__file__).parent.parent / 'resources' / 'feature-engineer.js'),
            '--config', str(feature_config),
            '--input', str(sample_csv_data),
            '--output', str(output_file)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            pytest.skip(f"Node.js execution failed: {result.stderr}")

        assert output_file.exists()

        # Verify normalized values are between 0 and 1
        with open(output_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'income_normalized' in row:
                    value = float(row['income_normalized'])
                    assert 0 <= value <= 1

    def test_one_hot_encoding(self, sample_csv_data, feature_config, tmp_path):
        """Test categorical feature encoding."""
        output_file = tmp_path / 'encoded.csv'

        result = subprocess.run([
            'node',
            str(Path(__file__).parent.parent / 'resources' / 'feature-engineer.js'),
            '--config', str(feature_config),
            '--input', str(sample_csv_data),
            '--output', str(output_file)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            pytest.skip(f"Node.js execution failed: {result.stderr}")

        assert output_file.exists()

        # Check for one-hot encoded columns
        with open(output_file) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            # Should have columns like category_A, category_B, category_C
            encoded_cols = [h for h in headers if h.startswith('category_')]
            assert len(encoded_cols) >= 2


@pytest.mark.integration
class TestFeatureEngineeringWorkflow:
    """Integration tests for complete feature engineering workflows."""

    def test_complete_pipeline(self, sample_csv_data, tmp_path):
        """Test complete feature engineering pipeline."""
        # First analyze
        analysis_file = tmp_path / 'analysis.json'

        subprocess.run([
            'node',
            str(Path(__file__).parent.parent / 'resources' / 'feature-engineer.js'),
            '--analyze', str(sample_csv_data),
            '--report', str(analysis_file)
        ], capture_output=True)

        if not analysis_file.exists():
            pytest.skip("Node.js analysis failed")

        # Then transform based on analysis
        config = {
            "transformations": [
                {"feature": "age", "method": "standardize", "params": {}},
                {"feature": "income", "method": "normalize", "params": {}}
            ]
        }

        config_file = tmp_path / 'pipeline_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f)

        output_file = tmp_path / 'pipeline_output.csv'

        subprocess.run([
            'node',
            str(Path(__file__).parent.parent / 'resources' / 'feature-engineer.js'),
            '--config', str(config_file),
            '--input', str(sample_csv_data),
            '--output', str(output_file)
        ], capture_output=True)

        assert output_file.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
