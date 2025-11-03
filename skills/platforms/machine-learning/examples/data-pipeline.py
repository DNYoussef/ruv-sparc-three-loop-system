#!/usr/bin/env python3
"""
Complete Data Pipeline Example
Demonstrates end-to-end data preprocessing workflow with:
- Data loading from multiple sources
- Data quality analysis
- Missing value handling
- Feature engineering
- Data augmentation
- Train/val/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from resources.scripts.data_preprocessor import DataPreprocessor


# Example 1: Basic Data Preprocessing
def example_basic_preprocessing():
    """
    Example 1: Basic data preprocessing pipeline
    Shows standard workflow for cleaning and preparing data
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Data Preprocessing")
    print("=" * 80)

    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'hours_worked': np.random.normal(40, 10, n_samples),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples),
        'label': np.random.randint(0, 2, n_samples)
    })

    # Add missing values
    data.loc[np.random.choice(n_samples, 50, replace=False), 'income'] = np.nan
    data.loc[np.random.choice(n_samples, 30, replace=False), 'hours_worked'] = np.nan

    # Add outliers
    data.loc[np.random.choice(n_samples, 20, replace=False), 'income'] = 500000

    # Save to temporary CSV
    temp_path = 'data/temp_data.csv'
    Path('data').mkdir(exist_ok=True)
    data.to_csv(temp_path, index=False)

    print(f"\nDataset created: {n_samples} samples, {len(data.columns)} columns")

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        scaling_method='standard',
        imputation_strategy='mean',
        handle_outliers=True,
        random_state=42
    )

    # Analyze data quality
    print("\n--- Data Quality Analysis ---")
    analysis = preprocessor.analyze_data(data)

    print(f"Shape: {analysis['shape']}")
    print(f"Data types: {analysis['dtypes']}")
    print(f"Missing values: {sum(analysis['missing_values'].values())} total")
    print(f"Duplicates: {analysis['duplicates']}")

    # Prepare data
    print("\n--- Data Preparation ---")
    train_df, val_df, test_df = preprocessor.prepare_data(
        temp_path,
        target='label',
        test_size=0.2,
        val_size=0.1,
        stratify=True
    )

    print(f"Train set: {train_df.shape}")
    print(f"Validation set: {val_df.shape}")
    print(f"Test set: {test_df.shape}")

    # Verify no missing values
    print(f"\nMissing values after preprocessing:")
    print(f"  Train: {train_df.isnull().sum().sum()}")
    print(f"  Val: {val_df.isnull().sum().sum()}")
    print(f"  Test: {test_df.isnull().sum().sum()}")

    # Save preprocessor
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    print("\nPreprocessor saved to: models/preprocessor.pkl")


# Example 2: Advanced Feature Engineering
def example_feature_engineering():
    """
    Example 2: Advanced feature engineering
    Shows how to create sophisticated features
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Advanced Feature Engineering")
    print("=" * 80)

    # Create synthetic dataset with time series features
    np.random.seed(42)
    n_samples = 500

    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'temperature': 20 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.randn(n_samples) * 2,
        'humidity': 60 + 20 * np.cos(np.linspace(0, 4*np.pi, n_samples)) + np.random.randn(n_samples) * 5,
        'wind_speed': np.random.gamma(2, 2, n_samples),
        'precipitation': np.random.exponential(2, n_samples),
        'sales': np.random.randint(100, 1000, n_samples)
    })

    print(f"Original features: {list(data.columns)}")

    # Feature engineering
    engineered_data = data.copy()

    # Time-based features
    engineered_data['day_of_week'] = engineered_data['date'].dt.dayofweek
    engineered_data['day_of_month'] = engineered_data['date'].dt.day
    engineered_data['month'] = engineered_data['date'].dt.month
    engineered_data['quarter'] = engineered_data['date'].dt.quarter
    engineered_data['is_weekend'] = (engineered_data['day_of_week'] >= 5).astype(int)

    # Rolling statistics
    window = 7
    engineered_data['temp_rolling_mean'] = engineered_data['temperature'].rolling(window).mean()
    engineered_data['temp_rolling_std'] = engineered_data['temperature'].rolling(window).std()
    engineered_data['sales_rolling_mean'] = engineered_data['sales'].rolling(window).mean()

    # Lag features
    for lag in [1, 7, 30]:
        engineered_data[f'sales_lag_{lag}'] = engineered_data['sales'].shift(lag)
        engineered_data[f'temp_lag_{lag}'] = engineered_data['temperature'].shift(lag)

    # Interaction features
    engineered_data['temp_humidity'] = engineered_data['temperature'] * engineered_data['humidity']
    engineered_data['temp_wind'] = engineered_data['temperature'] * engineered_data['wind_speed']

    # Polynomial features
    engineered_data['temperature_squared'] = engineered_data['temperature'] ** 2
    engineered_data['humidity_squared'] = engineered_data['humidity'] ** 2

    # Binning
    engineered_data['temp_category'] = pd.cut(
        engineered_data['temperature'],
        bins=[-np.inf, 15, 25, np.inf],
        labels=['cold', 'moderate', 'hot']
    )

    # Drop date column and NaN rows
    engineered_data = engineered_data.drop('date', axis=1)
    engineered_data = engineered_data.dropna()

    print(f"\nEngineered features: {len(engineered_data.columns)} total")
    print(f"New features added: {len(engineered_data.columns) - len(data.columns) + 1}")
    print(f"\nSample of engineered data:")
    print(engineered_data.head())


# Example 3: Cross-Validation Data Pipeline
def example_cv_pipeline():
    """
    Example 3: Cross-validation data pipeline
    Shows how to prepare data for k-fold cross-validation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Cross-Validation Data Pipeline")
    print("=" * 80)

    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create target with class imbalance
    y = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Class distribution: {np.bincount(y)}")

    # Setup stratified k-fold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"\nPerforming {n_splits}-fold stratified cross-validation")

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Initialize preprocessor per fold
        preprocessor = DataPreprocessor(
            scaling_method='standard',
            imputation_strategy='mean',
            random_state=42
        )

        # Fit scaler on training data only
        X_train_scaled = preprocessor.scale_features(X_train, fit=True)
        X_val_scaled = preprocessor.scale_features(X_val, fit=False)

        print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}")
        print(f"Train class dist: {np.bincount(y_train)}")
        print(f"Val class dist: {np.bincount(y_val)}")

        # Verify scaling
        print(f"Train mean: {X_train_scaled.mean().mean():.4f}, std: {X_train_scaled.std().mean():.4f}")
        print(f"Val mean: {X_val_scaled.mean().mean():.4f}, std: {X_val_scaled.std().mean():.4f}")

        fold_results.append({
            'fold': fold,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'train_positive_ratio': y_train.sum() / len(y_train),
            'val_positive_ratio': y_val.sum() / len(y_val)
        })

    # Summary
    print("\n--- Cross-Validation Summary ---")
    results_df = pd.DataFrame(fold_results)
    print(results_df)

    print(f"\nAverage positive class ratio:")
    print(f"  Train: {results_df['train_positive_ratio'].mean():.4f}")
    print(f"  Val: {results_df['val_positive_ratio'].mean():.4f}")


# Example 4: Data Augmentation Pipeline
def example_data_augmentation():
    """
    Example 4: Data augmentation for imbalanced datasets
    Shows techniques to handle class imbalance
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Data Augmentation for Imbalanced Data")
    print("=" * 80)

    # Create imbalanced dataset
    np.random.seed(42)
    n_majority = 900
    n_minority = 100
    n_features = 5

    # Majority class
    X_majority = pd.DataFrame(
        np.random.randn(n_majority, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_majority = np.zeros(n_majority)

    # Minority class
    X_minority = pd.DataFrame(
        np.random.randn(n_minority, n_features) + 2,  # Different distribution
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_minority = np.ones(n_minority)

    # Combine
    X = pd.concat([X_majority, X_minority], ignore_index=True)
    y = np.concatenate([y_majority, y_minority])

    print(f"Original dataset:")
    print(f"  Total: {len(y)} samples")
    print(f"  Class 0: {(y == 0).sum()} samples")
    print(f"  Class 1: {(y == 1).sum()} samples")
    print(f"  Imbalance ratio: {(y == 0).sum() / (y == 1).sum():.2f}:1")

    # Augmentation technique: SMOTE (simplified version)
    print("\n--- Applying SMOTE-like augmentation ---")

    # Get minority class samples
    minority_indices = np.where(y == 1)[0]
    X_minority = X.iloc[minority_indices]

    # Generate synthetic samples
    n_synthetic = n_majority - n_minority
    synthetic_samples = []

    for _ in range(n_synthetic):
        # Random pair of minority samples
        idx1, idx2 = np.random.choice(len(X_minority), 2, replace=False)
        sample1 = X_minority.iloc[idx1].values
        sample2 = X_minority.iloc[idx2].values

        # Interpolate
        alpha = np.random.random()
        synthetic = sample1 + alpha * (sample2 - sample1)
        synthetic_samples.append(synthetic)

    # Add synthetic samples
    X_synthetic = pd.DataFrame(
        synthetic_samples,
        columns=X.columns
    )
    y_synthetic = np.ones(n_synthetic)

    X_augmented = pd.concat([X, X_synthetic], ignore_index=True)
    y_augmented = np.concatenate([y, y_synthetic])

    print(f"\nAugmented dataset:")
    print(f"  Total: {len(y_augmented)} samples")
    print(f"  Class 0: {(y_augmented == 0).sum()} samples")
    print(f"  Class 1: {(y_augmented == 1).sum()} samples")
    print(f"  Imbalance ratio: {(y_augmented == 0).sum() / (y_augmented == 1).sum():.2f}:1")


# Main function
def main():
    """
    Run all data pipeline examples
    Demonstrates different preprocessing approaches
    """
    print("\n" + "#" * 80)
    print("# DATA PIPELINE EXAMPLES")
    print("#" * 80)

    # Create output directories
    Path('data').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)

    try:
        # Example 1: Basic preprocessing
        example_basic_preprocessing()

        # Example 2: Feature engineering
        example_feature_engineering()

        # Example 3: Cross-validation pipeline
        example_cv_pipeline()

        # Example 4: Data augmentation
        example_data_augmentation()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
