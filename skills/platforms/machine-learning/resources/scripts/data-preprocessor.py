#!/usr/bin/env python3
"""
Data Preprocessing and Feature Engineering
Comprehensive data pipeline with cleaning, transformation, and augmentation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Tuple, List, Optional, Dict, Any
import logging
from pathlib import Path
import pickle


class DataPreprocessor:
    """
    Complete data preprocessing pipeline with:
    - Missing value imputation
    - Outlier detection and handling
    - Feature scaling and normalization
    - Feature engineering
    - Train/validation/test splitting
    - Data augmentation
    """

    def __init__(
        self,
        scaling_method: str = 'standard',
        imputation_strategy: str = 'mean',
        handle_outliers: bool = True,
        random_state: int = 42
    ):
        """
        Initialize preprocessor

        Args:
            scaling_method: 'standard', 'minmax', or 'robust'
            imputation_strategy: 'mean', 'median', 'most_frequent', or 'knn'
            handle_outliers: Whether to detect and handle outliers
            random_state: Random seed for reproducibility
        """
        self.scaling_method = scaling_method
        self.imputation_strategy = imputation_strategy
        self.handle_outliers = handle_outliers
        self.random_state = random_state

        self.scaler = None
        self.imputer = None
        self.label_encoders = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from various formats

        Args:
            filepath: Path to data file (.csv, .parquet, .json, .xlsx)

        Returns:
            DataFrame
        """
        path = Path(filepath)

        if path.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif path.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        elif path.suffix == '.json':
            df = pd.read_json(filepath)
        elif path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self.logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data quality and statistics

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'shape': df.shape,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'numeric_summary': df.describe().to_dict(),
        }

        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        analysis['categorical_columns'] = list(categorical_cols)
        analysis['numeric_columns'] = list(df.select_dtypes(include=[np.number]).columns)

        # Check for outliers in numeric columns
        if self.handle_outliers:
            outlier_counts = {}
            for col in analysis['numeric_columns']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_counts[col] = outliers
            analysis['outliers'] = outlier_counts

        self.logger.info("Data analysis complete")
        return analysis

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using specified imputation strategy

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with imputed values
        """
        df = df.copy()

        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # Impute numeric columns
        if len(numeric_cols) > 0 and df[numeric_cols].isnull().any().any():
            if self.imputation_strategy == 'knn':
                self.imputer = KNNImputer(n_neighbors=5)
            else:
                self.imputer = SimpleImputer(strategy=self.imputation_strategy)

            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
            self.logger.info(f"Imputed numeric columns using {self.imputation_strategy}")

        # Impute categorical columns
        if len(categorical_cols) > 0 and df[categorical_cols].isnull().any().any():
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            self.logger.info("Imputed categorical columns")

        return df

    def handle_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers using IQR method

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with outliers handled
        """
        if not self.handle_outliers:
            return df

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap outliers
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        self.logger.info("Handled outliers using IQR method")
        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))

        if len(categorical_cols) > 0:
            self.logger.info(f"Encoded {len(categorical_cols)} categorical columns")

        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using specified method

        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for train, False for test)

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return df

        if self.scaler is None or fit:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaling_method == 'robust':
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        self.logger.info(f"Scaled features using {self.scaling_method} scaler")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Example feature engineering operations
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Polynomial features for key numeric columns (limit to avoid explosion)
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # Limit to first 3 columns
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_log'] = np.log1p(np.abs(df[col]))

        # Interaction features
        if len(numeric_cols) >= 2:
            df['interaction_1_2'] = df[numeric_cols[0]] * df[numeric_cols[1]]

        self.logger.info("Engineered new features")
        return df

    def prepare_data(
        self,
        filepath: str,
        target: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete data preparation pipeline

        Args:
            filepath: Path to data file
            target: Target column name
            test_size: Fraction of data for test set
            val_size: Fraction of training data for validation set
            stratify: Whether to stratify splits

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Load data
        df = self.load_data(filepath)

        # Analyze data
        analysis = self.analyze_data(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Handle outliers
        df = self.handle_outliers_iqr(df)

        # Encode categorical variables
        df = self.encode_categorical(df)

        # Engineer features
        df = self.engineer_features(df)

        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]

        # Split data
        stratify_y = y if stratify else None

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_y
        )

        # Train/validation split
        stratify_y_train = y_train if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size),
            random_state=self.random_state, stratify=stratify_y_train
        )

        # Scale features
        X_train = self.scale_features(pd.DataFrame(X_train, columns=X.columns), fit=True)
        X_val = self.scale_features(pd.DataFrame(X_val, columns=X.columns), fit=False)
        X_test = self.scale_features(pd.DataFrame(X_test, columns=X.columns), fit=False)

        # Combine back with targets
        train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
        val_df = pd.concat([X_val, y_val.reset_index(drop=True)], axis=1)
        test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

        self.logger.info(f"Data preparation complete:")
        self.logger.info(f"  Train: {train_df.shape}")
        self.logger.info(f"  Val: {val_df.shape}")
        self.logger.info(f"  Test: {test_df.shape}")

        return train_df, val_df, test_df

    def save_preprocessor(self, filepath: str):
        """Save fitted preprocessor for later use"""
        state = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoders': self.label_encoders,
            'scaling_method': self.scaling_method,
            'imputation_strategy': self.imputation_strategy
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        self.logger.info(f"Saved preprocessor to: {filepath}")

    @classmethod
    def load_preprocessor(cls, filepath: str) -> 'DataPreprocessor':
        """Load fitted preprocessor"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        preprocessor = cls(
            scaling_method=state['scaling_method'],
            imputation_strategy=state['imputation_strategy']
        )
        preprocessor.scaler = state['scaler']
        preprocessor.imputer = state['imputer']
        preprocessor.label_encoders = state['label_encoders']

        return preprocessor


def main():
    """Example usage"""
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        scaling_method='standard',
        imputation_strategy='mean',
        handle_outliers=True
    )

    # Prepare data
    train_df, val_df, test_df = preprocessor.prepare_data(
        'data/raw/dataset.csv',
        target='label',
        test_size=0.2,
        val_size=0.1
    )

    # Save preprocessor
    preprocessor.save_preprocessor('models/preprocessor.pkl')

    print(f"Training set: {train_df.shape}")
    print(f"Validation set: {val_df.shape}")
    print(f"Test set: {test_df.shape}")


if __name__ == '__main__':
    main()
