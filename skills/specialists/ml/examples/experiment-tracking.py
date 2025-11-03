#!/usr/bin/env python3
"""
Experiment Tracking Example (150 lines)

Demonstrates MLflow integration with:
- Auto-logging for multiple frameworks
- Custom metric tracking
- Artifact management
- Experiment comparison

This example shows how to:
1. Set up experiment tracking with MLflow
2. Track parameters and metrics during training
3. Log model artifacts and plots
4. Compare multiple experiment runs
5. Retrieve and analyze results
"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentTrackingDemo:
    """Demonstration of MLflow experiment tracking"""

    def __init__(self, experiment_name="ml-tracking-demo"):
        """Initialize experiment tracking"""
        self.experiment_name = experiment_name

        # Set tracking URI (use local file system)
        mlflow.set_tracking_uri("file:./mlruns")

        # Create or get experiment
        mlflow.set_experiment(experiment_name)

        logger.info(f"Initialized experiment: {experiment_name}")

    def generate_synthetic_data(self, n_samples=1000, n_features=20):
        """
        Generate synthetic classification dataset

        Args:
            n_samples: Number of samples
            n_features: Number of features

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Generating synthetic data: {n_samples} samples, {n_features} features")

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def run_experiment(self, params, run_name=None):
        """
        Run a single experiment with given parameters

        Args:
            params: Dictionary of model hyperparameters
            run_name: Optional custom run name

        Returns:
            Model performance metrics
        """
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"Starting run: {run.info.run_id}")

            # Log parameters
            mlflow.log_params(params)

            # Generate data
            X_train, X_test, y_train, y_test = self.generate_synthetic_data()

            # Log data characteristics
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])

            # Train model
            logger.info("Training model...")
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'precision': precision_score(y_test, y_pred_test),
                'recall': recall_score(y_test, y_pred_test),
                'f1_score': f1_score(y_test, y_pred_test)
            }

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(model, "random_forest_model")

            # Create and log feature importance plot
            self._log_feature_importance(model, X_train.shape[1], run.info.run_id)

            # Create and log confusion matrix
            self._log_confusion_matrix(y_test, y_pred_test, run.info.run_id)

            logger.info(f"Run completed. Test Accuracy: {metrics['test_accuracy']:.4f}")

            return run.info.run_id, metrics

    def _log_feature_importance(self, model, n_features, run_id):
        """Create and log feature importance visualization"""
        importance = model.feature_importances_
        features = [f"Feature_{i}" for i in range(n_features)]

        plt.figure(figsize=(10, 6))
        plt.barh(features[:10], importance[:10])  # Top 10 features
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()

        plot_path = f"feature_importance_{run_id}.png"
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path, "plots")
        Path(plot_path).unlink()  # Clean up

    def _log_confusion_matrix(self, y_true, y_pred, run_id):
        """Create and log confusion matrix visualization"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        plot_path = f"confusion_matrix_{run_id}.png"
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path, "plots")
        Path(plot_path).unlink()

    def run_hyperparameter_search(self):
        """
        Run multiple experiments with different hyperparameters
        for comparison
        """
        logger.info("Running hyperparameter search with experiment tracking...")

        param_configs = [
            {
                'n_estimators': 50,
                'max_depth': 10,
                'min_samples_split': 2
            },
            {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5
            },
            {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 10
            }
        ]

        results = []
        for i, params in enumerate(param_configs):
            run_id, metrics = self.run_experiment(
                params,
                run_name=f"hp_search_run_{i+1}"
            )
            results.append({
                'run_id': run_id,
                'params': params,
                'metrics': metrics
            })

        return results

    def compare_runs(self, results):
        """
        Compare multiple experiment runs

        Args:
            results: List of run results with IDs and metrics
        """
        logger.info("\n" + "="*60)
        logger.info("Experiment Comparison")
        logger.info("="*60)

        for i, result in enumerate(results, 1):
            logger.info(f"\nRun {i}:")
            logger.info(f"  Run ID: {result['run_id']}")
            logger.info(f"  Parameters:")
            for param, value in result['params'].items():
                logger.info(f"    {param}: {value}")
            logger.info(f"  Metrics:")
            for metric, value in result['metrics'].items():
                logger.info(f"    {metric}: {value:.4f}")

        # Find best run
        best_run = max(results, key=lambda x: x['metrics']['test_accuracy'])
        logger.info(f"\n{'='*60}")
        logger.info(f"Best Run: {best_run['run_id']}")
        logger.info(f"Test Accuracy: {best_run['metrics']['test_accuracy']:.4f}")
        logger.info(f"{'='*60}\n")


def main():
    """Main demonstration function"""
    print("\n" + "="*60)
    print("MLflow Experiment Tracking Demonstration")
    print("="*60 + "\n")

    # Initialize demo
    demo = ExperimentTrackingDemo(experiment_name="ml-tracking-demo")

    # Run hyperparameter search with tracking
    results = demo.run_hyperparameter_search()

    # Compare runs
    demo.compare_runs(results)

    print("\n" + "="*60)
    print("Experiment tracking completed!")
    print("\nView results with: mlflow ui")
    print("Then navigate to: http://localhost:5000")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
