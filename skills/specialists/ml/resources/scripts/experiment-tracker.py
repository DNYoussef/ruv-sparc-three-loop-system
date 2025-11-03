#!/usr/bin/env python3
"""
ML Experiment Tracker with MLflow/W&B Integration

Comprehensive experiment tracking system supporting:
- Auto-logging for TensorFlow, PyTorch, scikit-learn
- Custom metric and parameter tracking
- Artifact management (models, plots, data)
- Experiment comparison and analysis
- Integration with W&B and MLflow

Usage:
    python experiment-tracker.py --config experiment-config.yaml
    python experiment-tracker.py --list-experiments
    python experiment-tracker.py --compare exp1 exp2
"""

import argparse
import yaml
import json
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch
import wandb
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracking with MLflow and W&B support"""

    def __init__(self, config_path: str, backend: str = "mlflow"):
        """
        Initialize experiment tracker

        Args:
            config_path: Path to experiment configuration YAML
            backend: Tracking backend ('mlflow', 'wandb', or 'both')
        """
        self.config = self._load_config(config_path)
        self.backend = backend
        self.run_id = None

        # Initialize tracking backends
        if backend in ["mlflow", "both"]:
            self._init_mlflow()
        if backend in ["wandb", "both"]:
            self._init_wandb()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)

    def _init_mlflow(self):
        """Initialize MLflow tracking"""
        tracking_uri = self.config['experiment'].get('tracking_uri', './mlruns')
        mlflow.set_tracking_uri(tracking_uri)

        experiment_name = self.config['experiment']['name']
        mlflow.set_experiment(experiment_name)

        logger.info(f"Initialized MLflow: {tracking_uri}")

    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        project = self.config['experiment']['name']
        entity = self.config['experiment'].get('wandb_entity', None)

        wandb.init(
            project=project,
            entity=entity,
            config=self.config
        )

        logger.info(f"Initialized W&B project: {project}")

    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Start a new experiment run

        Args:
            run_name: Optional custom run name

        Returns:
            run_id: Unique identifier for this run
        """
        if not run_name:
            run_name = f"{self.config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self.backend in ["mlflow", "both"]:
            mlflow.start_run(run_name=run_name)
            self.run_id = mlflow.active_run().info.run_id
            logger.info(f"Started MLflow run: {self.run_id}")

        return self.run_id

    def log_params(self, params: Dict[str, Any]):
        """Log experiment parameters"""
        if self.backend in ["mlflow", "both"]:
            mlflow.log_params(params)

        if self.backend in ["wandb", "both"]:
            wandb.config.update(params)

        logger.info(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log experiment metrics"""
        if self.backend in ["mlflow", "both"]:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)

        if self.backend in ["wandb", "both"]:
            wandb.log(metrics, step=step)

        logger.info(f"Logged {len(metrics)} metrics at step {step}")

    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """
        Log artifact (model, plot, data file, etc.)

        Args:
            artifact_path: Path to artifact file
            artifact_type: Type of artifact ('model', 'plot', 'data', 'code')
        """
        if self.backend in ["mlflow", "both"]:
            if artifact_type == "model":
                mlflow.log_artifact(artifact_path, "models")
            else:
                mlflow.log_artifact(artifact_path, artifact_type)

        if self.backend in ["wandb", "both"]:
            wandb.save(artifact_path)

        logger.info(f"Logged artifact: {artifact_path} ({artifact_type})")

    def log_model(self, model, framework: str = "sklearn"):
        """
        Log trained model with auto-detection

        Args:
            model: Trained model object
            framework: ML framework ('sklearn', 'tensorflow', 'pytorch')
        """
        if self.backend in ["mlflow", "both"]:
            if framework == "sklearn":
                mlflow.sklearn.log_model(model, "model")
            elif framework == "tensorflow":
                mlflow.tensorflow.log_model(model, "model")
            elif framework == "pytorch":
                mlflow.pytorch.log_model(model, "model")

        logger.info(f"Logged {framework} model")

    def auto_log(self, framework: str = "sklearn"):
        """
        Enable auto-logging for supported frameworks

        Args:
            framework: ML framework to auto-log
        """
        if self.backend in ["mlflow", "both"]:
            if framework == "sklearn":
                mlflow.sklearn.autolog()
            elif framework == "tensorflow":
                mlflow.tensorflow.autolog()
            elif framework == "pytorch":
                mlflow.pytorch.autolog()

        logger.info(f"Enabled auto-logging for {framework}")

    def end_run(self, status: str = "FINISHED"):
        """
        End the current experiment run

        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED')
        """
        if self.backend in ["mlflow", "both"]:
            mlflow.end_run(status=status)

        if self.backend in ["wandb", "both"]:
            wandb.finish()

        logger.info(f"Ended run with status: {status}")

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiment runs

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Comparison results with metrics and parameters
        """
        if self.backend not in ["mlflow", "both"]:
            logger.warning("Comparison only supported for MLflow backend")
            return {}

        client = mlflow.tracking.MlflowClient()
        comparison = {}

        for run_id in run_ids:
            run = client.get_run(run_id)
            comparison[run_id] = {
                'params': run.data.params,
                'metrics': run.data.metrics,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'status': run.info.status
            }

        logger.info(f"Compared {len(run_ids)} runs")
        return comparison


def list_experiments():
    """List all experiments in MLflow tracking server"""
    client = mlflow.tracking.MlflowClient()
    experiments = client.list_experiments()

    print("\n=== Available Experiments ===")
    for exp in experiments:
        print(f"Name: {exp.name}")
        print(f"  ID: {exp.experiment_id}")
        print(f"  Artifact Location: {exp.artifact_location}")
        print(f"  Lifecycle Stage: {exp.lifecycle_stage}")
        print()


def main():
    """Main entry point for experiment tracker CLI"""
    parser = argparse.ArgumentParser(description="ML Experiment Tracker")
    parser.add_argument('--config', type=str, help='Path to experiment config YAML')
    parser.add_argument('--backend', type=str, default='mlflow',
                       choices=['mlflow', 'wandb', 'both'],
                       help='Tracking backend')
    parser.add_argument('--list-experiments', action='store_true',
                       help='List all experiments')
    parser.add_argument('--compare', nargs='+', help='Compare multiple run IDs')
    parser.add_argument('--run-name', type=str, help='Custom run name')

    args = parser.parse_args()

    if args.list_experiments:
        list_experiments()
        return

    if not args.config:
        parser.error("--config is required unless using --list-experiments")

    # Initialize tracker
    tracker = ExperimentTracker(args.config, backend=args.backend)

    if args.compare:
        results = tracker.compare_runs(args.compare)
        print("\n=== Run Comparison ===")
        print(json.dumps(results, indent=2))
    else:
        # Example workflow
        tracker.start_run(run_name=args.run_name)

        # Log parameters from config
        params = tracker.config.get('training', {})
        tracker.log_params(params)

        # Enable auto-logging
        tracker.auto_log(framework='sklearn')

        # Example metrics
        tracker.log_metrics({
            'accuracy': 0.95,
            'precision': 0.93,
            'recall': 0.94,
            'f1_score': 0.935
        }, step=0)

        tracker.end_run(status='FINISHED')

        print(f"\n✅ Experiment tracking completed. Run ID: {tracker.run_id}")


if __name__ == "__main__":
    main()
