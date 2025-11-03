#!/bin/bash
##############################################################################
# Ensemble Model Construction and Validation
#
# Automates the creation, training, and validation of ensemble machine
# learning models including bagging, boosting, and stacking strategies.
#
# Usage:
#   ./ensemble-builder.sh --config ensemble-strategy.yaml --data train.csv --output ensemble_model/
#   ./ensemble-builder.sh --validate ensemble_model/ --test test.csv --metrics metrics.json
#   ./ensemble-builder.sh --optimize ensemble_model/ --method grid_search --target accuracy
##############################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-./ensemble_workspace}"
PYTHON="${PYTHON:-python3}"
VERBOSE="${VERBOSE:-0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --data)
                DATA_FILE="$2"
                shift 2
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --validate)
                VALIDATE_DIR="$2"
                shift 2
                ;;
            --test)
                TEST_FILE="$2"
                shift 2
                ;;
            --metrics)
                METRICS_FILE="$2"
                shift 2
                ;;
            --optimize)
                OPTIMIZE_DIR="$2"
                shift 2
                ;;
            --method)
                OPTIMIZATION_METHOD="$2"
                shift 2
                ;;
            --target)
                TARGET_METRIC="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=1
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Show help message
show_help() {
    cat << EOF
Ensemble Model Builder - Automated ensemble ML pipeline

Usage:
    $0 [OPTIONS]

Build Options:
    --config FILE       Ensemble strategy configuration (YAML)
    --data FILE         Training data (CSV)
    --output DIR        Output directory for ensemble model

Validation Options:
    --validate DIR      Ensemble model directory to validate
    --test FILE         Test data (CSV)
    --metrics FILE      Output metrics (JSON)

Optimization Options:
    --optimize DIR      Ensemble model directory to optimize
    --method METHOD     Optimization method (grid_search, random_search, bayesian)
    --target METRIC     Target metric (accuracy, f1, auc)

General Options:
    --verbose           Enable verbose output
    --help              Show this help message

Examples:
    # Build ensemble from config
    $0 --config ensemble.yaml --data train.csv --output models/

    # Validate ensemble
    $0 --validate models/ --test test.csv --metrics results.json

    # Optimize ensemble
    $0 --optimize models/ --method grid_search --target f1
EOF
}

# Create Python ensemble builder script
create_ensemble_builder() {
    local output_script="$WORK_DIR/build_ensemble.py"

    cat > "$output_script" << 'PYTHON_SCRIPT'
import yaml
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

class EnsembleBuilder:
    """Build and manage ensemble ML models."""

    ENSEMBLE_METHODS = {
        'voting': VotingClassifier,
        'stacking': StackingClassifier,
        'bagging': RandomForestClassifier,  # RF is bagging
        'boosting': GradientBoostingClassifier,
        'adaboost': AdaBoostClassifier
    }

    BASE_MODELS = {
        'random_forest': RandomForestClassifier,
        'decision_tree': DecisionTreeClassifier,
        'logistic_regression': LogisticRegression,
        'gradient_boosting': GradientBoostingClassifier
    }

    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.ensemble = None
        self.base_models = []

    def build_base_models(self) -> List:
        """Build base models from configuration."""
        models = []

        for model_config in self.config.get('base_models', []):
            model_type = model_config['type']
            model_params = model_config.get('params', {})

            model_class = self.BASE_MODELS.get(model_type)
            if not model_class:
                raise ValueError(f"Unknown model type: {model_type}")

            model = model_class(**model_params)
            models.append((model_config.get('name', model_type), model))

        self.base_models = models
        return models

    def build_ensemble(self):
        """Build ensemble model from base models."""
        ensemble_type = self.config['ensemble']['type']
        ensemble_params = self.config['ensemble'].get('params', {})

        if ensemble_type == 'voting':
            self.ensemble = VotingClassifier(
                estimators=self.base_models,
                **ensemble_params
            )
        elif ensemble_type == 'stacking':
            # Use logistic regression as meta-classifier by default
            meta_classifier = LogisticRegression()
            self.ensemble = StackingClassifier(
                estimators=self.base_models,
                final_estimator=meta_classifier,
                **ensemble_params
            )
        elif ensemble_type in ['bagging', 'boosting']:
            # These are single ensemble methods
            ensemble_class = self.ENSEMBLE_METHODS[ensemble_type]
            self.ensemble = ensemble_class(**ensemble_params)

        return self.ensemble

    def train(self, X, y):
        """Train ensemble model."""
        if self.ensemble is None:
            self.build_base_models()
            self.build_ensemble()

        print(f"Training ensemble with {len(X)} samples...")
        self.ensemble.fit(X, y)

        # Cross-validation score
        cv_scores = cross_val_score(self.ensemble, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return self.ensemble

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate ensemble on test data."""
        if self.ensemble is None:
            raise ValueError("Ensemble not trained")

        y_pred = self.ensemble.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

        # AUC for binary classification
        if len(np.unique(y_test)) == 2:
            y_proba = self.ensemble.predict_proba(X_test)[:, 1]
            metrics['auc'] = roc_auc_score(y_test, y_proba)

        return metrics

    def save(self, output_dir: Path):
        """Save ensemble model and metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / 'ensemble_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.ensemble, f)

        # Save config
        config_path = output_dir / 'ensemble_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

        print(f"Ensemble saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: Path):
        """Load ensemble model."""
        model_path = model_dir / 'ensemble_model.pkl'
        config_path = model_dir / 'ensemble_config.yaml'

        with open(model_path, 'rb') as f:
            ensemble = pickle.load(f)

        # Create instance with config
        instance = cls(config_path)
        instance.ensemble = ensemble

        return instance

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 4:
        print("Usage: python build_ensemble.py <config> <data> <output>")
        sys.exit(1)

    config_file = Path(sys.argv[1])
    data_file = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])

    # Load data
    df = pd.read_csv(data_file)
    X = df.drop('target', axis=1)
    y = df['target']

    # Build and train ensemble
    builder = EnsembleBuilder(config_file)
    builder.build_base_models()
    builder.build_ensemble()
    builder.train(X, y)
    builder.save(output_dir)
PYTHON_SCRIPT

    echo "$output_script"
}

# Build ensemble model
build_ensemble() {
    log_info "Building ensemble model..."

    # Create work directory
    mkdir -p "$WORK_DIR"

    # Create ensemble builder script
    local builder_script
    builder_script=$(create_ensemble_builder)

    # Run ensemble builder
    $PYTHON "$builder_script" "$CONFIG_FILE" "$DATA_FILE" "$OUTPUT_DIR"

    log_success "Ensemble built successfully: $OUTPUT_DIR"
}

# Validate ensemble model
validate_ensemble() {
    log_info "Validating ensemble model..."

    local validation_script="$WORK_DIR/validate_ensemble.py"

    cat > "$validation_script" << 'PYTHON_SCRIPT'
import sys
import json
import pandas as pd
from pathlib import Path
from build_ensemble import EnsembleBuilder

if __name__ == '__main__':
    model_dir = Path(sys.argv[1])
    test_file = Path(sys.argv[2])
    metrics_file = Path(sys.argv[3])

    # Load model
    ensemble = EnsembleBuilder.load(model_dir)

    # Load test data
    df = pd.read_csv(test_file)
    X_test = df.drop('target', axis=1)
    y_test = df['target']

    # Evaluate
    metrics = ensemble.evaluate(X_test, y_test)

    # Save metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Validation metrics: {metrics}")
PYTHON_SCRIPT

    $PYTHON "$validation_script" "$VALIDATE_DIR" "$TEST_FILE" "$METRICS_FILE"

    log_success "Validation complete: $METRICS_FILE"
}

# Main execution
main() {
    parse_args "$@"

    if [[ -n "${CONFIG_FILE:-}" && -n "${DATA_FILE:-}" && -n "${OUTPUT_DIR:-}" ]]; then
        build_ensemble
    elif [[ -n "${VALIDATE_DIR:-}" && -n "${TEST_FILE:-}" && -n "${METRICS_FILE:-}" ]]; then
        validate_ensemble
    else
        log_error "Invalid arguments"
        show_help
        exit 1
    fi
}

main "$@"
