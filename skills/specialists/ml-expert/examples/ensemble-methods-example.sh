#!/bin/bash
##############################################################################
# Ensemble Methods Complete Example
#
# Demonstrates production-ready ensemble learning techniques including:
# - Voting ensembles (hard and soft voting)
# - Bagging (Bootstrap Aggregating) with Random Forests
# - Boosting (AdaBoost, Gradient Boosting, XGBoost)
# - Stacking with meta-learners
# - Ensemble optimization and hyperparameter tuning
# - Cross-validation for ensemble evaluation
# - Feature importance from ensemble models
#
# This example shows best practices for building high-performance
# ensemble models for classification and regression tasks.
##############################################################################

set -euo pipefail

# Configuration
PYTHON="${PYTHON:-python3}"
WORK_DIR="${WORK_DIR:-./ensemble_demo}"
VERBOSE="${VERBOSE:-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${CYAN}=== $1 ===${NC}\n"; }

##############################################################################
# Create comprehensive ensemble demonstration
##############################################################################

create_ensemble_demo() {
    local demo_script="$WORK_DIR/ensemble_demo.py"

    mkdir -p "$WORK_DIR"

    cat > "$demo_script" << 'PYTHON_DEMO'
"""
Comprehensive Ensemble Methods Demonstration

Shows production-ready ensemble techniques with real examples.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pickle
import json


class EnsembleDemo:
    """Comprehensive ensemble methods demonstration."""

    def __init__(self, n_samples=1000, n_features=20):
        self.n_samples = n_samples
        self.n_features = n_features
        self.results = {}

    def generate_data(self):
        """Generate synthetic classification dataset."""
        print("Generating synthetic dataset...")

        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Features: {self.n_features}\n")

    def demo_voting_ensemble(self):
        """Demonstrate voting ensemble."""
        print("\n--- Voting Ensemble ---")

        # Create base models
        models = [
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('dt', DecisionTreeClassifier(max_depth=5, random_state=42))
        ]

        # Hard voting
        hard_voting = VotingClassifier(estimators=models, voting='hard')
        hard_voting.fit(self.X_train, self.y_train)
        hard_acc = accuracy_score(self.y_test, hard_voting.predict(self.X_test))

        # Soft voting
        soft_voting = VotingClassifier(estimators=models, voting='soft')
        soft_voting.fit(self.X_train, self.y_train)
        soft_acc = accuracy_score(self.y_test, soft_voting.predict(self.X_test))

        print(f"Hard Voting Accuracy: {hard_acc:.4f}")
        print(f"Soft Voting Accuracy: {soft_acc:.4f}")

        self.results['voting'] = {
            'hard_accuracy': hard_acc,
            'soft_accuracy': soft_acc
        }

        return soft_voting

    def demo_bagging_ensemble(self):
        """Demonstrate bagging ensemble."""
        print("\n--- Bagging Ensemble ---")

        # Bagging with decision trees
        bagging = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=50,
            max_samples=0.8,
            max_features=0.8,
            random_state=42
        )

        bagging.fit(self.X_train, self.y_train)
        y_pred = bagging.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        print(f"Bagging Accuracy: {accuracy:.4f}")
        print(f"Bagging F1 Score: {f1:.4f}")

        # Random Forest (advanced bagging)
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )

        rf.fit(self.X_train, self.y_train)
        rf_acc = accuracy_score(self.y_test, rf.predict(self.X_test))

        print(f"Random Forest Accuracy: {rf_acc:.4f}")

        self.results['bagging'] = {
            'bagging_accuracy': accuracy,
            'bagging_f1': f1,
            'rf_accuracy': rf_acc
        }

        return rf

    def demo_boosting_ensemble(self):
        """Demonstrate boosting ensemble."""
        print("\n--- Boosting Ensemble ---")

        # AdaBoost
        adaboost = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        )

        adaboost.fit(self.X_train, self.y_train)
        ada_acc = accuracy_score(self.y_test, adaboost.predict(self.X_test))

        print(f"AdaBoost Accuracy: {ada_acc:.4f}")

        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        gb.fit(self.X_train, self.y_train)
        gb_acc = accuracy_score(self.y_test, gb.predict(self.X_test))

        print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

        self.results['boosting'] = {
            'adaboost_accuracy': ada_acc,
            'gradient_boosting_accuracy': gb_acc
        }

        return gb

    def demo_stacking_ensemble(self):
        """Demonstrate stacking ensemble."""
        print("\n--- Stacking Ensemble ---")

        # Base models
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('dt', DecisionTreeClassifier(max_depth=5, random_state=42))
        ]

        # Meta-learner
        meta_learner = LogisticRegression(max_iter=1000)

        # Stacking classifier
        stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5
        )

        stacking.fit(self.X_train, self.y_train)
        y_pred = stacking.predict(self.X_test)
        y_proba = stacking.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba)

        print(f"Stacking Accuracy: {accuracy:.4f}")
        print(f"Stacking F1 Score: {f1:.4f}")
        print(f"Stacking AUC-ROC: {auc:.4f}")

        self.results['stacking'] = {
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc
        }

        return stacking

    def optimize_ensemble(self, model_type='rf'):
        """Demonstrate hyperparameter optimization."""
        print(f"\n--- Optimizing {model_type.upper()} Ensemble ---")

        if model_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5]
            }
            model = RandomForestClassifier(random_state=42)

        elif model_type == 'gb':
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
            model = GradientBoostingClassifier(random_state=42)

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        best_acc = accuracy_score(
            self.y_test,
            grid_search.best_estimator_.predict(self.X_test)
        )
        print(f"Test accuracy: {best_acc:.4f}")

        self.results[f'{model_type}_optimized'] = {
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_accuracy': best_acc
        }

        return grid_search.best_estimator_

    def compare_all_ensembles(self):
        """Compare all ensemble methods."""
        print("\n" + "="*60)
        print("ENSEMBLE COMPARISON")
        print("="*60 + "\n")

        models = {
            'Voting (Soft)': VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(max_iter=1000)),
                    ('rf', RandomForestClassifier(n_estimators=50)),
                    ('dt', DecisionTreeClassifier(max_depth=5))
                ],
                voting='soft'
            ),
            'Bagging': BaggingClassifier(
                base_estimator=DecisionTreeClassifier(),
                n_estimators=50
            ),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'AdaBoost': AdaBoostClassifier(n_estimators=50),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
            'Stacking': StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=50)),
                    ('gb', GradientBoostingClassifier(n_estimators=50))
                ],
                final_estimator=LogisticRegression(max_iter=1000)
            )
        }

        comparison = []

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)

            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            test_acc = accuracy_score(self.y_test, model.predict(self.X_test))

            comparison.append({
                'model': name,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_acc
            })

        # Print results table
        df = pd.DataFrame(comparison)
        df = df.sort_values('test_accuracy', ascending=False)

        print("\n" + str(df.to_string(index=False)))

        self.results['comparison'] = df.to_dict('records')

    def save_results(self, output_dir='results'):
        """Save all results and best model."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save results JSON
        with open(f'{output_dir}/ensemble_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nResults saved to {output_dir}/ensemble_results.json")


def main():
    print("="*60)
    print("ENSEMBLE METHODS COMPREHENSIVE DEMONSTRATION")
    print("="*60)

    demo = EnsembleDemo(n_samples=2000, n_features=20)

    # Generate data
    demo.generate_data()

    # Run all demonstrations
    demo.demo_voting_ensemble()
    demo.demo_bagging_ensemble()
    demo.demo_boosting_ensemble()
    demo.demo_stacking_ensemble()

    # Optimize best model
    demo.optimize_ensemble('rf')

    # Compare all methods
    demo.compare_all_ensembles()

    # Save results
    demo.save_results()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
PYTHON_DEMO

    echo "$demo_script"
}

##############################################################################
# Main execution
##############################################################################

main() {
    log_section "Ensemble Methods Complete Example"

    # Check Python availability
    if ! command -v "$PYTHON" &> /dev/null; then
        log_error "Python not found. Please install Python 3."
        exit 1
    fi

    # Create demonstration script
    log_info "Creating ensemble demonstration..."
    demo_script=$(create_ensemble_demo)

    # Check for required packages
    log_info "Checking Python dependencies..."
    $PYTHON -c "import sklearn, numpy, pandas" 2>/dev/null || {
        log_warning "Missing dependencies. Installing scikit-learn, numpy, pandas..."
        $PYTHON -m pip install scikit-learn numpy pandas 2>/dev/null || {
            log_error "Failed to install dependencies. Please install manually:"
            log_error "  pip install scikit-learn numpy pandas"
            exit 1
        }
    }

    # Run demonstration
    log_section "Running Ensemble Demonstration"
    $PYTHON "$demo_script"

    log_section "Results"
    if [ -f "$WORK_DIR/results/ensemble_results.json" ]; then
        log_success "Results saved to $WORK_DIR/results/"
        log_info "View results:"
        echo "  cat $WORK_DIR/results/ensemble_results.json | ${PYTHON} -m json.tool"
    fi

    log_success "Ensemble methods demonstration complete!"
}

# Run if executed directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
