#!/usr/bin/env node
/**
 * Hyperparameter Optimization Example (250 lines)
 *
 * Demonstrates Optuna-based distributed hyperparameter tuning with:
 * - Bayesian optimization
 * - Pruning strategies
 * - Parallel trials
 * - Result visualization
 *
 * This example shows how to:
 * 1. Define search space for hyperparameters
 * 2. Set up optimization study with Optuna
 * 3. Run distributed parallel trials
 * 4. Implement early stopping (pruning)
 * 5. Analyze and visualize results
 */

const fs = require('fs').promises;
const { spawn } = require('child_process');
const path = require('path');

/**
 * Hyperparameter Optimization Demo
 */
class HyperparameterOptimizationDemo {
    constructor() {
        this.studyName = `hp_optimization_demo_${Date.now()}`;
        this.storageUrl = 'sqlite:///hp_optimization_demo.db';
        this.nTrials = 50;
        this.nJobs = 4; // Parallel jobs
    }

    /**
     * Define hyperparameter search space
     */
    defineSearchSpace() {
        return {
            // Neural network architecture
            n_layers: {
                type: 'int',
                low: 2,
                high: 5,
                description: 'Number of hidden layers'
            },
            hidden_units: {
                type: 'categorical',
                choices: [64, 128, 256, 512],
                description: 'Units per hidden layer'
            },

            // Optimization
            learning_rate: {
                type: 'float',
                low: 1e-5,
                high: 1e-1,
                log: true,
                description: 'Learning rate (log scale)'
            },
            optimizer: {
                type: 'categorical',
                choices: ['adam', 'sgd', 'rmsprop'],
                description: 'Optimization algorithm'
            },

            // Regularization
            dropout: {
                type: 'float',
                low: 0.1,
                high: 0.5,
                step: 0.05,
                description: 'Dropout rate'
            },
            l2_reg: {
                type: 'float',
                low: 1e-6,
                high: 1e-3,
                log: true,
                description: 'L2 regularization (log scale)'
            },

            // Training
            batch_size: {
                type: 'categorical',
                choices: [16, 32, 64, 128],
                description: 'Training batch size'
            },
            activation: {
                type: 'categorical',
                choices: ['relu', 'tanh', 'elu'],
                description: 'Activation function'
            }
        };
    }

    /**
     * Generate Python script for Optuna optimization
     */
    generateOptimizationScript() {
        return `
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_model(trial):
    """Create neural network model based on trial suggestions"""
    # Suggest hyperparameters
    n_layers = trial.suggest_int('n_layers', 2, 5)
    hidden_units = trial.suggest_categorical('hidden_units', [64, 128, 256, 512])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'elu'])

    # Build model
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.Dense(
        hidden_units,
        activation=activation,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        input_shape=(20,)
    ))
    model.add(keras.layers.Dropout(dropout))

    # Hidden layers
    for _ in range(n_layers - 1):
        model.add(keras.layers.Dense(
            hidden_units,
            activation=activation,
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        ))
        model.add(keras.layers.Dropout(dropout))

    # Output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Configure optimizer
    optimizers = {
        'adam': keras.optimizers.Adam(learning_rate=learning_rate),
        'sgd': keras.optimizers.SGD(learning_rate=learning_rate),
        'rmsprop': keras.optimizers.RMSprop(learning_rate=learning_rate)
    }

    model.compile(
        optimizer=optimizers[optimizer_name],
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model, batch_size

def objective(trial):
    """Objective function for optimization"""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Create model
    model, batch_size = create_model(trial)

    # Pruning callback
    class OptunaPruningCallback(keras.callbacks.Callback):
        def __init__(self, trial):
            super().__init__()
            self.trial = trial

        def on_epoch_end(self, epoch, logs=None):
            # Report intermediate value
            accuracy = logs['val_accuracy']
            self.trial.report(accuracy, epoch)

            # Prune if necessary
            if self.trial.should_prune():
                self.model.stop_training = True

    # Train model
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=batch_size,
            callbacks=[OptunaPruningCallback(trial)],
            verbose=0
        )

        # Return final validation accuracy
        return history.history['val_accuracy'][-1]

    except optuna.exceptions.TrialPruned:
        raise

# Create study
print(f"Creating Optuna study: ${this.studyName}")
study = optuna.create_study(
    study_name='${this.studyName}',
    storage='${this.storageUrl}',
    direction='maximize',
    sampler=TPESampler(),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    load_if_exists=True
)

# Run optimization
print(f"Running optimization: ${this.nTrials} trials, ${this.nJobs} jobs")
study.optimize(
    objective,
    n_trials=${this.nTrials},
    n_jobs=${this.nJobs},
    show_progress_bar=True
)

# Print results
print("\\n" + "="*70)
print("Optimization Results")
print("="*70)
print(f"Number of finished trials: {len(study.trials)}")
print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

print(f"\\nBest trial: {study.best_trial.number}")
print(f"Best value (accuracy): {study.best_value:.4f}")

print("\\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Save results
results = {
    'study_name': study.study_name,
    'n_trials': len(study.trials),
    'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    'best_trial': study.best_trial.number,
    'best_value': study.best_value,
    'best_params': study.best_params
}

with open('${this.studyName}_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\\n✓ Results saved to ${this.studyName}_results.json")

# Generate visualizations
print("\\nGenerating visualizations...")

try:
    from optuna.visualization import plot_optimization_history
    from optuna.visualization import plot_param_importances
    from optuna.visualization import plot_parallel_coordinate

    # Optimization history
    fig = plot_optimization_history(study)
    fig.write_html('${this.studyName}_history.html')
    print("  ✓ Optimization history: ${this.studyName}_history.html")

    # Parameter importance
    fig = plot_param_importances(study)
    fig.write_html('${this.studyName}_importance.html')
    print("  ✓ Parameter importance: ${this.studyName}_importance.html")

    # Parallel coordinate
    fig = plot_parallel_coordinate(study)
    fig.write_html('${this.studyName}_parallel.html')
    print("  ✓ Parallel coordinate: ${this.studyName}_parallel.html")

except ImportError:
    print("  ⚠ Visualization libraries not available (install plotly)")

print("\\n" + "="*70)
`;
    }

    /**
     * Run optimization
     */
    async runOptimization() {
        console.log('\n' + '='.repeat(70));
        console.log('Hyperparameter Optimization with Optuna');
        console.log('='.repeat(70) + '\n');

        console.log(`Study: ${this.studyName}`);
        console.log(`Trials: ${this.nTrials}`);
        console.log(`Parallel Jobs: ${this.nJobs}`);
        console.log(`Storage: ${this.storageUrl}\n`);

        // Generate and save optimization script
        const script = this.generateOptimizationScript();
        const scriptPath = `optuna_demo_${this.studyName}.py`;

        await fs.writeFile(scriptPath, script);
        console.log(`✓ Generated optimization script: ${scriptPath}\n`);

        // Run optimization
        console.log('Starting optimization...\n');

        return new Promise((resolve, reject) => {
            const python = spawn('python', [scriptPath]);

            python.stdout.on('data', (data) => {
                process.stdout.write(data.toString());
            });

            python.stderr.on('data', (data) => {
                process.stderr.write(data.toString());
            });

            python.on('close', (code) => {
                if (code === 0) {
                    console.log('\n✅ Optimization completed successfully\n');
                    resolve();
                } else {
                    console.error(`\n❌ Optimization failed with code ${code}\n`);
                    reject(new Error(`Process exited with code ${code}`));
                }
            });
        });
    }

    /**
     * Analyze results
     */
    async analyzeResults() {
        const resultsPath = `${this.studyName}_results.json`;

        try {
            const data = await fs.readFile(resultsPath, 'utf8');
            const results = JSON.parse(data);

            console.log('\n' + '='.repeat(70));
            console.log('Analysis Summary');
            console.log('='.repeat(70) + '\n');

            console.log(`Total trials: ${results.n_trials}`);
            console.log(`Complete trials: ${results.n_complete}`);
            console.log(`Pruned trials: ${results.n_pruned}`);
            console.log(`Pruning rate: ${(results.n_pruned / results.n_trials * 100).toFixed(1)}%`);

            console.log(`\nBest trial: #${results.best_trial}`);
            console.log(`Best accuracy: ${(results.best_value * 100).toFixed(2)}%`);

            console.log('\nOptimal hyperparameters:');
            for (const [param, value] of Object.entries(results.best_params)) {
                console.log(`  ${param}: ${value}`);
            }

            console.log('\n' + '='.repeat(70) + '\n');

            return results;
        } catch (error) {
            console.error(`Failed to analyze results: ${error.message}`);
            throw error;
        }
    }
}

/**
 * Main demonstration function
 */
async function main() {
    const demo = new HyperparameterOptimizationDemo();

    try {
        // Run optimization
        await demo.runOptimization();

        // Analyze results
        await demo.analyzeResults();

        console.log('📊 Visualization files generated:');
        console.log(`  - ${demo.studyName}_history.html`);
        console.log(`  - ${demo.studyName}_importance.html`);
        console.log(`  - ${demo.studyName}_parallel.html\n`);

    } catch (error) {
        console.error('Error:', error.message);
        process.exit(1);
    }
}

// Run if executed directly
if (require.main === module) {
    main();
}

module.exports = { HyperparameterOptimizationDemo };
