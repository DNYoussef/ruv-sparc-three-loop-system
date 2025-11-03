# Machine Learning Development Skill (Gold Tier)

Complete machine learning workflow from data preprocessing to production deployment.

## Quick Start

```python
# Train a model
from resources.scripts.model_trainer import ModelTrainer

trainer = ModelTrainer('config/training-config.yaml')
model = trainer.train()
trainer.evaluate(model)
trainer.save(model, 'models/my_model.pth')
```

## Directory Structure

```
machine-learning/
├── skill.md                          # Skill metadata and documentation
├── README.md                         # This file
├── resources/
│   ├── scripts/
│   │   ├── model-trainer.py          # Training pipeline
│   │   ├── data-preprocessor.py      # Data preprocessing
│   │   ├── model-evaluator.js        # Evaluation framework
│   │   └── ml-pipeline.sh            # End-to-end pipeline
│   └── templates/
│       ├── training-config.yaml      # Training configuration
│       ├── model-architecture.json   # Model architectures
│       └── evaluation-metrics.yaml   # Evaluation metrics
├── tests/
│   ├── test_trainer.py               # Training tests
│   ├── test_preprocessor.py          # Preprocessing tests
│   └── test_evaluator.py             # Evaluation tests
├── examples/
│   ├── model-training.py             # Training example
│   ├── data-pipeline.py              # Data pipeline example
│   └── model-deployment.py           # Deployment example
└── when-*/                           # Sub-skills
```

## Features

### Data Processing
- Automated data cleaning and validation
- Feature engineering and selection
- Data augmentation strategies
- Train/val/test splitting with stratification
- Data versioning with DVC integration

### Model Training
- PyTorch and TensorFlow support
- Distributed training (DDP, Horovod)
- Hyperparameter optimization (Optuna, Ray Tune)
- Transfer learning from pretrained models
- Training monitoring with TensorBoard/Weights & Biases

### Model Evaluation
- Multi-metric evaluation (accuracy, F1, ROC-AUC, etc.)
- Cross-validation and bootstrapping
- Confusion matrices and classification reports
- Fairness metrics (demographic parity, equalized odds)
- Model interpretability (SHAP, LIME)

### Model Deployment
- Model serialization (ONNX, TorchScript)
- REST API with FastAPI
- Docker containerization
- Kubernetes deployment manifests
- Monitoring and logging
- A/B testing framework

## Usage Examples

### 1. Train a Classification Model

```python
from resources.scripts.model_trainer import ModelTrainer
from resources.scripts.data_preprocessor import DataPreprocessor

# Preprocess data
preprocessor = DataPreprocessor()
train_data, val_data, test_data = preprocessor.prepare_data(
    'data/raw/dataset.csv',
    target='label',
    test_size=0.2
)

# Train model
trainer = ModelTrainer('resources/templates/training-config.yaml')
model = trainer.train(train_data, val_data)

# Evaluate
metrics = trainer.evaluate(model, test_data)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
```

### 2. Deploy Model as API

```python
from examples.model_deployment import create_app

# Load trained model
model = load_model('models/my_model.pth')

# Create FastAPI app
app = create_app(model)

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. Run Complete Pipeline

```bash
# End-to-end ML pipeline
bash resources/scripts/ml-pipeline.sh \
  --data data/raw/dataset.csv \
  --config resources/templates/training-config.yaml \
  --output models/
```

## Integration with SPARC System

### Agent Coordination

```javascript
// Multi-agent ML workflow
Task("Data Scientist", "Analyze dataset and engineer features", "researcher")
Task("ML Engineer", "Implement and train model", "ml-developer")
Task("Evaluator", "Run holistic evaluation", "evaluator")
Task("DevOps Engineer", "Deploy model to production", "coder")
```

### Memory Integration

```bash
# Store model metadata
npx claude-flow@alpha memory store \
  --key "ml/models/my_model/metadata" \
  --value "$(cat model_card.json)"

# Retrieve training history
npx claude-flow@alpha memory retrieve \
  --key "ml/experiments/training_history"
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_trainer.py -v

# Run with coverage
pytest tests/ --cov=resources/scripts --cov-report=html
```

## Configuration

### Training Configuration (training-config.yaml)

```yaml
model:
  architecture: resnet50
  pretrained: true
  num_classes: 10

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: adam
  scheduler: cosine

augmentation:
  enabled: true
  transforms:
    - random_flip
    - random_crop
    - color_jitter

logging:
  tensorboard: true
  wandb: true
  checkpoint_every: 10
```

## Best Practices

1. **Data Quality**: Always validate and clean data before training
2. **Reproducibility**: Set random seeds and track all hyperparameters
3. **Evaluation**: Use multiple metrics and cross-validation
4. **Fairness**: Check for bias in predictions across different groups
5. **Monitoring**: Track model performance in production
6. **Versioning**: Version models, data, and code together

## Advanced Features

### Distributed Training

```python
# PyTorch DDP
trainer = ModelTrainer('config.yaml', distributed=True)
trainer.train(world_size=4)  # 4 GPUs
```

### Hyperparameter Optimization

```python
# Optuna integration
from resources.scripts.model_trainer import optimize_hyperparameters

best_params = optimize_hyperparameters(
    train_data, val_data,
    n_trials=100,
    metric='accuracy'
)
```

### Model Interpretation

```python
# SHAP values for model interpretability
from resources.scripts.model_evaluator import explain_predictions

explainer = explain_predictions(model, test_data)
explainer.plot_summary()
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Overfitting**: Add regularization, dropout, or data augmentation
3. **Poor Performance**: Check data quality, try different architectures
4. **Slow Training**: Use mixed precision, distributed training, or smaller model

### Debug Mode

```python
trainer = ModelTrainer('config.yaml', debug=True)
# Enables verbose logging and validation checks
```

## Related Skills

- `when-debugging-ml-training-use-ml-training-debugger` - ML-specific debugging
- `when-developing-ml-models-use-ml-expert` - Advanced ML development
- `agentdb-learning` - Reinforcement learning algorithms
- `holistic-evaluation` - Comprehensive evaluation
- `deployment-readiness` - Production deployment

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [SHAP for Model Interpretation](https://shap.readthedocs.io/)

## License

Part of the SPARC Three-Loop System - MIT License
