---
skill_name: machine-learning
description: Comprehensive machine learning development with training, evaluation, and deployment capabilities
version: 2.0.0
tier: gold
author: SPARC System
tags: [ml, deep-learning, training, evaluation, deployment, data-science]
dependencies:
  - python-specialist
  - agentdb-learning
  - ml-expert
agents:
  - ml-developer
  - data-steward
  - evaluator
related_skills:
  - when-debugging-ml-training-use-ml-training-debugger
  - when-developing-ml-models-use-ml-expert
  - agentdb-learning
  - holistic-evaluation
  - baseline-replication
---

# Machine Learning Development Skill

Complete workflow for machine learning model development, training, evaluation, and deployment.

## When to Use

Auto-trigger when detecting:
- "train model", "machine learning", "ML pipeline"
- "deep learning", "neural network", "model training"
- "data preprocessing", "feature engineering"
- "model evaluation", "hyperparameter tuning"
- "model deployment", "ML ops"

## Capabilities

### 1. Data Pipeline
- Data preprocessing and cleaning
- Feature engineering and selection
- Data augmentation
- Train/validation/test splitting
- Data versioning with DVC

### 2. Model Training
- Neural network architectures
- Hyperparameter optimization
- Transfer learning
- Distributed training
- Training monitoring and logging

### 3. Model Evaluation
- Multi-metric evaluation
- Cross-validation
- Confusion matrices and ROC curves
- Fairness and bias detection
- Performance benchmarking

### 4. Model Deployment
- Model serialization and versioning
- API endpoint creation
- Containerization
- Monitoring and logging
- A/B testing support

## Agent Workflow

```javascript
// Auto-spawned agents for ML development
Task("ML Researcher", "Research SOTA models and best practices for [task]", "researcher")
Task("Data Engineer", "Preprocess data and engineer features", "coder")
Task("ML Developer", "Implement and train model architecture", "ml-developer")
Task("Model Evaluator", "Evaluate model performance and fairness", "evaluator")
Task("ML Ops Engineer", "Deploy model with monitoring", "coder")
```

## Integration Points

- **AgentDB Learning**: Reinforcement learning algorithms
- **ML Expert**: Advanced model development
- **Holistic Evaluation**: Comprehensive evaluation metrics
- **Data Steward**: Dataset documentation and bias auditing
- **Deployment Readiness**: Production ML deployment

## Resources

- `resources/scripts/model-trainer.py` - Complete training pipeline
- `resources/scripts/data-preprocessor.py` - Data preprocessing utilities
- `resources/scripts/model-evaluator.js` - Evaluation framework
- `resources/scripts/ml-pipeline.sh` - End-to-end ML pipeline
- `resources/templates/training-config.yaml` - Training configuration
- `resources/templates/model-architecture.json` - Model architecture definitions
- `resources/templates/evaluation-metrics.yaml` - Evaluation metrics configuration

## Examples

See `examples/` directory:
- `model-training.py` - Complete model training workflow
- `data-pipeline.py` - Data preprocessing and feature engineering
- `model-deployment.py` - Model deployment with FastAPI

## Testing

Run tests with:
```bash
pytest tests/
```

## Performance

- Supports distributed training with PyTorch DDP
- GPU acceleration for training and inference
- Automated hyperparameter tuning with Optuna
- Model versioning and experiment tracking with MLflow
