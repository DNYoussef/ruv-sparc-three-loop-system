---
name: ml
description: Machine Learning development workflow with experiment tracking, hyperparameter optimization, and MLOps integration
tier: gold
version: 2.0.0
category: specialized-development
tags:
  - machine-learning
  - mlops
  - experiment-tracking
  - hyperparameter-tuning
  - model-registry
agents:
  - ml-developer
  - data-scientist
  - mlops-engineer
tools:
  - experiment-tracker
  - hyperparameter-tuner
  - model-registry
  - ml-ops-pipeline
dependencies:
  - python-specialist
  - testing-quality
  - functionality-audit
prerequisites:
  - Python 3.8+
  - ML frameworks (TensorFlow/PyTorch/scikit-learn)
  - Docker (for MLOps)
  - Git LFS (for model versioning)
---

# ML Development Skill

Comprehensive machine learning development workflow with enterprise-grade experiment tracking, automated hyperparameter optimization, model registry management, and production MLOps pipelines.

## Overview

This Gold-tier skill provides a complete ML development lifecycle with:
- **Experiment Tracking**: MLflow/W&B integration for reproducible experiments
- **Hyperparameter Optimization**: Optuna/Ray Tune for automated tuning
- **Model Registry**: Centralized model versioning and deployment
- **MLOps Pipeline**: Production-ready model serving and monitoring

## Quick Start

```bash
# Initialize ML project
npx claude-flow sparc run ml "Create ML project for image classification"

# Track experiment
python resources/scripts/experiment-tracker.py --config experiment-config.yaml

# Optimize hyperparameters
node resources/scripts/hyperparameter-tuner.js --space hyperparameter-space.json

# Deploy model
bash resources/scripts/model-registry.sh deploy production latest
```

## Workflow Phases

### 1. Experiment Design
- Define hypothesis and metrics
- Configure experiment tracking
- Set up data pipelines
- Validate data quality

### 2. Model Development
- Implement model architecture
- Configure training pipeline
- Set up validation strategy
- Enable experiment logging

### 3. Hyperparameter Optimization
- Define search space
- Select optimization algorithm
- Run distributed trials
- Analyze results

### 4. Model Evaluation
- Comprehensive metrics analysis
- Cross-validation
- Error analysis
- Model interpretability

### 5. Model Deployment
- Register model in registry
- Create deployment pipeline
- Set up monitoring
- Enable A/B testing

## Resources

### Scripts
- `experiment-tracker.py`: MLflow/W&B experiment tracking with auto-logging
- `hyperparameter-tuner.js`: Distributed hyperparameter optimization
- `model-registry.sh`: Model versioning and deployment automation
- `ml-ops.py`: End-to-end MLOps pipeline orchestration

### Templates
- `experiment-config.yaml`: Experiment configuration template
- `hyperparameter-space.json`: Hyperparameter search space definition
- `model-card.md`: Model documentation template

## Examples

### 1. Experiment Tracking
150-line example showing MLflow integration with auto-logging, artifact tracking, and metric visualization.

### 2. Hyperparameter Optimization
250-line example demonstrating Optuna-based distributed hyperparameter tuning with pruning and parallel trials.

### 3. MLOps Pipeline
300-line example implementing complete MLOps workflow with model registry, CI/CD, and monitoring.

## Best Practices

1. **Reproducibility**
   - Track all experiment parameters
   - Version control data and code
   - Use deterministic random seeds
   - Document environment dependencies

2. **Experiment Organization**
   - Use hierarchical experiment structure
   - Tag experiments meaningfully
   - Archive failed experiments
   - Maintain experiment runbooks

3. **Model Management**
   - Semantic versioning for models
   - Comprehensive model cards
   - Automated model testing
   - Deployment staging (dev/staging/prod)

4. **Performance Optimization**
   - Distributed training for large models
   - Mixed precision training
   - Efficient data loading
   - Model compression techniques

5. **Monitoring & Observability**
   - Real-time metric tracking
   - Data drift detection
   - Model performance degradation alerts
   - Resource utilization monitoring

## Integration Points

- **Data**: AgentDB for vector search, PostgreSQL for metadata
- **Compute**: Flow Nexus sandboxes for distributed training
- **CI/CD**: Automated model testing and deployment
- **Memory**: Store experiment insights in Memory MCP

## Advanced Features

- **AutoML**: Automated architecture search and feature engineering
- **Distributed Training**: Multi-GPU and multi-node training
- **Model Compression**: Quantization, pruning, distillation
- **Federated Learning**: Privacy-preserving distributed training
- **Continuous Training**: Automated retraining on new data

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size, enable gradient checkpointing
2. **Slow Training**: Use mixed precision, optimize data pipeline
3. **Poor Convergence**: Adjust learning rate, check data quality
4. **Deployment Failures**: Validate model compatibility, test inference

### Debug Mode
```bash
# Enable verbose logging
export ML_DEBUG=1
python resources/scripts/experiment-tracker.py --debug
```

## Performance Metrics

- **Experiment Setup**: 2-5 minutes
- **Hyperparameter Optimization**: 30min - 6 hours (depending on search space)
- **Model Deployment**: 5-10 minutes
- **Full MLOps Pipeline**: 1-2 hours

## Support

For issues or questions:
- Check examples directory for reference implementations
- Review test files for usage patterns
- Consult MLflow/Optuna documentation
- Use `functionality-audit` skill for validation
