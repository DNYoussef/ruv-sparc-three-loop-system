# ML Development Skill - Gold Tier

Enterprise-grade machine learning development workflow with comprehensive experiment tracking, automated hyperparameter optimization, and production MLOps capabilities.

## 🏆 Gold Tier Features

This skill includes:
- ✅ Complete skill.md with 5 workflow phases
- ✅ 4 production-ready scripts (Python, JavaScript, Bash)
- ✅ 3 configuration templates (YAML, JSON, Markdown)
- ✅ 3 comprehensive examples (150-300 lines each)
- ✅ 3 test files with integration testing
- ✅ Best practices and troubleshooting guides

## 📁 Directory Structure

```
ml/
├── skill.md                          # Main skill documentation
├── README.md                         # This file
├── resources/
│   ├── scripts/
│   │   ├── experiment-tracker.py     # MLflow/W&B integration
│   │   ├── hyperparameter-tuner.js   # Distributed optimization
│   │   ├── model-registry.sh         # Model versioning
│   │   └── ml-ops.py                 # MLOps orchestration
│   └── templates/
│       ├── experiment-config.yaml    # Experiment configuration
│       ├── hyperparameter-space.json # Search space definition
│       └── model-card.md             # Model documentation
├── tests/
│   ├── test-experiment-tracking.py   # Experiment tracking tests
│   ├── test-hyperparameter-tuning.js # Optimization tests
│   └── test-mlops-pipeline.py        # MLOps integration tests
└── examples/
    ├── experiment-tracking.py        # 150 lines: MLflow example
    ├── hyperparameter-optimization.js # 250 lines: Optuna example
    └── mlops-pipeline.py             # 300 lines: Full MLOps workflow
```

## 🚀 Quick Start

### 1. Experiment Tracking

```bash
# Configure experiment
cp resources/templates/experiment-config.yaml my-experiment.yaml

# Run experiment with tracking
python resources/scripts/experiment-tracker.py --config my-experiment.yaml

# View results
mlflow ui
```

### 2. Hyperparameter Optimization

```bash
# Define search space
cp resources/templates/hyperparameter-space.json my-space.json

# Run optimization
node resources/scripts/hyperparameter-tuner.js --space my-space.json --trials 100

# Analyze results
node resources/scripts/hyperparameter-tuner.js --analyze
```

### 3. Model Deployment

```bash
# Register model
bash resources/scripts/model-registry.sh register "my-model" "v1.0.0" "./model.pkl"

# Deploy to staging
bash resources/scripts/model-registry.sh deploy staging my-model v1.0.0

# Promote to production
bash resources/scripts/model-registry.sh promote my-model v1.0.0
```

### 4. Full MLOps Pipeline

```bash
# Run complete pipeline
python resources/scripts/ml-ops.py --pipeline full --model-name "image-classifier"

# Monitor pipeline
python resources/scripts/ml-ops.py --monitor --interval 60
```

## 📚 Examples

### Example 1: Experiment Tracking (150 lines)
Demonstrates MLflow integration with:
- Auto-logging for multiple frameworks
- Custom metric tracking
- Artifact management
- Experiment comparison

**Run**: `python examples/experiment-tracking.py`

### Example 2: Hyperparameter Optimization (250 lines)
Shows Optuna-based distributed tuning with:
- Bayesian optimization
- Pruning strategies
- Parallel trials
- Result visualization

**Run**: `node examples/hyperparameter-optimization.js`

### Example 3: MLOps Pipeline (300 lines)
Complete production workflow featuring:
- Data validation
- Model training & evaluation
- Registry integration
- Deployment automation
- Performance monitoring

**Run**: `python examples/mlops-pipeline.py`

## 🧪 Testing

```bash
# Run all tests
cd tests
python -m pytest test-experiment-tracking.py -v
node test-hyperparameter-tuning.js
python -m pytest test-mlops-pipeline.py -v

# Integration tests
python -m pytest test-mlops-pipeline.py::test_full_pipeline_integration -v
```

## 🎯 Use Cases

1. **Image Classification**: Train and deploy CNN models with experiment tracking
2. **NLP Tasks**: Fine-tune transformers with distributed hyperparameter search
3. **Recommendation Systems**: Build and monitor collaborative filtering models
4. **Time Series Forecasting**: Develop and productionize forecasting models
5. **Anomaly Detection**: Create and deploy unsupervised learning models

## 🔧 Configuration

### Experiment Config (YAML)
```yaml
experiment:
  name: "image-classification"
  tracking_uri: "http://localhost:5000"
  artifact_location: "./mlruns"

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

model:
  architecture: "resnet50"
  optimizer: "adam"
  loss: "categorical_crossentropy"
```

### Hyperparameter Space (JSON)
```json
{
  "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": true},
  "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
  "dropout": {"type": "float", "low": 0.1, "high": 0.5}
}
```

## 📊 Performance Benchmarks

| Operation | Duration | Resources |
|-----------|----------|-----------|
| Experiment Setup | 2-5 min | 1 CPU core |
| Single Training Run | 10-60 min | GPU recommended |
| Hyperparameter Optimization (50 trials) | 2-6 hours | Multi-GPU/distributed |
| Model Deployment | 5-10 min | 2 CPU cores |
| Full MLOps Pipeline | 1-2 hours | Distributed cluster |

## 🔗 Integration

- **AgentDB**: Vector search for similar experiments
- **Memory MCP**: Store experiment insights and learnings
- **Flow Nexus**: Distributed training in cloud sandboxes
- **GitHub**: Version control for code and DVC for data
- **CI/CD**: Automated model testing and deployment

## 🛡️ Best Practices

1. **Always version your data** using Git LFS or DVC
2. **Track every experiment** even failed ones (valuable insights)
3. **Use semantic versioning** for models (major.minor.patch)
4. **Create comprehensive model cards** for documentation
5. **Implement gradual rollouts** for production deployments
6. **Monitor data drift** to detect when retraining is needed
7. **Set up alerts** for model performance degradation

## 📖 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [ML Model Card Toolkit](https://github.com/tensorflow/model-card-toolkit)
- [DVC Documentation](https://dvc.org/doc)

## 🤝 Contributing

To enhance this skill:
1. Add new optimization algorithms
2. Integrate additional tracking backends
3. Create framework-specific examples
4. Improve deployment strategies
5. Add monitoring dashboards

## 📝 License

Part of the SPARC Three-Loop System - MIT License
