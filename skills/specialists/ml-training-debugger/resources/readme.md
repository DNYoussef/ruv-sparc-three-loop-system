# ML Training Debugger Resources

This directory contains utilities, templates, and reference materials for debugging machine learning training failures.

## Directory Structure

```
resources/
├── readme.md                    # This file
├── scripts/                     # Diagnostic scripts
│   ├── loss-analyzer.py         # Loss curve analysis and visualization
│   ├── gradient-debugger.py     # Gradient flow analysis
│   ├── overfitting-detector.js  # Overfitting detection and metrics
│   └── training-monitor.sh      # Real-time training monitoring
└── templates/                   # Configuration templates
    ├── debug-config.yaml        # Debug configuration template
    ├── loss-curve-template.json # Loss curve data format
    └── training-metrics.yaml    # Training metrics schema
```

## Scripts

### loss-analyzer.py
Analyzes training loss curves to detect anomalies, trends, and pathological behavior.

**Features**:
- Loss divergence detection
- Plateau identification
- Noise analysis
- Visualization generation

**Usage**:
```bash
python loss-analyzer.py --log-file train.log --output loss_analysis.png
```

### gradient-debugger.py
Analyzes gradient flow through neural network layers to identify vanishing/exploding gradients.

**Features**:
- Layer-wise gradient statistics
- Gradient norm tracking
- Gradient flow visualization
- Anomaly detection

**Usage**:
```bash
python gradient-debugger.py --checkpoint model.pt --config config.yaml
```

### overfitting-detector.js
Detects overfitting by analyzing train vs validation metrics.

**Features**:
- Train/val gap analysis
- Early stopping recommendations
- Regularization suggestions
- Metric tracking

**Usage**:
```bash
node overfitting-detector.js --metrics metrics.json
```

### training-monitor.sh
Real-time monitoring of training progress with alert triggers.

**Features**:
- Live metric streaming
- Alert thresholds
- Resource monitoring (GPU, memory)
- Log aggregation

**Usage**:
```bash
./training-monitor.sh --log-dir ./logs --alert-threshold 0.1
```

## Templates

### debug-config.yaml
Standard configuration for debugging sessions.

**Includes**:
- Artifact paths
- Analysis parameters
- Output settings
- Alert thresholds

### loss-curve-template.json
JSON schema for loss curve data format.

**Fields**:
- Epoch/step numbers
- Loss values (train/val)
- Learning rates
- Gradient norms
- Custom metrics

### training-metrics.yaml
YAML schema for comprehensive training metrics.

**Sections**:
- Loss metrics
- Gradient statistics
- Model parameters
- Resource utilization
- Custom tracking

## Best Practices

1. **Data Collection**: Ensure comprehensive logging from the start
2. **Artifact Preservation**: Save checkpoints at key training milestones
3. **Systematic Analysis**: Follow the debugging methodology in order
4. **Evidence-Based**: Always verify hypotheses against actual data
5. **Documentation**: Record findings and fixes for future reference

## Common Workflows

### Diagnosing Loss Divergence
1. Run `loss-analyzer.py` on training logs
2. Identify divergence point and magnitude
3. Run `gradient-debugger.py` at divergence checkpoint
4. Correlate with learning rate schedule changes
5. Apply recommended fixes from analysis

### Detecting Mode Collapse
1. Analyze model outputs for diversity
2. Check architecture balance with parameter counts
3. Review loss curve for premature plateau
4. Examine gradient flow for bottlenecks
5. Test regularization and capacity adjustments

### Fixing Gradient Issues
1. Run `gradient-debugger.py` across multiple checkpoints
2. Identify layers with vanishing/exploding gradients
3. Check activation functions and initialization
4. Review architecture depth and skip connections
5. Apply gradient clipping or normalization

## Integration

These resources are used automatically by the ML debugging agent when spawned via the skill. They can also be used standalone for manual analysis.

## Updates

Resources are version-controlled and tested against real ML training scenarios. Submit feedback or contributions via the skill repository.
