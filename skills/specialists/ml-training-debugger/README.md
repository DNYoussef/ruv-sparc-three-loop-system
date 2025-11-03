# ML Training Debugger Skill

**Tier**: Enhanced
**Version**: 1.1.0
**Type**: Agent-based skill with SDK implementation
**Domain**: Machine learning training diagnostics

## Overview

Diagnose machine learning training failures including loss divergence, mode collapse, gradient issues, architecture problems, and optimization failures. This skill spawns a specialist ML debugging agent that systematically analyzes training artifacts to identify root causes and propose evidence-based fixes.

## Quick Start

```bash
# Activate skill
claude-skill ml-training-debugger

# Example usage
"Debug my training run - loss diverging at epoch 7"
"Why is my model outputting all zeros?"
"Help diagnose gradient vanishing in transformer"
```

## Resources

### Scripts
- `resources/scripts/loss-analyzer.py` - Loss curve analysis and visualization
- `resources/scripts/gradient-debugger.py` - Gradient flow analysis
- `resources/scripts/overfitting-detector.js` - Overfitting detection and metrics
- `resources/scripts/training-monitor.sh` - Real-time training monitoring

### Templates
- `resources/templates/debug-config.yaml` - Debug configuration template
- `resources/templates/loss-curve-template.json` - Loss curve data format
- `resources/templates/training-metrics.yaml` - Training metrics schema

### Tests
- `tests/test-loss-divergence.py` - Loss divergence test cases
- `tests/test-gradient-analysis.py` - Gradient analysis validation
- `tests/test-mode-collapse.js` - Mode collapse detection tests

### Examples
- `examples/vanishing-gradients.py` - Vanishing gradients debugging workflow
- `examples/overfitting-detection.py` - Overfitting analysis example
- `examples/convergence-debugging.py` - Convergence issues diagnosis

## Features

- **Systematic Analysis**: Apply debugging methodology to training artifacts
- **Root Cause Identification**: Diagnose underlying issues with evidence
- **Fix Prioritization**: Rank solutions by impact and confidence
- **Evidence-Based Recommendations**: Propose fixes with detailed reasoning

## Use Cases

1. **Loss Divergence**: Training loss suddenly increases after stable epochs
2. **Mode Collapse**: Model outputs degenerate to single token or pattern
3. **Gradient Issues**: Vanishing or exploding gradients preventing learning
4. **Architecture Problems**: Parameter imbalance or capacity issues
5. **Optimization Failures**: Learning rate, optimizer, or scheduler misconfiguration

## Integration

This skill works with:
- `ml-expert` - For implementing recommended fixes
- `code-analyzer` - For architecture review
- `functionality-audit` - For validating fixes in sandbox

## Quality Standards

- ✅ Identify root cause with >80% confidence or request more data
- ✅ Provide evidence from actual artifacts (not speculation)
- ✅ Propose fixes with expected impact and reasoning
- ✅ Complete analysis within 5 minutes for typical cases
- ✅ Handle missing artifacts gracefully

## Documentation

- Full documentation: `skill.md`
- Agent prompt: `agents/ml-debugger-specialist.prompt`
- Process visualization: `ml-training-debugger-process.dot`

## Version History

- **1.1.0**: Enhanced tier with resources, tests, examples
- **1.0.0**: Initial release with basic debugging capabilities
