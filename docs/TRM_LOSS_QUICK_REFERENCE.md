# TRM Loss Functions - Quick Reference

## Installation

No installation needed - pure PyTorch implementation.

```python
from src.training.trm_loss_functions import TRMLoss
```

---

## Quick Start

```python
# Initialize
criterion = TRMLoss(lambda_halt=0.01, lambda_profit=1.0)

# Forward pass
loss = criterion(task_logits, halt_logits, labels, pnl)

# Backward
loss.backward()
```

---

## Loss Components

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Task Loss** | `CrossEntropy(logits, labels)` | 8-way strategy classification |
| **Halt Loss** | `BCE(halt_logits, dynamic_target)` | Halt when confident & correct |
| **Profit Weight** | `1 - tanh(pnl / 0.05)` | Learn more from losses |

**Combined:**
```
L = 0.01 * L_halt + 1.0 * (profit_weight * L_task)
```

---

## Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lambda_halt` | 0.01 | 0-1 | Weight for halt loss |
| `lambda_profit` | 1.0 | 0-10 | Weight for profit-weighted loss |
| `confidence_threshold` | 0.7 | 0.5-0.95 | Threshold for halting |
| `pnl_scale` | 0.05 | 0.01-0.20 | PnL scaling factor (5%) |

---

## Input Requirements

```python
task_logits: (batch_size, 8)      # 8-way strategy logits
halt_logits: (batch_size, 1)      # Halt decision logits
labels:      (batch_size,)        # Ground truth (0-7)
pnl:         (batch_size,)        # Profit/loss per sample
```

---

## Profit Weighting Examples

| PnL | Weight | Effect |
|-----|--------|--------|
| +10% | -0.87 | Discourage |
| +5% | 0.00 | Neutral |
| 0% | 1.00 | Normal |
| -5% | 2.00 | Emphasize |
| -10% | 2.87 | Strongly emphasize |

**Formula:** `weight = 1 - tanh(pnl / pnl_scale)`

---

## Halt Logic

**Halt target = 1 when:**
- Confidence > threshold (e.g., 0.7)
- AND prediction is correct

**Otherwise halt target = 0**

```python
probs = softmax(task_logits)
is_confident = max(probs) > 0.7
is_correct = argmax(probs) == label
halt_target = is_confident AND is_correct
```

---

## Class Balancing

For imbalanced strategies:

```python
# Compute weights from training data
counts = torch.bincount(train_labels)
weights = 1.0 / counts
weights = weights / weights.sum() * len(weights)

# Use in loss
criterion = TRMLoss(class_weights=weights)
```

---

## Monitoring Loss Components

```python
components = criterion(
    task_logits, halt_logits, labels, pnl,
    return_components=True
)

# Log individual components
wandb.log({
    'loss/total': components['total_loss'],
    'loss/halt': components['halt_loss'],
    'loss/task': components['task_loss'],
    'loss/profit_weighted': components['profit_weighted_task_loss']
})
```

---

## Testing

```bash
# Run all tests
pytest tests/test_trm_loss_functions.py -v

# Run specific test class
pytest tests/test_trm_loss_functions.py::TestTRMLoss -v

# Run sanity test
python src/training/trm_loss_functions.py
```

---

## Common Issues

### 1. Wrong number of classes
```python
# ❌ ERROR
task_logits = torch.randn(batch, 10)  # Wrong!

# ✅ CORRECT
task_logits = torch.randn(batch, 8)   # Must be 8 classes
```

### 2. Batch size mismatch
```python
# ❌ ERROR
labels = torch.randint(0, 8, (32,))
pnl = torch.randn(64)  # Wrong size!

# ✅ CORRECT
labels = torch.randint(0, 8, (32,))
pnl = torch.randn(32)  # Same batch size
```

### 3. Halt logits shape
```python
# Both work:
halt_logits = torch.randn(batch, 1)  # 2D
halt_logits = torch.randn(batch)     # 1D
```

---

## Tuning Tips

### Lambda Weights

**High `lambda_halt` (e.g., 0.1)**
- More emphasis on halting decisions
- Use if model halts too often/rarely

**High `lambda_profit` (e.g., 5.0)**
- More emphasis on profitability
- Use if maximizing returns is critical

### Confidence Threshold

**Low threshold (e.g., 0.5)**
- Halt more often
- Lower latency but might be wrong

**High threshold (e.g., 0.9)**
- Halt less often
- Higher latency but more accurate

### PnL Scale

**Small scale (e.g., 0.01)**
- Stronger profit weighting
- More sensitive to PnL variations

**Large scale (e.g., 0.20)**
- Weaker profit weighting
- More robust to PnL noise

---

## Files

- **Implementation**: `src/training/trm_loss_functions.py` (266 lines)
- **Tests**: `tests/test_trm_loss_functions.py` (469 lines)
- **Docs**: `docs/TRM_LOSS_FUNCTIONS_IMPLEMENTATION.md`

---

## Status

✅ **Complete** - 23/23 tests passing
✅ **Validated** - Sanity tests pass
✅ **Ready** - For Phase 3 integration
