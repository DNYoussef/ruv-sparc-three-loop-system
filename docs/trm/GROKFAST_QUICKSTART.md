# GrokFast Quick Start Guide for Trainer Agent

**TL;DR**: Add 3 lines to training loop for 10-50× faster convergence.

---

## Minimal Integration (Copy-Paste Ready)

### Step 1: Add filter function

```python
import torch

def gradfilter_ema(model, grads=None, alpha=0.99, lamb=5.0):
    """Apply GrokFast EMA gradient filter.

    Args:
        model: PyTorch model
        grads: Previous gradient EMA (None for first call)
        alpha: EMA decay (0.95-0.99, default 0.98)
        lamb: Amplification (2.0-10.0, default 2.0)

    Returns:
        Updated gradient EMA
    """
    if grads is None:
        grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                 if p.requires_grad}

    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            # Update EMA
            grads[n] = alpha * grads[n] + (1 - alpha) * p.grad
            # Amplify slow component
            p.grad = p.grad + lamb * grads[n]

    return grads
```

### Step 2: Initialize before training

```python
grokfast_grads = None  # One line before loop
```

### Step 3: Apply filter in training loop

```python
for batch in dataloader:
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()

    # INSERT HERE: Apply GrokFast (between backward and step)
    grokfast_grads = gradfilter_ema(
        model,
        grads=grokfast_grads,
        alpha=0.98,   # Conservative
        lamb=2.0      # Safe start
    )

    optimizer.step()
```

---

## Recommended Hyperparameters

### Conservative (Safe Start)
```python
alpha = 0.98
lamb = 2.0
# Expected: 10-20× speedup, very stable
```

### Moderate (Balanced)
```python
alpha = 0.99
lamb = 5.0
# Expected: 30-50× speedup, may need weight_decay tuning
```

### Aggressive (Max Speed)
```python
alpha = 0.99
lamb = 10.0
# Expected: 50-100× speedup, requires careful tuning
# Increase weight_decay by 5-10× baseline
```

---

## Verification Checklist

✅ **Before first run**: Log baseline convergence speed (no GrokFast)
✅ **After integration**: Compare val accuracy curves (should converge faster)
✅ **Monitor**: Generalization gap (train_acc - val_acc) should shrink faster
✅ **Confirm**: Final accuracy matches or exceeds baseline

---

## Common Issues

| Problem | Solution |
|---------|----------|
| **"No acceleration"** | Increase `lamb` (try 5.0, 10.0) |
| **"Training unstable"** | Decrease `lamb` (try 1.0, 2.0) or increase weight_decay |
| **"Memory overflow"** | Use EMA filter (this one), not MA filter |
| **"Still slow"** | Check filter is called BETWEEN backward/step (not after) |

---

## Quick Test

```python
# Test with 1 batch to verify no errors
batch = next(iter(dataloader))
optimizer.zero_grad()
loss = compute_loss(model, batch)
loss.backward()
grokfast_grads = gradfilter_ema(model, grokfast_grads, alpha=0.98, lamb=2.0)
optimizer.step()
print("✅ GrokFast integration successful!")
```

---

## References

📄 Full Research: `docs/trm/GROKFAST_RESEARCH.md`
📄 Paper: https://arxiv.org/abs/2405.20233
📄 Official Code: https://github.com/ironjr/grokfast
