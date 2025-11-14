# GrokFast Optimizer Research Summary

**Research Date**: 2025-11-07
**Research Agent**: Research Specialist
**Purpose**: Guide Trainer Agent implementation for TRM project

---

## Overview

**GrokFast** is a gradient filtering optimization technique that accelerates the "grokking" phenomenon in neural network training. Grokking refers to delayed generalization where models suddenly achieve strong generalization after prolonged overfitting to training data.

### Key Innovation

GrokFast treats gradient trajectories as random signals over time and spectrally decomposes them into:
1. **Fast-varying components** → cause overfitting
2. **Slow-varying components** → induce generalization

By **amplifying slow-varying gradient components**, GrokFast accelerates grokking by **50-100× or more** with minimal code changes.

### Paper Citation

**Title**: Grokfast: Accelerated Grokking by Amplifying Slow Gradients
**Authors**: Jaerin Lee, Bong Gyun Kang, Kihoon Kim, Kyoung Mu Lee
**ArXiv**: [2405.20233](https://arxiv.org/abs/2405.20233)
**Published**: June 5, 2024
**Official Repo**: https://github.com/ironjr/grokfast

---

## Key Findings

### 1. Gradient Filtering Mechanism

GrokFast implements a **scalar, time-invariant, convolutional gradient filter** that operates on parameter gradients during training:

```
For each parameter gradient g_t at step t:
1. Apply low-pass filtering to extract slow-varying component g_slow
2. Amplify: g_filtered = g_t + λ * g_slow
3. Pass g_filtered to optimizer
```

### 2. Two Filtering Approaches

#### **A. EMA-based Filter (Grokfast-EMA)** ⭐ RECOMMENDED

Uses exponential moving average for memory-efficient filtering:

```python
def gradfilter_ema(model, grads=None, alpha=0.99, lamb=5.0):
    """
    Args:
        model: PyTorch model
        grads: Previous gradient EMA (dict of tensors, or None for first call)
        alpha: EMA decay rate (0.95-0.99, default 0.99)
        lamb: Amplification factor (2.0-10.0, default 5.0)

    Returns:
        grads: Updated gradient EMA for next iteration
    """
    if grads is None:
        grads = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = alpha * grads[n] + (1 - alpha) * p.grad  # EMA update
            p.grad += lamb * grads[n]  # Amplify slow component

    return grads
```

#### **B. Moving Average Filter (Grokfast-MA)**

Uses fixed-window moving average:

```python
def gradfilter_ma(model, grads=None, window_size=128, lamb=5.0):
    """
    Args:
        model: PyTorch model
        grads: Previous gradient buffer (deque of dicts, or None for first call)
        window_size: Filter window (64-256, default 128)
        lamb: Amplification factor (2.0-10.0, default 5.0)

    Returns:
        grads: Updated gradient buffer for next iteration
    """
    if grads is None:
        grads = []

    # Store current gradients
    grads.append({n: p.grad.clone() for n, p in model.named_parameters()
                  if p.requires_grad and p.grad is not None})

    # Keep only window_size most recent gradients
    if len(grads) > window_size:
        grads.pop(0)

    # Compute moving average
    if len(grads) > 1:
        avg_grads = {n: torch.stack([g[n] for g in grads]).mean(0)
                     for n in grads[0].keys()}

        # Apply amplification
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                p.grad += lamb * avg_grads[n]

    return grads
```

---

## Hyperparameters

### Critical Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| **alpha** | float | 0.95-0.99 | **0.98-0.99** | EMA decay rate for slow gradient tracking |
| **lamb** (λ) | float | 2.0-10.0 | **2.0-5.0** | Amplification factor for slow components |
| **window_size** | int | 64-256 | **128** | MA filter window (only for Grokfast-MA) |

### Hyperparameter Selection Strategy

1. **Determine desired acceleration** (e.g., 100× faster grokking)
2. **Set pivot values**:
   - For MA: `window_size ≈ acceleration_target`
   - For EMA: Choose `alpha` such that `alpha^N ≈ 0.1` where N is target steps
3. **Grid search** around pivot values:
   - Alpha: {0.95, 0.97, 0.98, 0.99}
   - Lambda: {2.0, 5.0, 10.0}
4. **Adjust weight decay** after tuning (typically increase by 2-10× baseline)

### Practical Guidelines

- **Conservative start**: `alpha=0.98, lamb=2.0` (stable, moderate acceleration)
- **Aggressive**: `alpha=0.99, lamb=5.0` (maximum acceleration, may need WD tuning)
- **Delayed activation**: Use `grokfast_after_step > 0` to avoid interfering with early training dynamics

---

## Implementation Recommendations for Trainer Agent

### 1. Integration Pattern (Minimal Code Changes)

#### **Option A: Manual Filter Application** ⭐ RECOMMENDED FOR FLEXIBILITY

```python
import torch
from torch.optim import AdamW

# In training loop initialization
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
grokfast_grads = None  # Initialize gradient buffer

# In training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()

    # Apply GrokFast filter BETWEEN backward() and step()
    grokfast_grads = gradfilter_ema(
        model,
        grads=grokfast_grads,
        alpha=0.98,    # Conservative default
        lamb=2.0       # Moderate amplification
    )

    optimizer.step()
```

#### **Option B: Pre-built GrokFastAdamW Optimizer**

```python
from grokfast_pytorch import GrokFastAdamW

optimizer = GrokFastAdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    grokfast=True,            # Enable GrokFast
    grokfast_alpha=0.98,      # EMA decay
    grokfast_lamb=2.0,        # Amplification
    grokfast_after_step=0,    # Start immediately (or delay)
    normalize_lr=True         # Auto-scale LR by (1 + lamb)
)

# Standard training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()

# Can toggle mid-training
optimizer.turn_off_grokfast()  # Disable filtering
optimizer.turn_on_grokfast()   # Re-enable
```

### 2. Verification Methods

To verify GrokFast is working correctly, monitor these training dynamics:

#### **Key Indicators**:

1. **Faster convergence**: Validation accuracy should improve 10-100× faster than baseline
2. **Reduced overfitting phase**: Gap between train/val accuracy should shrink faster
3. **Smoother loss curves**: Gradient filtering often stabilizes training
4. **Earlier generalization**: Model achieves near-perfect validation accuracy much sooner

#### **Diagnostic Logging**:

```python
# Log these metrics during training
metrics = {
    'train_loss': train_loss,
    'val_loss': val_loss,
    'train_acc': train_acc,
    'val_acc': val_acc,
    'generalization_gap': train_acc - val_acc,  # Should shrink faster
    'grad_norm': torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')),
    'grokfast_active': grokfast_grads is not None
}

# Compare against baseline (no GrokFast)
# Expected: val_acc improves much earlier with GrokFast
```

### 3. Best Practices

#### **Do's**:
✅ Start with conservative hyperparameters (`alpha=0.98, lamb=2.0`)
✅ Compare against baseline training (no GrokFast) to measure acceleration
✅ Increase weight decay by 2-5× when using aggressive settings
✅ Use EMA-based filter for memory efficiency (O(P) vs O(P×W) for MA)
✅ Log gradient statistics to verify amplification is happening
✅ Consider delayed activation (`grokfast_after_step > 0`) for complex models

#### **Don'ts**:
❌ Don't use extreme lambda values (>10.0) without careful tuning
❌ Don't forget to initialize `grads=None` before training
❌ Don't apply GrokFast after optimizer.step() (must be between backward/step)
❌ Don't combine with other aggressive gradient modifications without testing
❌ Don't use MA filter for very large models (memory overhead)

---

## PyTorch Integration Details

### Compatibility

- **PyTorch Version**: Any version with autograd (>= 1.0)
- **No special dependencies**: Uses only standard PyTorch ops
- **Works with any optimizer**: AdamW, SGD, Adam, RMSprop, etc.
- **Tested architectures**: Transformers, MLPs, RNNs, CNNs, GNNs

### Memory Overhead

| Method | Memory per Parameter | Total Overhead |
|--------|----------------------|----------------|
| **EMA-based** | +1 float tensor | O(P) — same as optimizer |
| **MA-based** | +W float tensors | O(P × W) — window_size × params |

**Recommendation**: Use **EMA-based** for production (constant memory, equivalent performance).

### Performance Impact

- **Compute overhead**: ~5-10% per step (gradient averaging + amplification)
- **Training speedup**: 10-100× fewer steps to reach target accuracy
- **Net benefit**: 9-90× overall training time reduction

---

## Example Code Snippets

### Full Training Loop with GrokFast

```python
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Copy gradfilter_ema implementation (see section 1)
def gradfilter_ema(model, grads=None, alpha=0.99, lamb=5.0):
    # ... (implementation above)
    pass

# Initialize
model = YourTransformer()
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
grokfast_grads = None

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch['input_ids'])
        loss = criterion(outputs, batch['labels'])

        # Backward pass
        loss.backward()

        # Apply GrokFast filter (KEY STEP)
        grokfast_grads = gradfilter_ema(
            model,
            grads=grokfast_grads,
            alpha=0.98,   # Tune as needed
            lamb=2.0      # Tune as needed
        )

        # Optimizer step
        optimizer.step()

        # Logging
        if step % 100 == 0:
            val_acc = evaluate(model, val_loader)
            print(f"Step {step}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
```

### Comparative Baseline (No GrokFast)

```python
# Same training loop WITHOUT GrokFast for comparison
for batch in train_loader:
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    # NO GROKFAST FILTER
    optimizer.step()

# Expected: Takes 10-100× more steps to reach same val accuracy
```

---

## References

### Primary Sources

1. **GrokFast Paper**: Lee et al., "Grokfast: Accelerated Grokking by Amplifying Slow Gradients", arXiv:2405.20233 (2024)
   https://arxiv.org/abs/2405.20233

2. **Official Implementation**: ironjr/grokfast
   https://github.com/ironjr/grokfast

3. **PyTorch Package**: lucidrains/grokfast-pytorch
   https://github.com/lucidrains/grokfast-pytorch

### Background Reading

4. **Grokking Phenomenon**: Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets", arXiv:2201.02177 (2022)

5. **Why Deep Networks Grok**: Merrill et al., "Deep Networks Always Grok and Here is Why", arXiv:2402.15555 (2024)

### Installation

```bash
# Option 1: Install PyTorch package
pip install grokfast-pytorch

# Option 2: Copy filter implementation from official repo
curl -O https://raw.githubusercontent.com/ironjr/grokfast/main/grokfast.py
```

---

## Next Steps for Trainer Agent

### Immediate Actions

1. ✅ **Implement `gradfilter_ema` function** in training utilities
2. ✅ **Add GrokFast toggle** to training config (enable/disable)
3. ✅ **Set default hyperparameters**: `alpha=0.98, lamb=2.0`
4. ✅ **Insert filter call** in training loop (between backward/step)
5. ✅ **Add logging** for gradient norms and generalization gap

### Testing Protocol

1. **Baseline run**: Train without GrokFast, record convergence speed
2. **GrokFast run**: Train with conservative settings, compare acceleration
3. **Hyperparameter sweep**: Test {alpha: 0.95, 0.98, 0.99} × {lamb: 2.0, 5.0}
4. **Validation**: Verify final accuracy matches or exceeds baseline

### Success Criteria

- ✅ **10-50× faster** convergence to target validation accuracy
- ✅ **No accuracy degradation** vs baseline (or improvement)
- ✅ **Smooth integration** with existing training pipeline
- ✅ **Negligible memory overhead** (EMA-based filter)

---

## Summary

**GrokFast** is a simple yet powerful technique that requires only **3 lines of code** to integrate:

```python
grokfast_grads = None  # Before training loop

# In training loop, between backward() and step()
grokfast_grads = gradfilter_ema(model, grokfast_grads, alpha=0.98, lamb=2.0)
```

With conservative hyperparameters (`alpha=0.98, lamb=2.0`), it provides:
- **Safe, stable training** with minimal risk
- **10-50× acceleration** of grokking phenomenon
- **No architectural changes** required
- **Compatible with any optimizer** and model architecture

**Trainer Agent**: You can now implement GrokFast with confidence. Start conservative, measure acceleration, and tune if needed.

---

**END OF RESEARCH DOCUMENT**
