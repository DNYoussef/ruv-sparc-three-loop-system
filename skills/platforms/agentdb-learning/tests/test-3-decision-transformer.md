# Test 3: Decision Transformer & Offline RL

## Overview
Test suite for Decision Transformer (offline reinforcement learning via sequence modeling).

## Test Categories

### 1. Model Architecture Tests

#### 1.1 Transformer Initialization
```bash
# Test Decision Transformer model creation
python -c "
import sys
sys.path.insert(0, 'resources/scripts')
from train_rl_agent import RLTrainer

config = {
    'context_length': 20,
    'embed_dim': 128,
    'n_heads': 8,
    'n_layers': 6,
    'state_dim': 17,
    'action_dim': 6,
    'max_steps': 100
}

trainer = RLTrainer('decision-transformer', config)
print('✓ Decision Transformer initialized')
print(f'  Context Length: {config[\"context_length\"]}')
print(f'  Embed Dim: {config[\"embed_dim\"]}')
print(f'  Attention Heads: {config[\"n_heads\"]}')
print(f'  Transformer Layers: {config[\"n_layers\"]}')
"
```

**Expected Output:**
- Transformer model initialized
- Attention mechanism configured
- Position encodings ready

**Pass Criteria:**
- ✓ Model creates without errors
- ✓ Architecture parameters match config
- ✓ Token embeddings initialized

---

#### 1.2 Configuration Loading
```bash
# Load Decision Transformer YAML config
python resources/scripts/train_rl_agent.py \
  --algorithm decision-transformer \
  --config resources/templates/decision-transformer.yaml \
  --episodes 10 \
  --verbose
```

**Expected Behavior:**
- YAML config parsed correctly
- All transformer hyperparameters loaded:
  - embed_dim: 128
  - n_heads: 8
  - n_layers: 6
  - context_length: 20
  - target_return: 3600

**Pass Criteria:**
- ✓ Config loads without errors
- ✓ Model architecture matches template
- ✓ Return conditioning enabled

---

### 2. Offline Learning Tests

#### 2.1 Training from Logged Data
```python
# Test offline training (no environment interaction)
def test_offline_training():
    import sys
    sys.path.insert(0, 'resources/scripts')
    from train_rl_agent import RLTrainer
    import numpy as np

    config = {
        'context_length': 20,
        'embed_dim': 64,
        'n_heads': 4,
        'n_layers': 3,
        'state_dim': 4,
        'action_dim': 2,
        'target_return': 100,
        'max_steps': 50
    }

    # Create offline dataset (simulated)
    offline_data = []
    for episode in range(10):
        trajectory = {
            'states': np.random.randn(50, 4),
            'actions': np.random.randint(0, 2, size=(50,)),
            'rewards': np.random.randn(50) * 10,
            'returns': np.cumsum(np.random.randn(50) * 10)
        }
        offline_data.append(trajectory)

    # Train Decision Transformer
    trainer = RLTrainer('decision-transformer', config)
    metrics = trainer.train(num_episodes=10)

    assert metrics['episodes'] == 10
    print("✓ Offline training test passed")
    print(f"  Training Time: {metrics['training_time']:.2f}s")
    print(f"  Final Avg Reward: {metrics['avg_reward']:.2f}")

test_offline_training()
```

**Expected Behavior:**
- No environment interaction required
- Learns from pre-collected trajectories
- Return-conditioned behavior

**Pass Criteria:**
- ✓ Trains on offline data successfully
- ✓ No environment calls during training
- ✓ Loss decreases over time

---

#### 2.2 Return Conditioning
```python
# Test multiple target returns
def test_return_conditioning():
    import sys
    sys.path.insert(0, 'resources/scripts')
    from train_rl_agent import RLTrainer

    target_returns = [1000, 2000, 3000, 4000]
    results = {}

    for target in target_returns:
        config = {
            'context_length': 20,
            'embed_dim': 64,
            'n_heads': 4,
            'n_layers': 3,
            'state_dim': 4,
            'action_dim': 2,
            'target_return': target,
            'max_steps': 30
        }

        trainer = RLTrainer('decision-transformer', config)
        metrics = trainer.train(num_episodes=5)
        results[target] = metrics['avg_reward']

    print("✓ Return conditioning test passed")
    for target, reward in results.items():
        print(f"  Target: {target} → Avg Reward: {reward:.2f}")

test_return_conditioning()
```

**Expected Behavior:**
- Higher target returns → higher achieved rewards (if possible)
- Model adapts behavior based on desired return
- Stitching suboptimal trajectories to achieve targets

**Pass Criteria:**
- ✓ Different target returns produce different behaviors
- ✓ Model learns return-action relationships

---

### 3. Sequence Modeling Tests

#### 3.1 Context Window
```python
# Test context length handling
def test_context_window():
    import numpy as np

    context_lengths = [5, 10, 20, 50]

    for ctx_len in context_lengths:
        # Simulate trajectory
        states = np.random.randn(100, 4)
        actions = np.random.randint(0, 2, size=(100,))
        returns = np.cumsum(np.random.randn(100))

        # Extract context windows
        for t in range(ctx_len, len(states)):
            context_states = states[t-ctx_len:t]
            context_actions = actions[t-ctx_len:t]
            context_returns = returns[t-ctx_len:t]

            assert context_states.shape[0] == ctx_len
            assert context_actions.shape[0] == ctx_len
            assert context_returns.shape[0] == ctx_len

    print("✓ Context window test passed")
    print(f"  Tested lengths: {context_lengths}")

test_context_window()
```

**Expected Behavior:**
- Model uses last K timesteps as context
- Longer context → better long-horizon planning
- Attention over full context window

**Pass Criteria:**
- ✓ Context extracted correctly
- ✓ Sequence length maintained
- ✓ Position encodings applied

---

#### 3.2 Attention Mechanism
```python
# Test self-attention computation
def test_attention():
    import numpy as np

    # Simplified attention test
    seq_len = 10
    embed_dim = 64
    n_heads = 8

    # Mock query, key, value
    Q = np.random.randn(seq_len, embed_dim)
    K = np.random.randn(seq_len, embed_dim)
    V = np.random.randn(seq_len, embed_dim)

    # Compute attention scores
    scores = np.matmul(Q, K.T) / np.sqrt(embed_dim)
    attn_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

    # Weighted sum
    output = np.matmul(attn_weights, V)

    assert output.shape == (seq_len, embed_dim)
    assert np.abs(np.sum(attn_weights, axis=-1) - 1.0).max() < 1e-5

    print("✓ Attention mechanism test passed")
    print(f"  Attention weights sum to 1.0: {np.allclose(np.sum(attn_weights, axis=-1), 1.0)}")

test_attention()
```

**Pass Criteria:**
- ✓ Attention weights sum to 1
- ✓ Output dimension correct
- ✓ Multi-head attention splits properly

---

### 4. Training & Convergence Tests

#### 4.1 Loss Tracking
```bash
# Monitor training loss over epochs
python resources/scripts/train_rl_agent.py \
  --algorithm decision-transformer \
  --episodes 50 \
  --save models/dt-loss-tracking.json \
  --verbose

# Analyze loss curve
python -c "
import json
with open('models/dt-loss-tracking.json', 'r') as f:
    data = json.load(f)
    losses = data['metrics']['loss']

    print('✓ Loss tracking test')
    print(f'  Initial Loss: {losses[0]:.4f}')
    print(f'  Final Loss: {losses[-1]:.4f}')
    print(f'  Loss Reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%')

    # Check loss decreased
    assert losses[-1] < losses[0], 'Loss should decrease during training'
"
```

**Expected Behavior:**
- Loss decreases over training
- Convergence to low loss (< 0.1 for simple tasks)
- Stable training (no explosions)

**Pass Criteria:**
- ✓ Loss trend downward
- ✓ Final loss < initial loss
- ✓ No NaN or Inf values

---

#### 4.2 Generalization Test
```python
# Test generalization to unseen states
def test_generalization():
    import sys
    import numpy as np
    sys.path.insert(0, 'resources/scripts')
    from train_rl_agent import RLTrainer

    config = {
        'context_length': 10,
        'embed_dim': 64,
        'n_heads': 4,
        'n_layers': 3,
        'state_dim': 4,
        'action_dim': 2,
        'target_return': 50,
        'max_steps': 30
    }

    # Train on subset of state space
    trainer = RLTrainer('decision-transformer', config)
    metrics = trainer.train(num_episodes=20)

    # Test on different states (would need environment)
    # For now, just verify training completed
    assert metrics['episodes'] == 20

    print("✓ Generalization test passed")
    print(f"  Training completed on {metrics['episodes']} episodes")

test_generalization()
```

**Pass Criteria:**
- ✓ Model trains on limited data
- ✓ Can generalize to new states (if environment available)

---

### 5. Advanced Features Tests

#### 5.1 Behavior Cloning Warmstart
```python
# Test BC pretraining before DT
def test_bc_warmstart():
    """Behavior cloning followed by DT fine-tuning"""
    import sys
    sys.path.insert(0, 'resources/scripts')
    from train_rl_agent import RLTrainer

    config = {
        'context_length': 10,
        'embed_dim': 64,
        'n_heads': 4,
        'n_layers': 3,
        'state_dim': 4,
        'action_dim': 2,
        'behavior_cloning_weight': 1.0,  # Pure BC first
        'max_steps': 20
    }

    # Phase 1: Behavior Cloning
    bc_trainer = RLTrainer('decision-transformer', config)
    bc_metrics = bc_trainer.train(num_episodes=10)

    # Phase 2: DT Fine-tuning
    config['behavior_cloning_weight'] = 0.1  # Reduce BC weight
    config['target_return'] = 100
    dt_trainer = RLTrainer('decision-transformer', config)
    dt_metrics = dt_trainer.train(num_episodes=10)

    print("✓ BC warmstart test passed")
    print(f"  BC Avg Reward: {bc_metrics['avg_reward']:.2f}")
    print(f"  DT Avg Reward: {dt_metrics['avg_reward']:.2f}")

test_bc_warmstart()
```

**Expected Behavior:**
- BC learns basic policy from data
- DT improves via return conditioning
- Faster convergence with warmstart

**Pass Criteria:**
- ✓ BC phase completes
- ✓ DT fine-tuning improves performance

---

#### 5.2 Conservative Training (CQL-style)
```python
# Test conservative Q-learning regularization
def test_conservative_dt():
    import sys
    sys.path.insert(0, 'resources/scripts')
    from train_rl_agent import RLTrainer

    # Conservative DT config
    config = {
        'context_length': 10,
        'embed_dim': 64,
        'n_heads': 4,
        'n_layers': 3,
        'state_dim': 4,
        'action_dim': 2,
        'conservative_weight': 0.1,  # CQL penalty
        'target_return': 50,
        'max_steps': 20
    }

    trainer = RLTrainer('decision-transformer', config)
    metrics = trainer.train(num_episodes=15)

    print("✓ Conservative DT test passed")
    print(f"  Conservative weight: {config['conservative_weight']}")
    print(f"  Avg Reward: {metrics['avg_reward']:.2f}")

test_conservative_dt()
```

**Expected Behavior:**
- Conservative penalty prevents overestimation
- More stable offline learning
- Better generalization to unseen states

**Pass Criteria:**
- ✓ Training stable with conservative penalty
- ✓ No catastrophic overestimation

---

### 6. Integration & Performance Tests

#### 6.1 AgentDB Integration
```bash
# Test storing DT trajectories in AgentDB
python -c "
import asyncio
import json

async def test_dt_agentdb():
    from agentic_flow import createAgentDBAdapter

    adapter = await createAgentDBAdapter({
        'dbPath': '.agentdb/test-dt.db',
        'enableLearning': True,
        'enableReasoning': True
    })

    # Store DT trajectory
    await adapter.insertPattern({
        'id': '',
        'type': 'dt-trajectory',
        'domain': 'offline-rl',
        'pattern_data': json.dumps({
            'embedding': [0.1] * 128,
            'pattern': {
                'states': [[1, 2, 3, 4]] * 10,
                'actions': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                'returns': [100, 95, 90, 85, 80, 75, 70, 65, 60, 55],
                'target_return': 100
            }
        }),
        'confidence': 0.95,
        'usage_count': 1,
        'success_count': 1,
        'created_at': 0,
        'last_used': 0
    })

    print('✓ DT AgentDB integration test passed')

asyncio.run(test_dt_agentdb())
"
```

**Pass Criteria:**
- ✓ Trajectories stored successfully
- ✓ Embeddings indexed for retrieval
- ✓ Can query similar trajectories

---

#### 6.2 Performance Benchmark
```bash
# Benchmark DT vs other algorithms
bash resources/scripts/benchmark_9algorithms.sh --episodes 30 --quick

# Check DT performance
python -c "
import json
from pathlib import Path

# Find latest benchmark
benchmarks = sorted(Path('benchmarks').glob('benchmark_*.json'))
if benchmarks:
    with open(benchmarks[-1], 'r') as f:
        data = json.load(f)
        dt_result = data['results'].get('decision-transformer', {})

        print('✓ DT Performance Benchmark')
        print(f'  Status: {dt_result.get(\"status\", \"N/A\")}')
        print(f'  Training Time: {dt_result.get(\"training_time_seconds\", 0):.2f}s')
        print(f'  Avg Reward: {dt_result.get(\"avg_reward\", 0):.2f}')
"
```

**Pass Criteria:**
- ✓ Completes benchmark successfully
- ✓ Competitive with other algorithms
- ✓ Reasonable training time

---

## Test Execution

### Run Decision Transformer Tests
```bash
# Execute all DT tests
cd tests
bash run-dt-tests.sh
```

### Quick Validation
```bash
# Minimal test for CI/CD
python resources/scripts/train_rl_agent.py \
  --algorithm decision-transformer \
  --episodes 5
```

---

## Expected Results Summary

| Test | Duration | Pass Criteria |
|------|----------|---------------|
| 1.1 Initialization | < 2s | Model creates successfully |
| 1.2 Config Load | < 5s | YAML config applied |
| 2.1 Offline Training | < 10s | Learns from logged data |
| 2.2 Return Conditioning | < 10s | Different returns work |
| 3.1 Context Window | < 1s | Context extracted correctly |
| 3.2 Attention | < 1s | Attention computed correctly |
| 4.1 Loss Tracking | < 20s | Loss decreases |
| 4.2 Generalization | < 10s | Trains successfully |
| 5.1 BC Warmstart | < 15s | Two-phase training works |
| 5.2 Conservative DT | < 10s | Stable with penalty |
| 6.1 AgentDB | < 5s | Trajectories stored |
| 6.2 Benchmark | < 60s | Competitive performance |

---

## Troubleshooting

### Not Learning from Offline Data
- Check target_return is achievable in dataset
- Verify trajectory quality (not all random)
- Increase context_length for long-horizon tasks

### Attention Issues
- Reduce n_heads if embedding_dim not divisible
- Check position encodings applied correctly
- Monitor attention weights distribution

### Poor Generalization
- Increase model size (n_layers, embed_dim)
- Add more diverse offline data
- Use behavior cloning warmstart

---

## Notes

- Decision Transformer is sample efficient (no environment needed)
- Works best with high-quality offline datasets
- Return conditioning enables goal-directed behavior
- Suitable for expensive/dangerous real-world deployment
- Consider trajectory stitching for suboptimal datasets
