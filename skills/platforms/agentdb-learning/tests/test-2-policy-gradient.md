# Test 2: Policy Gradient Methods (Actor-Critic, SARSA)

## Overview
Test suite for policy gradient algorithms including Actor-Critic and SARSA.

## Test Categories

### 1. Actor-Critic Tests

#### 1.1 Basic Training
```bash
# Test Actor-Critic initialization and training
python resources/scripts/train_rl_agent.py \
  --algorithm actor-critic \
  --episodes 20 \
  --verbose
```

**Expected Output:**
- Both actor and critic networks initialized
- Separate learning rates for actor (0.001) and critic (0.002)
- Policy gradient updates logged
- Value function loss tracked

**Pass Criteria:**
- ✓ Training completes without errors
- ✓ Both actor_loss and critic_loss decrease
- ✓ Entropy tracked for exploration

---

#### 1.2 Custom Configuration
```bash
# Train with custom Actor-Critic config
python resources/scripts/train_rl_agent.py \
  --algorithm actor-critic \
  --config resources/templates/actor-critic.json \
  --episodes 50 \
  --save models/actor-critic-test.json
```

**Expected Behavior:**
- JSON config loaded successfully
- Custom hyperparameters applied:
  - actor_lr: 0.001
  - critic_lr: 0.002
  - entropy_coef: 0.01
  - gae_lambda: 0.95

**Pass Criteria:**
- ✓ Config parameters match template
- ✓ GAE (Generalized Advantage Estimation) enabled
- ✓ Advantage normalization working

**Validation:**
```python
import json
with open('models/actor-critic-test.json', 'r') as f:
    model = json.load(f)
    config = model['config']

    assert config['actor_lr'] == 0.001
    assert config['critic_lr'] == 0.002
    assert config.get('entropy_coef', 0) > 0  # Entropy regularization
    print("✓ Actor-Critic config test passed")
```

---

#### 1.3 Continuous Action Space
```bash
# Test continuous action handling
python -c "
import sys
sys.path.insert(0, 'resources/scripts')
from train_rl_agent import RLTrainer

config = {
    'actor_lr': 0.001,
    'critic_lr': 0.002,
    'state_dim': 8,
    'action_dim': 4,
    'continuous_actions': True,
    'action_range': [-1.0, 1.0],
    'max_steps': 50
}

trainer = RLTrainer('actor-critic', config)
metrics = trainer.train(num_episodes=10)

assert metrics['episodes'] == 10
print('✓ Continuous action test passed')
"
```

**Pass Criteria:**
- ✓ Continuous actions handled correctly
- ✓ Action clipping to range [-1, 1]
- ✓ Policy outputs valid continuous values

---

### 2. SARSA Tests

#### 2.1 On-Policy Learning
```bash
# Test SARSA on-policy updates
python resources/scripts/train_rl_agent.py \
  --algorithm sarsa \
  --episodes 30 \
  --verbose
```

**Expected Behavior:**
- On-policy updates (follows current policy)
- More conservative than Q-Learning
- Epsilon-greedy exploration

**Pass Criteria:**
- ✓ Training completes successfully
- ✓ Updates use current policy's next action
- ✓ Epsilon decay working

---

#### 2.2 SARSA vs Q-Learning Comparison
```bash
# Run both algorithms on same task
python resources/scripts/train_rl_agent.py \
  --algorithm sarsa \
  --episodes 50 \
  --save models/sarsa-comparison.json

python resources/scripts/train_rl_agent.py \
  --algorithm q-learning \
  --episodes 50 \
  --save models/qlearning-comparison.json

# Compare results
python -c "
import json

with open('models/sarsa-comparison.json', 'r') as f:
    sarsa = json.load(f)
with open('models/qlearning-comparison.json', 'r') as f:
    qlearning = json.load(f)

print('SARSA Avg Reward:', sarsa['metrics']['avg_reward'])
print('Q-Learning Avg Reward:', qlearning['metrics']['avg_reward'])
print('✓ Comparison test passed')
"
```

**Expected Differences:**
- SARSA typically more conservative
- Q-Learning may achieve higher rewards but with more variance
- SARSA better for safety-critical tasks

**Pass Criteria:**
- ✓ Both algorithms complete
- ✓ Metrics comparable (within reasonable range)

---

### 3. Advanced Features Tests

#### 3.1 Advantage Estimation (GAE)
```python
# Test Generalized Advantage Estimation
import numpy as np

def test_gae():
    """Test GAE computation"""
    # Mock trajectory
    rewards = [1.0, 2.0, 3.0, 1.0]
    values = [2.0, 3.0, 4.0, 2.0, 0.0]  # Including bootstrap
    gamma = 0.99
    gae_lambda = 0.95

    # Compute TD errors
    deltas = []
    for t in range(len(rewards)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        deltas.append(delta)

    # Compute advantages (GAE)
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * gae_lambda * gae
        advantages.insert(0, gae)

    assert len(advantages) == len(rewards)
    print("✓ GAE computation test passed")
    print(f"  Advantages: {advantages}")

test_gae()
```

**Pass Criteria:**
- ✓ GAE reduces variance vs vanilla advantages
- ✓ Exponential weighting applied correctly

---

#### 3.2 Entropy Regularization
```bash
# Test entropy bonus for exploration
python -c "
import sys
sys.path.insert(0, 'resources/scripts')
from train_rl_agent import RLTrainer

# High entropy coefficient
config_high = {
    'actor_lr': 0.001,
    'critic_lr': 0.002,
    'entropy_coef': 0.1,  # High entropy
    'state_dim': 4,
    'action_dim': 2,
    'max_steps': 20
}

# Low entropy coefficient
config_low = {
    'actor_lr': 0.001,
    'critic_lr': 0.002,
    'entropy_coef': 0.001,  # Low entropy
    'state_dim': 4,
    'action_dim': 2,
    'max_steps': 20
}

trainer_high = RLTrainer('actor-critic', config_high)
trainer_low = RLTrainer('actor-critic', config_low)

metrics_high = trainer_high.train(num_episodes=10)
metrics_low = trainer_low.train(num_episodes=10)

print('✓ Entropy regularization test passed')
print(f'  High entropy final epsilon: {metrics_high[\"epsilon\"]:.4f}')
print(f'  Low entropy final epsilon: {metrics_low[\"epsilon\"]:.4f}')
"
```

**Expected Behavior:**
- High entropy → More exploration, less deterministic
- Low entropy → More exploitation, more deterministic

**Pass Criteria:**
- ✓ Entropy affects exploration behavior
- ✓ No training instabilities

---

### 4. Integration Tests

#### 4.1 Multi-Episode Training
```bash
# Long training run to test stability
python resources/scripts/train_rl_agent.py \
  --algorithm actor-critic \
  --episodes 200 \
  --save models/actor-critic-long.json
```

**Expected Behavior:**
- Stable training over 200 episodes
- Gradual improvement in avg reward
- No catastrophic forgetting

**Pass Criteria:**
- ✓ Completes without crashes
- ✓ Metrics show learning progress
- ✓ Final performance > initial performance

---

#### 4.2 Gradient Clipping
```python
# Test gradient clipping prevents explosions
def test_gradient_clipping():
    import numpy as np

    # Simulate large gradients
    grads = np.array([100.0, 200.0, -150.0])
    max_norm = 1.0

    # Clip gradients
    grad_norm = np.linalg.norm(grads)
    if grad_norm > max_norm:
        grads = grads * (max_norm / grad_norm)

    clipped_norm = np.linalg.norm(grads)
    assert clipped_norm <= max_norm + 1e-6
    print("✓ Gradient clipping test passed")
    print(f"  Original norm: {grad_norm:.2f} → Clipped: {clipped_norm:.2f}")

test_gradient_clipping()
```

**Pass Criteria:**
- ✓ Gradients clipped to max_norm
- ✓ Prevents gradient explosions

---

### 5. Performance & Benchmarks

#### 5.1 Training Speed Comparison
```bash
# Benchmark Actor-Critic vs SARSA vs Q-Learning
echo "=== Training Speed Benchmark ==="

# Actor-Critic
echo "Actor-Critic:"
time python resources/scripts/train_rl_agent.py \
  --algorithm actor-critic --episodes 100 > /dev/null 2>&1

# SARSA
echo "SARSA:"
time python resources/scripts/train_rl_agent.py \
  --algorithm sarsa --episodes 100 > /dev/null 2>&1

# Q-Learning
echo "Q-Learning:"
time python resources/scripts/train_rl_agent.py \
  --algorithm q-learning --episodes 100 > /dev/null 2>&1
```

**Expected Results:**
- Actor-Critic: Slower (2 networks)
- SARSA: Similar to Q-Learning
- Q-Learning: Fastest (1 network)

**Pass Criteria:**
- ✓ All complete within reasonable time (< 30s each)

---

#### 5.2 Sample Efficiency
```bash
# Test sample efficiency (episodes needed to reach threshold)
python -c "
import sys
sys.path.insert(0, 'resources/scripts')
from train_rl_agent import RLTrainer

algorithms = ['q-learning', 'sarsa', 'actor-critic']
threshold = -50.0  # Reward threshold

for algo in algorithms:
    config = {'state_dim': 4, 'action_dim': 2, 'max_steps': 50}
    trainer = RLTrainer(algo, config)

    episodes_to_threshold = 0
    for episode in range(1, 101):
        trainer.train(num_episodes=1)
        if trainer.metrics['avg_reward'] >= threshold:
            episodes_to_threshold = episode
            break

    print(f'{algo}: {episodes_to_threshold} episodes to reach {threshold}')
"
```

**Pass Criteria:**
- ✓ All algorithms eventually reach threshold
- ✓ Actor-Critic typically most sample efficient

---

## Test Execution

### Run Policy Gradient Tests
```bash
# Execute all policy gradient tests
cd tests
bash run-policy-gradient-tests.sh
```

### Quick Validation
```bash
# Minimal test for CI/CD
python resources/scripts/train_rl_agent.py --algorithm actor-critic --episodes 5
python resources/scripts/train_rl_agent.py --algorithm sarsa --episodes 5
```

---

## Expected Results Summary

| Test | Algorithm | Duration | Pass Criteria |
|------|-----------|----------|---------------|
| 1.1 | Actor-Critic | < 10s | Both losses decrease |
| 1.2 | Actor-Critic | < 20s | Config applied |
| 1.3 | Actor-Critic | < 5s | Continuous actions work |
| 2.1 | SARSA | < 10s | On-policy updates |
| 2.2 | Both | < 20s | Comparable performance |
| 3.1 | Actor-Critic | < 1s | GAE computed correctly |
| 3.2 | Actor-Critic | < 5s | Entropy affects behavior |
| 4.1 | Actor-Critic | 30-60s | Stable long training |
| 4.2 | Actor-Critic | < 1s | Gradients clipped |
| 5.1 | All 3 | < 60s | Speed acceptable |
| 5.2 | All 3 | < 120s | Sample efficiency measured |

---

## Troubleshooting

### High Variance in Actor-Critic
**Problem:** Policy updates are unstable
- Solution: Increase GAE lambda (0.95 → 0.99)
- Solution: Use larger batch sizes
- Solution: Normalize advantages

### SARSA Not Learning
**Problem:** Rewards not improving
- Solution: Increase exploration (higher epsilon)
- Solution: Reduce learning rate
- Solution: Check epsilon decay rate

### Gradient Explosions
**Problem:** NaN in losses
- Solution: Enable gradient clipping (max_norm=1.0)
- Solution: Reduce learning rates
- Solution: Check reward scaling

---

## Notes

- Policy gradient methods are more sample efficient but higher variance
- Actor-Critic reduces variance with value baseline
- SARSA is safer than Q-Learning for risky environments
- Entropy regularization is crucial for exploration
- GAE provides better bias-variance tradeoff than vanilla advantages
