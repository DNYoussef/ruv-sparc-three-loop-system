# AgentDB Learning Algorithm Templates

## Overview
Production-ready configuration templates for all 9 reinforcement learning algorithms.

## Available Templates

### 1. q-learning-config.yaml
**Algorithm:** Q-Learning (Value-Based, Off-Policy)

**Use Cases:**
- Grid world navigation
- Board games (Chess, Go)
- Resource allocation
- Discrete decision-making

**Key Parameters:**
```yaml
learning_rate: 0.001
gamma: 0.99
epsilon: 0.1
epsilon_decay: 0.995
```

**Best For:** Simple discrete action spaces, sample efficiency required

---

### 2. actor-critic.json
**Algorithm:** Actor-Critic (Policy Gradient with Baseline)

**Use Cases:**
- Continuous control (robotics)
- Autonomous driving
- Portfolio optimization
- Multi-agent coordination

**Key Parameters:**
```json
{
  "actor_lr": 0.001,
  "critic_lr": 0.002,
  "entropy_coef": 0.01,
  "gae_lambda": 0.95
}
```

**Best For:** Continuous action spaces, variance reduction needed

---

### 3. decision-transformer.yaml
**Algorithm:** Decision Transformer (Offline RL via Sequence Modeling)

**Use Cases:**
- Learning from logged data
- Imitation learning from experts
- Safe RL (no environment interaction)
- Batch RL for expensive simulations

**Key Parameters:**
```yaml
embed_dim: 128
n_heads: 8
n_layers: 6
context_length: 20
target_return: 3600
```

**Best For:** Offline datasets, return-conditioned behavior, long-horizon tasks

---

## Template Structure

All templates follow this structure:

### YAML Templates
```yaml
algorithm: <name>
description: <brief description>

hyperparameters:
  <key>: <value>
  ...

environment:
  state_dim: <int>
  action_dim: <int>
  ...

training:
  num_episodes: <int>
  batch_size: <int>
  ...

agentdb:
  db_path: <path>
  enable_learning: true
  ...

performance:
  use_wasm: true
  ...

use_cases:
  - <use case 1>
  - <use case 2>
  ...

best_practices:
  - <tip 1>
  - <tip 2>
  ...
```

### JSON Templates
```json
{
  "algorithm": "<name>",
  "description": "<brief>",
  "hyperparameters": { ... },
  "environment": { ... },
  "training": { ... },
  "agentdb": { ... },
  "performance": { ... },
  "use_cases": [ ... ],
  "best_practices": [ ... ]
}
```

---

## Usage

### 1. Direct Use with Training Script
```bash
# Use Q-Learning template
python ../scripts/train_rl_agent.py \
  --algorithm q-learning \
  --config q-learning-config.yaml \
  --episodes 100

# Use Actor-Critic template
python ../scripts/train_rl_agent.py \
  --algorithm actor-critic \
  --config actor-critic.json \
  --episodes 200

# Use Decision Transformer template
python ../scripts/train_rl_agent.py \
  --algorithm decision-transformer \
  --config decision-transformer.yaml \
  --episodes 50
```

### 2. Customize Templates
```bash
# Copy and modify
cp q-learning-config.yaml my-custom-config.yaml
nano my-custom-config.yaml

# Use customized config
python ../scripts/train_rl_agent.py \
  --config my-custom-config.yaml \
  --episodes 100
```

### 3. Programmatic Loading
```python
import json
import yaml

# Load JSON template
with open('actor-critic.json', 'r') as f:
    config = json.load(f)

# Load YAML template
with open('q-learning-config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Modify config
config['hyperparameters']['learning_rate'] = 0.0005

# Use in training
from train_rl_agent import RLTrainer
trainer = RLTrainer('q-learning', config)
metrics = trainer.train(num_episodes=100)
```

---

## Template Customization Guide

### Common Modifications

#### 1. Learning Rate Adjustment
```yaml
# Conservative (slower, more stable)
learning_rate: 0.0001

# Moderate (default)
learning_rate: 0.001

# Aggressive (faster, less stable)
learning_rate: 0.01
```

#### 2. Exploration Settings
```yaml
# High exploration (early training)
epsilon: 0.5
epsilon_decay: 0.99

# Balanced
epsilon: 0.1
epsilon_decay: 0.995

# Low exploration (late training/deployment)
epsilon: 0.01
epsilon_decay: 1.0  # No decay
```

#### 3. Network Architecture
```json
{
  "network": {
    "hidden_layers": [64, 64],      // Small
    "hidden_layers": [128, 128],    // Medium (default)
    "hidden_layers": [256, 256, 128] // Large
  }
}
```

#### 4. Training Duration
```yaml
training:
  num_episodes: 100     # Quick test
  num_episodes: 1000    # Standard training
  num_episodes: 10000   # Full training
```

---

## Algorithm Selection Guide

### Choose Q-Learning If:
- Discrete action space
- Small/medium state space
- Sample efficiency important
- Off-policy learning preferred

### Choose Actor-Critic If:
- Continuous action space
- Variance reduction needed
- Stable gradient updates required
- Multi-agent coordination

### Choose Decision Transformer If:
- Offline dataset available
- No environment interaction possible
- Return-conditioned behavior desired
- Long-horizon planning needed

---

## Parameter Tuning Tips

### Learning Rate
- Start with default (0.001)
- Reduce if training unstable
- Increase if convergence too slow
- Monitor loss curves

### Discount Factor (Gamma)
- 0.9: Short-term rewards
- 0.99: Balanced (default)
- 0.999: Very long-term planning

### Batch Size
- 32: Small (less memory, more noise)
- 64: Medium (default)
- 128+: Large (more stable, slower)

### Network Size
- Start small (64 units)
- Increase if underfitting
- Decrease if overfitting
- Monitor validation performance

---

## Template Validation

### Check Template Syntax
```bash
# Validate JSON
python -m json.tool actor-critic.json

# Validate YAML
python -c "import yaml; yaml.safe_load(open('q-learning-config.yaml'))"
```

### Test Template
```bash
# Quick test with template
python ../scripts/train_rl_agent.py \
  --config <template> \
  --episodes 5 \
  --verbose
```

---

## Creating Custom Templates

### Template Checklist
- [ ] Algorithm name specified
- [ ] All required hyperparameters included
- [ ] Environment dimensions defined
- [ ] Training configuration complete
- [ ] AgentDB integration settings
- [ ] Performance options set
- [ ] Use cases documented
- [ ] Best practices listed

### Example: Create SARSA Template
```yaml
algorithm: "sarsa"
description: "On-policy TD learning algorithm"

hyperparameters:
  learning_rate: 0.001
  gamma: 0.99
  epsilon: 0.1
  epsilon_decay: 0.995

environment:
  state_dim: 4
  action_dim: 2
  max_steps: 100

training:
  num_episodes: 1000
  batch_size: 32

agentdb:
  db_path: "./.agentdb/sarsa.db"
  enable_learning: true
  enable_reasoning: true
  cache_size: 1000

performance:
  use_wasm: true
  quantization: false

use_cases:
  - "Safety-critical applications"
  - "Risk-sensitive decision-making"
  - "Online learning with exploration"

best_practices:
  - "More conservative than Q-Learning"
  - "Better for environments with penalties"
  - "Use for safe exploration"
```

---

## Advanced Templates

### Multi-Task Learning Template
```json
{
  "algorithm": "multi-task",
  "tasks": [
    {"name": "task1", "weight": 0.5},
    {"name": "task2", "weight": 0.3},
    {"name": "task3", "weight": 0.2}
  ],
  "shared_layers": [128, 128],
  "task_specific_layers": [64],
  "transfer_learning": true
}
```

### Curriculum Learning Template
```yaml
algorithm: "curriculum"
stages:
  - name: "easy"
    episodes: 100
    difficulty: 0.2
  - name: "medium"
    episodes: 200
    difficulty: 0.5
  - name: "hard"
    episodes: 300
    difficulty: 1.0
progression_metric: "success_rate"
progression_threshold: 0.8
```

---

## Template Versioning

Templates follow semantic versioning:
- `v1.0.0` - Initial release
- `v1.1.0` - Add new parameters (backward compatible)
- `v2.0.0` - Breaking changes

Check template version:
```yaml
version: "1.0.0"
last_updated: "2025-01-02"
```

---

## Support

- **Examples**: See `../../examples/` for usage examples
- **Tests**: See `../../tests/` for validation
- **Scripts**: See `../scripts/` for training tools
- **Docs**: See `../../README.md` for skill overview

---

## Contributing

To add new templates:
1. Follow template structure above
2. Include all required sections
3. Add validation tests
4. Document use cases
5. Submit pull request

---

## License

MIT License - See LICENSE file for details
