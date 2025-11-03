# AgentDB Learning Scripts

## Overview
Production-ready training scripts for all 9 AgentDB reinforcement learning algorithms.

## Scripts

### 1. train_rl_agent.py
Primary training script supporting all algorithms.

**Usage:**
```bash
python train_rl_agent.py --algorithm <name> --episodes <N> [options]
```

**Supported Algorithms:**
- `q-learning` - Value-based (off-policy)
- `sarsa` - Value-based (on-policy)
- `actor-critic` - Policy gradient with baseline
- `decision-transformer` - Offline RL via sequence modeling
- `active-learning` - Query-based learning
- `adversarial` - Robustness training
- `curriculum` - Progressive difficulty
- `federated` - Distributed learning
- `multi-task` - Transfer learning

**Options:**
- `--episodes, -e` - Number of training episodes (default: 100)
- `--config, -c` - Path to config file (JSON/YAML)
- `--save, -s` - Save trained model to path
- `--verbose, -v` - Enable verbose logging

**Examples:**
```bash
# Basic training
python train_rl_agent.py --algorithm q-learning --episodes 100

# With custom config
python train_rl_agent.py -a actor-critic -c ../templates/actor-critic.json -e 200

# Train and save
python train_rl_agent.py -a decision-transformer -e 50 -s models/dt-model.json
```

---

### 2. benchmark_9algorithms.sh
Comprehensive benchmark comparing all 9 algorithms.

**Usage:**
```bash
bash benchmark_9algorithms.sh [options]
```

**Options:**
- `-e, --episodes N` - Episodes per algorithm (default: 50)
- `-q, --quick` - Quick mode (10 episodes)
- `-o, --output DIR` - Output directory (default: ./benchmarks)
- `-h, --help` - Show help

**Examples:**
```bash
# Full benchmark
bash benchmark_9algorithms.sh --episodes 100

# Quick validation
bash benchmark_9algorithms.sh --quick

# Custom output
bash benchmark_9algorithms.sh -e 50 -o results/
```

**Output:**
- `benchmark_TIMESTAMP.json` - Results in JSON format
- `*_output_TIMESTAMP.log` - Individual algorithm logs
- `*_model_TIMESTAMP.json` - Trained models

**Results Format:**
```json
{
  "timestamp": "20250102_143022",
  "episodes": 50,
  "results": {
    "q-learning": {
      "status": "success",
      "training_time_seconds": 12.34,
      "avg_reward": 45.67,
      "final_loss": 0.023,
      "model_file": "..."
    },
    ...
  }
}
```

---

### 3. test_learning.py
Comprehensive test suite for all algorithms.

**Usage:**
```bash
# Run all tests
python test_learning.py

# Run specific test class
python test_learning.py TestRLAlgorithms

# Run single test
python test_learning.py TestRLAlgorithms.test_q_learning

# Verbose output
python -m unittest test_learning.py -v
```

**Test Categories:**
1. **TestRLAlgorithms** - All 9 algorithm tests
2. **TestConfiguration** - Config loading & validation
3. **TestMetrics** - Training metrics tracking
4. **TestIntegration** - End-to-end workflows

**Example Output:**
```
Running tests...
test_q_learning ... ok
test_sarsa ... ok
test_actor_critic ... ok
test_decision_transformer ... ok
...
----------------------------------------------------------------------
Ran 15 tests in 12.345s

OK (successes=15)
```

---

## Installation

### Prerequisites
```bash
# Python 3.8+
python --version

# Install dependencies
pip install numpy

# Optional: YAML support
pip install pyyaml

# Optional: AgentDB integration
npm install agentdb@latest
```

### Setup
```bash
# Make scripts executable
chmod +x train_rl_agent.py
chmod +x benchmark_9algorithms.sh
chmod +x test_learning.py

# Create output directories
mkdir -p models benchmarks logs
```

---

## Quick Start

### 1. Train Single Algorithm
```bash
# Train Q-Learning for 50 episodes
python train_rl_agent.py --algorithm q-learning --episodes 50
```

### 2. Run Benchmark
```bash
# Compare all algorithms (quick mode)
bash benchmark_9algorithms.sh --quick
```

### 3. Validate Installation
```bash
# Run test suite
python test_learning.py
```

---

## Configuration Files

Scripts support both JSON and YAML configs:

### JSON Example (`config.json`)
```json
{
  "algorithm": "q-learning",
  "learning_rate": 0.001,
  "gamma": 0.99,
  "epsilon": 0.1,
  "state_dim": 4,
  "action_dim": 2
}
```

### YAML Example (`config.yaml`)
```yaml
algorithm: actor-critic
hyperparameters:
  actor_lr: 0.001
  critic_lr: 0.002
  gamma: 0.99
environment:
  state_dim: 8
  action_dim: 4
```

---

## Performance Tips

### Speed Optimization
- Use `--episodes 10` for quick testing
- Enable WASM acceleration (automatic in AgentDB)
- Reduce `state_dim` and `action_dim` for faster convergence

### Memory Optimization
- Reduce `context_length` for Decision Transformer
- Use smaller `batch_size` if memory limited
- Enable model quantization (if supported)

### Training Stability
- Start with low `learning_rate` (0.0001)
- Use gradient clipping for policy gradients
- Monitor loss curves for divergence

---

## Troubleshooting

### Import Errors
```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify script location
ls -la train_rl_agent.py
```

### Config Not Loading
```bash
# Validate JSON
python -m json.tool config.json

# Validate YAML
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### Training Crashes
- Check available memory: `free -h`
- Reduce episode count or batch size
- Enable verbose mode: `--verbose`

---

## Advanced Usage

### Custom Algorithm Config
```bash
# Create custom config
cat > my_config.yaml << EOF
algorithm: actor-critic
hyperparameters:
  actor_lr: 0.0005
  critic_lr: 0.001
  entropy_coef: 0.05
training:
  num_episodes: 200
  batch_size: 128
EOF

# Train with custom config
python train_rl_agent.py -a actor-critic -c my_config.yaml
```

### Parallel Training
```bash
# Train multiple algorithms in parallel
for algo in q-learning sarsa actor-critic; do
  python train_rl_agent.py -a $algo -e 100 &
done
wait
```

### Automated Testing
```bash
# Run tests in CI/CD pipeline
python test_learning.py 2>&1 | tee test_results.log
if [ $? -eq 0 ]; then
  echo "All tests passed"
else
  echo "Tests failed"
  exit 1
fi
```

---

## Integration with AgentDB

### Store Training Data
```python
from train_rl_agent import RLTrainer
from agentic_flow import createAgentDBAdapter

# Train model
trainer = RLTrainer('q-learning', config)
metrics = trainer.train(num_episodes=100)

# Store in AgentDB
adapter = await createAgentDBAdapter({
    'dbPath': '.agentdb/training.db',
    'enableLearning': True
})

await adapter.insertPattern({
    'type': 'training-run',
    'domain': 'q-learning',
    'pattern_data': JSON.stringify(metrics),
    'confidence': 0.9
})
```

---

## Support

- **Documentation**: See `../README.md` for skill overview
- **Templates**: See `../templates/` for configuration examples
- **Tests**: See `../../tests/` for validation examples
- **GitHub**: https://github.com/ruvnet/agentic-flow

---

## License

MIT License - See LICENSE file for details
