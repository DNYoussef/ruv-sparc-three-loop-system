# Test 1: Q-Learning Algorithm

## Overview
Comprehensive test suite for Q-Learning implementation in AgentDB.

## Test Categories

### 1. Basic Functionality Tests

#### 1.1 Algorithm Initialization
```bash
# Test basic Q-Learning agent creation
python resources/scripts/train_rl_agent.py \
  --algorithm q-learning \
  --episodes 10 \
  --verbose
```

**Expected Output:**
- Agent initializes successfully
- Epsilon starts at 0.1
- Learning rate set to default (0.001)
- Training completes without errors

**Pass Criteria:**
- ✓ Exit code 0
- ✓ 10 episodes completed
- ✓ Metrics reported (avg reward, epsilon, loss)

---

#### 1.2 Configuration Loading
```bash
# Test with custom YAML config
python resources/scripts/train_rl_agent.py \
  --algorithm q-learning \
  --config resources/templates/q-learning-config.yaml \
  --episodes 20
```

**Expected Output:**
- Config loaded from YAML
- Hyperparameters match template
- Training uses custom settings

**Pass Criteria:**
- ✓ Config parameters applied correctly
- ✓ Learning rate = 0.001 (from config)
- ✓ Gamma = 0.99 (from config)

---

### 2. Performance Tests

#### 2.1 Convergence Test
```bash
# Test convergence over 100 episodes
python resources/scripts/train_rl_agent.py \
  --algorithm q-learning \
  --episodes 100 \
  --save models/q-learning-convergence.json
```

**Expected Behavior:**
- Average reward increases over time
- Loss decreases
- Epsilon decays from 0.1 to ~0.60 (with decay 0.995)

**Pass Criteria:**
- ✓ Final avg reward > initial avg reward
- ✓ Epsilon decayed appropriately
- ✓ Model saved successfully

**Metrics to Check:**
```python
# Load saved model
import json
with open('models/q-learning-convergence.json', 'r') as f:
    data = json.load(f)

assert data['metrics']['episodes'] == 100
assert data['metrics']['avg_reward'] > -1000  # Not catastrophically bad
assert data['metrics']['epsilon'] < 0.1       # Decayed from initial
```

---

#### 2.2 Exploration vs Exploitation
```bash
# Test high epsilon (more exploration)
python resources/scripts/train_rl_agent.py \
  --algorithm q-learning \
  --episodes 50 \
  --config <(cat resources/templates/q-learning-config.yaml | sed 's/epsilon: 0.1/epsilon: 0.5/')

# Test low epsilon (more exploitation)
python resources/scripts/train_rl_agent.py \
  --algorithm q-learning \
  --episodes 50 \
  --config <(cat resources/templates/q-learning-config.yaml | sed 's/epsilon: 0.1/epsilon: 0.01/')
```

**Expected Behavior:**
- High epsilon: More varied actions, slower convergence
- Low epsilon: Faster convergence to suboptimal policy

**Pass Criteria:**
- ✓ Both configurations complete successfully
- ✓ Observable difference in training behavior

---

### 3. Integration Tests

#### 3.1 AgentDB Integration
```python
# Test AgentDB experience storage
from agentic_flow import createAgentDBAdapter

async def test_agentdb_qlearning():
    adapter = await createAgentDBAdapter({
        'dbPath': '.agentdb/test-qlearning.db',
        'enableLearning': True,
        'enableReasoning': True
    })

    # Store Q-Learning experience
    await adapter.insertPattern({
        'id': '',
        'type': 'q-learning-experience',
        'domain': 'test',
        'pattern_data': json.dumps({
            'embedding': [0.1, 0.2, 0.3, 0.4],
            'pattern': {
                'state': [1, 2, 3, 4],
                'action': 1,
                'reward': 10.0,
                'next_state': [1.1, 2.1, 3.1, 4.1],
                'done': False
            }
        }),
        'confidence': 0.9,
        'usage_count': 1,
        'success_count': 1,
        'created_at': Date.now(),
        'last_used': Date.now()
    })

    # Retrieve similar experiences
    result = await adapter.retrieveWithReasoning(
        [0.1, 0.2, 0.3, 0.4],
        {'domain': 'test', 'k': 5}
    )

    assert len(result.memories) > 0
    print("✓ AgentDB integration test passed")
```

**Pass Criteria:**
- ✓ Experience stored successfully
- ✓ Similar experiences retrieved
- ✓ No database errors

---

#### 3.2 Model Save/Load
```bash
# Train and save
python resources/scripts/train_rl_agent.py \
  --algorithm q-learning \
  --episodes 30 \
  --save models/test-qlearning.json

# Verify saved model
test -f models/test-qlearning.json || echo "FAIL: Model not saved"

# Check model contents
python -c "
import json
with open('models/test-qlearning.json', 'r') as f:
    data = json.load(f)
    assert data['algorithm'] == 'q-learning'
    assert 'metrics' in data
    assert data['metrics']['episodes'] == 30
    print('✓ Model save/load test passed')
"
```

**Pass Criteria:**
- ✓ Model file created
- ✓ JSON structure valid
- ✓ All metrics present

---

### 4. Edge Cases & Error Handling

#### 4.1 Invalid Configuration
```bash
# Test with invalid epsilon (negative)
python resources/scripts/train_rl_agent.py \
  --algorithm q-learning \
  --episodes 5 \
  --config <(echo "{epsilon: -0.5}") 2>&1 | grep -q "ERROR"

# Should fail gracefully
echo $? -eq 0 && echo "✓ Error handling test passed" || echo "FAIL: Should have errored"
```

**Pass Criteria:**
- ✓ Invalid config rejected
- ✓ Helpful error message displayed

---

#### 4.2 Zero Episodes
```bash
# Test training with 0 episodes
python resources/scripts/train_rl_agent.py \
  --algorithm q-learning \
  --episodes 0

# Should complete immediately
```

**Pass Criteria:**
- ✓ No errors
- ✓ Zero episodes reported in metrics

---

### 5. Performance Benchmarks

#### 5.1 Training Speed
```bash
# Benchmark 100 episodes
time python resources/scripts/train_rl_agent.py \
  --algorithm q-learning \
  --episodes 100 \
  > /dev/null 2>&1

# Should complete in < 10 seconds
```

**Pass Criteria:**
- ✓ Completes in reasonable time (< 10s for 100 episodes)
- ✓ No memory leaks

---

#### 5.2 Memory Usage
```bash
# Monitor memory during training
/usr/bin/time -v python resources/scripts/train_rl_agent.py \
  --algorithm q-learning \
  --episodes 500 2>&1 | grep "Maximum resident set size"

# Should use < 500MB
```

**Pass Criteria:**
- ✓ Memory usage reasonable (< 500MB)
- ✓ No excessive memory growth

---

## Test Execution

### Run All Tests
```bash
# Execute complete test suite
cd tests
bash run-all-tests.sh test-1-q-learning.md
```

### Quick Test (Smoke Test)
```bash
# Run minimal test to verify basic functionality
python resources/scripts/train_rl_agent.py \
  --algorithm q-learning \
  --episodes 5
```

---

## Expected Results Summary

| Test | Expected Duration | Pass Criteria |
|------|------------------|---------------|
| 1.1 Basic Init | < 5s | Exit 0, metrics present |
| 1.2 Config Load | < 5s | Config applied correctly |
| 2.1 Convergence | 10-30s | Avg reward improves |
| 2.2 Exploration | 10-20s | Both configs work |
| 3.1 AgentDB | < 10s | DB operations succeed |
| 3.2 Save/Load | < 10s | Model persists correctly |
| 4.1 Invalid Config | < 2s | Error caught gracefully |
| 4.2 Zero Episodes | < 1s | Handles edge case |
| 5.1 Speed | < 10s | Fast enough for CI/CD |
| 5.2 Memory | < 30s | No memory leaks |

---

## Troubleshooting

### Test Failures

**Problem: Training doesn't converge**
- Solution: Increase episodes to 200+
- Check: Learning rate not too high/low

**Problem: AgentDB connection fails**
- Solution: Verify AgentDB installed (`npm list agentdb`)
- Check: Database path writable

**Problem: Model save fails**
- Solution: Create `models/` directory
- Check: Disk space available

---

## Notes

- All tests should be automated in CI/CD pipeline
- Tests should be idempotent (can run multiple times)
- Use temporary directories for test artifacts
- Clean up generated files after tests
