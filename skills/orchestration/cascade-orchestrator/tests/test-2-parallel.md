# Test 2: Parallel Execution with Swarm Coordination

## Test Objective
Validate parallel stage execution with dependency management, swarm coordination, and result aggregation.

## Test Scenario
Execute a workflow with parallel code quality checks:
1. Setup phase (sequential)
2. Parallel quality checks (4 concurrent tasks)
   - Linting
   - Security scanning
   - Complexity analysis
   - Test coverage
3. Aggregation phase (sequential)

## Prerequisites
- Parallel executor script installed
- Sample codebase for analysis
- Swarm coordination (ruv-swarm MCP) available

## Test Steps

### 1. Prepare Test Environment

```bash
# Create test directory
mkdir -p /tmp/cascade-test-parallel
cd /tmp/cascade-test-parallel

# Create sample codebase
mkdir -p src tests

cat > src/app.py << 'EOF'
def calculate(a, b):
    """Calculate sum of two numbers"""
    return a + b

def divide(a, b):
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b
EOF

cat > tests/test_app.py << 'EOF'
import pytest
from src.app import calculate, divide

def test_calculate():
    assert calculate(2, 3) == 5

def test_divide():
    assert divide(10, 2) == 5

def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(10, 0)
EOF
```

### 2. Execute Parallel Workflow

```bash
# Run parallel executor with quality checks
python3 ../resources/scripts/parallel_exec.py \
  test_parallel_quality_checks \
  --max-workers 4 \
  --verbose \
  --output results.json
```

### 3. Verify Parallel Execution

```bash
# Check that tasks ran concurrently
cat results.json | jq '.stages |
  [.[] | select(.status == "COMPLETED")] |
  length'
# Expected: 4 (all quality checks)

# Verify timing (parallel should be faster than sequential)
cat results.json | jq '
  .stages |
  to_entries[] |
  {task: .key, duration: .value.duration}'

# Calculate total vs. sum of individual durations
cat results.json | jq '
  {
    total_duration: .duration,
    sum_of_stages: (.stages | [.[].duration] | add),
    speedup: ((.stages | [.[].duration] | add) / .duration)
  }'
# Expected speedup: 2.5-3.5x
```

### 4. Verify Dependency Management

```bash
# Create workflow with dependencies
cat > workflow_with_deps.json << 'EOF'
{
  "tasks": [
    {"id": "task_a", "dependencies": []},
    {"id": "task_b", "dependencies": []},
    {"id": "task_c", "dependencies": ["task_a", "task_b"]},
    {"id": "task_d", "dependencies": ["task_c"]}
  ]
}
EOF

# Run with dependencies
python3 ../resources/scripts/parallel_exec.py \
  workflow_with_deps.json \
  --verbose

# Verify execution order
cat results.json | jq '.stages |
  to_entries[] |
  {task: .key, started: .value.start_time} |
  sort_by(.started)'
# Expected: task_a and task_b start first, then task_c, then task_d
```

### 5. Test Swarm Coordination

```bash
# Create cascade with swarm-parallel stage
cat > swarm_cascade.yaml << 'EOF'
stages:
  - stage_id: quality_checks
    type: swarm-parallel
    skills:
      - lint-code
      - security-scan
      - complexity-analysis
      - test-coverage
    swarm_config:
      topology: mesh
      max_agents: 4
      strategy: balanced
      memory_shared: true
EOF

# Execute with swarm
python3 ../resources/scripts/workflow_executor.py \
  swarm_cascade.yaml \
  --verbose
```

## Expected Results

### Parallel Execution
- ✓ All 4 quality checks start simultaneously
- ✓ Execution time ≈ max(individual durations), not sum
- ✓ Speedup factor 2.5-3.5x compared to sequential
- ✓ CPU utilization increases (multiple cores used)

### Dependency Management
- ✓ Tasks with no dependencies start immediately
- ✓ Dependent tasks wait for prerequisites
- ✓ No task starts before its dependencies complete
- ✓ Circular dependencies detected and rejected

### Swarm Coordination
- ✓ Swarm initialized with correct topology (mesh)
- ✓ 4 agents spawned
- ✓ Tasks distributed across agents
- ✓ Load balancing achieved
- ✓ Memory shared between agents

### Result Aggregation
- ✓ All parallel results collected
- ✓ No results lost
- ✓ Aggregated output contains all task outputs
- ✓ Failed tasks marked correctly

## Success Criteria

- [ ] All 4 quality checks complete successfully
- [ ] Parallel execution faster than sequential (>2x speedup)
- [ ] Dependencies respected (no premature execution)
- [ ] Swarm coordination functional
- [ ] No deadlocks or race conditions
- [ ] Results properly aggregated

## Failure Scenarios to Test

### 1. Failed Task in Parallel Group

```python
# Inject failure into one task
def failing_task():
    raise Exception("Simulated failure")

executor.add_task(ParallelTask("fail_task", "Failing Task", failing_task))
```

**Expected:**
- Failed task marked as FAILED
- Other parallel tasks complete successfully
- Dependent tasks on failed task are CANCELLED
- Overall cascade status: partial failure

### 2. Circular Dependency

```python
# Create circular dependency
executor.add_task(ParallelTask("task_a", "Task A", lambda: None, dependencies={"task_b"}))
executor.add_task(ParallelTask("task_b", "Task B", lambda: None, dependencies={"task_a"}))
```

**Expected:**
- Executor detects cycle before execution
- Raises ValueError with "Circular dependency detected"
- No tasks executed

### 3. Timeout in Parallel Task

```python
import time

def slow_task():
    time.sleep(10)  # Exceeds timeout

executor.add_task(ParallelTask(
    "slow_task",
    "Slow Task",
    slow_task,
    timeout=2.0  # 2 second timeout
))
```

**Expected:**
- Task times out after 2 seconds
- Task status: FAILED
- Error: "Task execution exceeded 2.0s"
- Other tasks unaffected

## Performance Benchmarks

### Parallel Execution (4 tasks, each 0.5s)
- **Sequential:** 2.0s (0.5 + 0.5 + 0.5 + 0.5)
- **Parallel:** 0.5-0.6s (max of all + overhead)
- **Speedup:** 3.3-4.0x

### Swarm Coordination Overhead
- Swarm init: <0.2s
- Agent spawn (4 agents): <0.3s
- Memory sync: <0.1s per operation
- **Total overhead:** <0.6s

### Dependency Resolution
- Graph construction: <0.01s for 100 tasks
- Cycle detection: <0.05s for 100 tasks
- Ready task lookup: <0.001s per iteration

## Resource Usage

### CPU
- Utilization: 80-100% across available cores
- Context switches: Minimal with ThreadPoolExecutor
- CPU time: ~2x wall time for 4 parallel tasks

### Memory
- Base memory: ~30MB
- Per task overhead: ~2MB
- Peak memory: <50MB for 4 concurrent tasks
- Memory growth: Linear with task count

### Network (Swarm)
- Swarm coordination: ~10KB/s
- Memory sync: ~5KB per operation
- Total network: <100KB for typical workflow

## Clean Up

```bash
# Remove test artifacts
rm -rf /tmp/cascade-test-parallel
```

## Test Report Template

```markdown
## Parallel Execution Test Results

**Date:** YYYY-MM-DD
**Executor Version:** 2.0.0
**Test Duration:** X.XXs

### Results
- ✓ Parallel execution functional
- ✓ Dependency management correct
- ✓ Swarm coordination working
- ✓ Result aggregation successful
- ✓ Performance meets benchmarks

### Metrics
- Total tasks: 4
- Successful: 4
- Failed: 0
- Cancelled: 0
- Sequential time: 2.0s (estimated)
- Parallel time: 0.6s
- **Speedup: 3.3x**

### Swarm Metrics
- Topology: mesh
- Agents: 4
- Memory shared: Yes
- Load balance: Good (25% each)

### Issues Found
None

### Recommendations
- Performance excellent for production
- Consider increasing max_workers for larger workflows
- Monitor memory usage with >10 concurrent tasks
```

## Integration with CI/CD

```yaml
# .github/workflows/test-cascade-parallel.yml
name: Test Parallel Cascade

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run parallel workflow test
        run: |
          bash skills/cascade-orchestrator/tests/test-2-parallel.md

      - name: Verify speedup
        run: |
          speedup=$(cat results.json | jq '.speedup')
          echo "Speedup: $speedup"
          if (( $(echo "$speedup < 2.0" | bc -l) )); then
            echo "ERROR: Speedup below threshold"
            exit 1
          fi

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results-parallel
          path: /tmp/cascade-test-parallel/results.json
```

## Advanced Test: Large-Scale Parallel Execution

```python
# Test with 50 concurrent tasks
executor = ParallelExecutor(max_workers=10)

for i in range(50):
    task = ParallelTask(
        f"task_{i}",
        f"Task {i}",
        lambda x=i: time.sleep(0.1) and f"Result {x}"
    )
    executor.add_task(task)

results = executor.execute_all()

# Expected:
# - All 50 tasks complete
# - Total time: ~0.5-1.0s (50 tasks / 10 workers * 0.1s)
# - No deadlocks or resource exhaustion
```
