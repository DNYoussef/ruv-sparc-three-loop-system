# Example 1: Python Performance Specialist Agent

**Agent Type**: Specialist | **Domain**: Python Code Optimization | **Complexity**: Medium

## Overview

This example demonstrates creating a domain-specific specialist agent focused on Python code optimization. The agent applies profiling-driven analysis, algorithmic improvements, and data structure optimizations to enhance Python code performance.

## Phase 1: Specification

### Agent Definition

**Name**: Python Performance Optimizer

**Domain**: Python code optimization and performance tuning

**Core Capabilities**:
1. Profile Python code to identify bottlenecks
2. Analyze algorithmic complexity (Big-O analysis)
3. Recommend data structure improvements
4. Apply Cython/NumPy optimizations for numerical code
5. Implement caching and memoization strategies
6. Optimize I/O operations and memory usage

**Input Format**:
- Python source code files (.py)
- Performance requirements (target: 2x, 5x, 10x improvement)
- Current profiling data (optional)

**Output Format**:
```
1. Bottleneck Analysis
   - Profiling results with hotspots
   - Complexity analysis (Big-O)
   - Memory usage patterns

2. Optimization Strategy
   - Recommended techniques
   - Expected improvements
   - Trade-offs and considerations

3. Optimized Implementation
   - Modified code with improvements
   - Inline comments explaining changes
   - Alternative approaches

4. Benchmarks
   - Before/after performance metrics
   - Memory usage comparison
   - Scalability analysis
```

**Quality Criteria**:
- Minimum 2x performance improvement
- Maintain 100% functional correctness
- Preserve code readability
- Include unit tests for optimized code
- Document trade-offs and assumptions

## Phase 2: Prompt Engineering

### Evidence-Based Prompt

```markdown
You are a senior Python performance engineer with 10+ years of experience optimizing production Python systems at scale. Your expertise includes profiling-driven optimization, algorithmic complexity analysis, data structure selection, and high-performance Python techniques (Cython, NumPy, multiprocessing).

## Your Approach

**Step 1: Profile and Analyze**
- Use cProfile or line_profiler to identify hotspots
- Analyze algorithmic complexity (Big-O)
- Identify memory bottlenecks with memory_profiler
- Quantify the performance baseline

**Step 2: Develop Optimization Strategy**
Before implementing, explain your reasoning:
- Why is this code slow? (root cause)
- What optimization technique applies? (algorithm, data structure, caching, vectorization)
- What are the trade-offs? (memory vs speed, readability vs performance)
- What improvement can we expect? (estimated speedup)

**Step 3: Implement Optimizations**
Apply techniques in priority order:
1. Algorithmic improvements (O(n²) → O(n log n))
2. Data structure optimization (list → dict, set)
3. Built-in functions and libraries (itertools, NumPy)
4. Caching and memoization (functools.lru_cache)
5. Vectorization (NumPy, pandas)
6. Parallelization (multiprocessing, asyncio)

**Step 4: Validate and Benchmark**
- Verify functional correctness with unit tests
- Measure performance improvement with timeit
- Profile optimized code to confirm improvements
- Document before/after metrics

## Output Format

Provide your response in this structure:

### 1. Bottleneck Analysis
- Profiling results with line-by-line timing
- Complexity analysis (current Big-O)
- Memory usage patterns

### 2. Optimization Strategy
- Root cause of performance issues
- Recommended optimization techniques
- Expected improvement estimate
- Trade-offs to consider

### 3. Optimized Implementation
```python
# Optimized code here with inline comments
# Explain each optimization decision
```

### 4. Benchmarks
```
Before: [execution time, memory usage]
After:  [execution time, memory usage]
Improvement: [X.Xx speedup]
Scalability: [Big-O before → Big-O after]
```

### 5. Testing
```python
# Unit tests to verify correctness
```

## Few-Shot Examples

**Example 1: List Iteration → NumPy Vectorization**

Input:
```python
def process_data(data):
    result = []
    for item in data:
        result.append(item * 2 + 10)
    return result
```

Bottleneck: O(n) iteration with list append overhead

Optimization:
```python
import numpy as np

def process_data(data):
    return np.array(data) * 2 + 10
```

Improvement: 10x faster for 1M+ elements (vectorized operations)

---

**Example 2: Nested Loops → Dictionary Lookup**

Input:
```python
def find_matches(list1, list2):
    matches = []
    for item1 in list1:
        for item2 in list2:
            if item1 == item2:
                matches.append(item1)
    return matches
```

Bottleneck: O(n*m) nested iteration

Optimization:
```python
def find_matches(list1, list2):
    set2 = set(list2)  # O(m) preprocessing
    return [item for item in list1 if item in set2]  # O(n) lookup
```

Improvement: O(n*m) → O(n+m), 100x faster for large inputs

---

**Example 3: Recursive Fibonacci → Memoization**

Input:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

Bottleneck: O(2^n) exponential time due to repeated calculations

Optimization:
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

Improvement: O(2^n) → O(n), fibonacci(35) from 5 seconds to <0.001 seconds

---

**Example 4: String Concatenation → join()**

Input:
```python
def build_string(items):
    result = ""
    for item in items:
        result += str(item) + ","
    return result
```

Bottleneck: O(n²) due to string immutability

Optimization:
```python
def build_string(items):
    return ",".join(str(item) for item in items)
```

Improvement: O(n²) → O(n), 50x faster for 10,000+ items

---

**Example 5: File I/O → Buffered Reading**

Input:
```python
def process_large_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            process_line(line.strip())
```

Optimization:
```python
def process_large_file(filename):
    with open(filename, 'r', buffering=1024*1024) as f:  # 1MB buffer
        for line in f:
            process_line(line.strip())
```

Improvement: 3x faster for multi-GB files through reduced syscalls

## Quality Constraints

- Always verify correctness with unit tests
- Maintain code readability (avoid premature optimization)
- Document trade-offs clearly
- Provide benchmark data with timeit or pytest-benchmark
- Consider edge cases (empty input, large input, boundary values)
- Preserve function signatures and API compatibility
- Include type hints for clarity

## When to Apply Each Technique

- **Algorithmic**: When complexity > O(n log n) or nested loops
- **Data Structures**: When lookups, insertions, or membership tests are frequent
- **Caching**: When expensive functions called repeatedly with same args
- **Vectorization**: When processing numerical arrays or large datasets
- **Parallelization**: When tasks are independent and CPU-bound
- **I/O Optimization**: When disk/network I/O dominates runtime
```

### Prompt Engineering Principles Applied

1. **Role Definition**: Senior Python performance engineer with 10+ years experience
2. **Context Provision**: Profiling-driven optimization, algorithmic analysis, Python-specific techniques
3. **Task Decomposition**: 4-step process (Profile → Strategy → Implement → Validate)
4. **Chain-of-Thought**: Explicit reasoning before implementing ("explain your reasoning")
5. **Few-Shot Learning**: 5 concrete examples covering common optimization patterns
6. **Output Formatting**: Structured response with 5 sections
7. **Quality Constraints**: Explicit correctness, readability, and benchmarking requirements

## Phase 3: Testing & Validation

### Test Suite

```python
# test_python_optimizer_agent.py
import pytest
from python_optimizer_agent import optimize_code

class TestPythonOptimizer:
    """Test suite for Python Performance Optimizer agent"""

    def test_simple_loop_optimization(self):
        """Test: Replace loop with list comprehension"""
        input_code = """
def double_values(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
"""
        output = optimize_code(input_code, target_improvement=2.0)

        # Verify optimization was applied
        assert "list comprehension" in output.strategy.lower() or "[" in output.optimized_code

        # Verify correctness
        assert output.correctness_verified is True

        # Verify improvement
        assert output.speedup >= 2.0

    def test_nested_dict_lookup(self):
        """Test: Replace nested try/except with dict.get()"""
        input_code = """
def get_nested_value(data, k1, k2, k3):
    try:
        return data[k1][k2][k3]
    except KeyError:
        return None
"""
        output = optimize_code(input_code, target_improvement=1.5)

        # Verify dict.get() chaining suggested
        assert "get(" in output.optimized_code
        assert output.correctness_verified is True

    def test_recursive_fibonacci(self):
        """Test: Add memoization to recursive function"""
        input_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        output = optimize_code(input_code, target_improvement=100.0)

        # Verify memoization applied
        assert "lru_cache" in output.optimized_code or "memo" in output.optimized_code.lower()
        assert output.speedup >= 100.0  # Exponential to linear
        assert output.complexity_before == "O(2^n)"
        assert output.complexity_after == "O(n)"

    def test_string_concatenation(self):
        """Test: Replace += with str.join()"""
        input_code = """
def build_csv(items):
    result = ""
    for item in items:
        result += str(item) + ","
    return result
"""
        output = optimize_code(input_code, target_improvement=5.0)

        # Verify join() suggested
        assert "join" in output.optimized_code
        assert output.speedup >= 5.0
        assert output.complexity_before == "O(n²)"
        assert output.complexity_after == "O(n)"

    def test_numpy_vectorization(self):
        """Test: Replace loop with NumPy operations"""
        input_code = """
def scale_and_shift(data, scale, shift):
    result = []
    for value in data:
        result.append(value * scale + shift)
    return result
"""
        output = optimize_code(input_code, target_improvement=10.0)

        # Verify NumPy suggested
        assert "numpy" in output.optimized_code.lower() or "np." in output.optimized_code
        assert output.speedup >= 10.0

    def test_edge_case_empty_input(self):
        """Test: Handle empty input gracefully"""
        input_code = """
def process_data(data):
    return [x * 2 for x in data]
"""
        output = optimize_code(input_code, target_improvement=1.5)

        # Verify edge cases mentioned
        assert output.edge_cases_tested is True
        assert "empty" in output.considerations.lower()

    def test_preserves_api_compatibility(self):
        """Test: Optimized code maintains same function signature"""
        input_code = """
def calculate_stats(numbers, precision=2):
    mean = sum(numbers) / len(numbers)
    return round(mean, precision)
"""
        output = optimize_code(input_code, target_improvement=1.5)

        # Verify function signature unchanged
        original_sig = extract_function_signature(input_code)
        optimized_sig = extract_function_signature(output.optimized_code)
        assert original_sig == optimized_sig

    def test_includes_benchmarks(self):
        """Test: Output includes before/after benchmark data"""
        input_code = """
def find_duplicates(data):
    duplicates = []
    for i, item in enumerate(data):
        if item in data[:i]:
            duplicates.append(item)
    return duplicates
"""
        output = optimize_code(input_code, target_improvement=10.0)

        # Verify benchmark data present
        assert output.benchmark_before is not None
        assert output.benchmark_after is not None
        assert output.benchmark_method in ["timeit", "pytest-benchmark"]

    def test_provides_unit_tests(self):
        """Test: Output includes unit tests for correctness"""
        input_code = """
def merge_sorted_lists(list1, list2):
    result = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result
"""
        output = optimize_code(input_code, target_improvement=1.5)

        # Verify unit tests provided
        assert output.unit_tests is not None
        assert "def test_" in output.unit_tests
        assert "assert" in output.unit_tests

    def test_documents_tradeoffs(self):
        """Test: Output documents trade-offs of optimizations"""
        input_code = """
def process_in_parallel(items, func):
    results = []
    for item in items:
        results.append(func(item))
    return results
"""
        output = optimize_code(input_code, target_improvement=4.0)

        # Verify trade-offs documented
        assert output.tradeoffs is not None
        assert len(output.tradeoffs) > 0
        assert any("memory" in t.lower() or "readability" in t.lower()
                   for t in output.tradeoffs)
```

### Performance Validation

```python
# benchmark_python_optimizer.py
import timeit
from python_optimizer_agent import optimize_code

def benchmark_agent_performance():
    """Benchmark the Python Optimizer agent's performance"""

    test_cases = [
        ("Simple loop", "def f(x): result = []; [result.append(i*2) for i in x]; return result"),
        ("Nested loop", "def f(x, y): return [[i*j for j in y] for i in x]"),
        ("Recursive fib", "def fib(n): return fib(n-1)+fib(n-2) if n>1 else n"),
        ("String concat", "def f(x): s=''; [s:=s+str(i) for i in x]; return s"),
        ("Dict lookup", "def f(d, k): return d[k[0]][k[1]][k[2]] if k[0] in d else None"),
    ]

    results = []
    for name, code in test_cases:
        # Time the optimization process
        start = timeit.default_timer()
        output = optimize_code(code, target_improvement=2.0)
        elapsed = timeit.default_timer() - start

        results.append({
            "test_case": name,
            "agent_time": elapsed,
            "speedup_achieved": output.speedup,
            "correctness": output.correctness_verified
        })

    # Print results
    print("\n=== Python Optimizer Agent Benchmarks ===")
    for r in results:
        print(f"{r['test_case']:20} | Agent: {r['agent_time']:.3f}s | "
              f"Speedup: {r['speedup_achieved']:.1f}x | "
              f"Correct: {r['correctness']}")

    # Calculate aggregate metrics
    avg_agent_time = sum(r['agent_time'] for r in results) / len(results)
    avg_speedup = sum(r['speedup_achieved'] for r in results) / len(results)
    correctness_rate = sum(1 for r in results if r['correctness']) / len(results)

    print(f"\n=== Aggregate Metrics ===")
    print(f"Average Agent Time: {avg_agent_time:.3f}s")
    print(f"Average Speedup: {avg_speedup:.1f}x")
    print(f"Correctness Rate: {correctness_rate*100:.1f}%")

    assert avg_speedup >= 2.0, "Agent should achieve 2x+ average speedup"
    assert correctness_rate >= 0.95, "Agent should maintain 95%+ correctness"

if __name__ == "__main__":
    benchmark_agent_performance()
```

### Iteration Log

**Iteration 1**: Initial prompt with basic role definition
- **Issue**: Too generic, didn't provide enough Python-specific guidance
- **Fix**: Added profiling-driven approach, specific Python techniques

**Iteration 2**: Added few-shot examples
- **Issue**: Examples too simple, didn't cover edge cases
- **Fix**: Added 5 diverse examples with different optimization patterns

**Iteration 3**: Enhanced output format
- **Issue**: Missing benchmark data and unit tests in output
- **Fix**: Added explicit sections for benchmarks and testing

**Iteration 4**: Added quality constraints
- **Issue**: Some optimizations broke correctness or readability
- **Fix**: Added explicit constraints about correctness, testing, and trade-offs

**Iteration 5**: Chain-of-Thought reasoning
- **Issue**: Agent jumped to solutions without explaining reasoning
- **Fix**: Added "explain your reasoning" step in strategy phase

**Final Results**:
- Correctness: 98% (49/50 test cases)
- Average speedup: 4.2x
- Agent response time: 3.5s average
- Test pass rate: 96%

## Phase 4: Integration

### Coordination Protocol

```bash
# Pre-task: Initialize Python optimizer agent
npx claude-flow@alpha hooks pre-task \
  --description "Optimize Python data processing pipeline" \
  --agent "python-optimizer" \
  --priority "high"

# Session restore: Load prior optimization patterns
npx claude-flow@alpha hooks session-restore \
  --session-id "swarm-python-opt-001"
```

### Memory Integration

```javascript
// hooks/python-optimizer-memory.js
const { taggedMemoryStore } = require('./hooks/12fa/memory-mcp-tagging-protocol.js');

function storeOptimization(optimization) {
  return taggedMemoryStore(
    'python-optimizer',
    JSON.stringify({
      original_code: optimization.input,
      optimized_code: optimization.output,
      technique: optimization.strategy,
      speedup: optimization.speedup,
      complexity_improvement: `${optimization.complexity_before} → ${optimization.complexity_after}`
    }),
    {
      task_id: optimization.task_id,
      file: optimization.file_path,
      improvement: `${optimization.speedup}x`,
      technique: optimization.strategy
    }
  );
}

module.exports = { storeOptimization };
```

### Communication Patterns

```bash
# Notify completion
npx claude-flow@alpha hooks notify \
  --message "Python optimization complete: 3.2x speedup on data_processor.py"

# Update shared memory
npx claude-flow@alpha hooks post-edit \
  --file "src/data_processor.py" \
  --memory-key "swarm/python-optimizer/output/data_processor"

# Post-task metrics
npx claude-flow@alpha hooks post-task \
  --task-id "python-opt-001" \
  --metrics '{"speedup": 3.2, "correctness": true, "time": 4.5}'
```

### Monitoring & Metrics

```bash
# Export session metrics
npx claude-flow@alpha hooks session-end \
  --export-metrics true \
  --output "metrics/python-optimizer-session.json"
```

**Metrics Dashboard**:
```json
{
  "session_id": "swarm-python-opt-001",
  "agent": "python-optimizer",
  "tasks_completed": 15,
  "average_speedup": 4.2,
  "correctness_rate": 0.98,
  "average_response_time": 3.5,
  "techniques_used": {
    "algorithmic": 6,
    "data_structure": 4,
    "vectorization": 3,
    "memoization": 2
  }
}
```

## Usage Example

```javascript
// Spawn Python Optimizer agent via Claude Code Task tool
Task(
  "Python Performance Optimizer",
  `Optimize the data processing pipeline in src/data_processor.py.
   Current bottleneck: nested loops processing 1M+ records taking 45 seconds.
   Target: Reduce to <10 seconds (4.5x improvement).

   Use hooks for coordination:
   - Pre-task: Initialize with profiling data
   - Post-edit: Store optimized code in memory
   - Post-task: Export performance metrics`,
  "code-analyzer"
)
```

## Results

**Metrics**:
- Average speedup: 4.2x (range: 2.1x to 15.3x)
- Correctness rate: 98%
- Agent response time: 3.5s average
- Test coverage: 96%
- Code readability score: 8.2/10

**Common Optimizations Applied**:
1. Algorithmic improvements (40% of cases)
2. Data structure optimization (27% of cases)
3. NumPy vectorization (20% of cases)
4. Caching/memoization (13% of cases)

**Lessons Learned**:
- Profiling data significantly improves optimization accuracy
- Few-shot examples crucial for consistent output format
- Chain-of-Thought reasoning reduces incorrect optimizations
- Explicit correctness testing prevents regressions
- Trade-off documentation helps users make informed decisions

---

**Next Steps**: Adapt this pattern for other specialist domains (React, SQL, Docker, etc.)
