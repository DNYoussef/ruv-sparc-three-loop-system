# Example 3: Debugging Concurrent Data Processing Race Condition

## Scenario

A data analytics platform processes large datasets using a Python worker pool that splits CSV files into chunks, processes each chunk in parallel, and aggregates results. The system works perfectly with small files during development but fails intermittently in production when processing multi-gigabyte datasets. Sometimes results are correct, sometimes they're missing data, and occasionally the process crashes with corrupted state.

## Problem Statement

A concurrent data processing pipeline exhibits non-deterministic failures that only manifest under production load. The code passes all sequential tests, but race conditions cause data corruption during parallel execution. The system needs validation through stress testing in an isolated sandbox to identify and fix concurrency bugs.

**Initial Code:**
```python
class DataProcessor:
    def __init__(self):
        self.results = []  # Shared state - PROBLEM!
        self.processed_chunks = 0
        self.lock = threading.Lock()

    def process_chunk(self, chunk_data):
        """Process a chunk of data."""
        # Simulate processing
        processed = [self.transform_row(row) for row in chunk_data]

        # Aggregate results
        self.results.extend(processed)  # Race condition here!

        with self.lock:
            self.processed_chunks += 1

    def process_file(self, filepath, num_workers=4):
        """Process file in parallel."""
        chunks = self.split_file(filepath, num_workers)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(self.process_chunk, chunks)

        return self.results
```

**Symptom:**
- Small files (100 rows): Works perfectly
- Large files (1M rows): Missing ~5-10% of results randomly
- Concurrent access: Sometimes returns duplicate rows
- Under stress: Occasionally raises IndexError or returns corrupted data

## Audit Process

### Step 1: Setup Sandbox

Create isolated test environment with race detection tools:

```bash
# Initialize sandbox with concurrent testing tools
npx claude-flow@alpha hooks pre-task --description "Debug race condition in data processor"

# Create test environment
cat > requirements.txt << 'EOF'
pytest==7.4.0
pytest-xdist==3.3.1  # Parallel test execution
pytest-timeout==2.5.0
memory-profiler==0.61.0
pyinstrument==4.5.0  # Profiler
faker==19.2.0  # Generate test data
EOF

pip install -r requirements.txt

# Enable Python race condition detection
export PYTHONTHREADDEBUG=1

# Install ThreadSanitizer (TSan) for deeper analysis
apt-get update && apt-get install -y clang
```

**Generate Large Test Dataset:**
```python
# tools/generate_test_data.py
import csv
from faker import Faker
import sys

def generate_large_csv(filename, num_rows=1000000):
    """Generate large test dataset."""
    fake = Faker()

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'email', 'age', 'city', 'score'])

        for i in range(num_rows):
            writer.writerow([
                i,
                fake.name(),
                fake.email(),
                fake.random_int(min=18, max=90),
                fake.city(),
                fake.random_int(min=0, max=100)
            ])

    print(f"Generated {num_rows} rows in {filename}")

if __name__ == '__main__':
    generate_large_csv('test_data_large.csv', 1000000)
    generate_large_csv('test_data_medium.csv', 100000)
    generate_large_csv('test_data_small.csv', 1000)
```

```bash
python tools/generate_test_data.py
```

### Step 2: Generate Test Cases

Design stress tests to expose race conditions:

```python
# tests/test_race_conditions.py
import pytest
import threading
import time
from processor import DataProcessor

class TestConcurrencyBugs:

    def test_sequential_baseline(self):
        """Baseline test with no concurrency - should always work."""
        processor = DataProcessor()
        results = processor.process_file('test_data_small.csv', num_workers=1)

        assert len(results) == 1000, "All rows should be processed"
        assert len(set(r['id'] for r in results)) == 1000, "No duplicates"

    def test_parallel_processing_medium_dataset(self):
        """Test with moderate parallelism."""
        processor = DataProcessor()
        results = processor.process_file('test_data_medium.csv', num_workers=4)

        expected_count = 100000
        actual_count = len(results)

        print(f"\nExpected: {expected_count}, Got: {actual_count}")
        print(f"Missing: {expected_count - actual_count}")
        print(f"Duplicates: {actual_count - len(set(r['id'] for r in results))}")

        assert actual_count == expected_count, \
            f"Race condition detected: missing {expected_count - actual_count} rows"

    @pytest.mark.stress
    def test_high_concurrency_stress(self):
        """Stress test with maximum parallelism."""
        processor = DataProcessor()
        results = processor.process_file('test_data_large.csv', num_workers=16)

        expected_count = 1000000
        actual_count = len(results)
        unique_count = len(set(r['id'] for r in results))

        print(f"\n=== STRESS TEST RESULTS ===")
        print(f"Expected rows: {expected_count}")
        print(f"Actual rows: {actual_count}")
        print(f"Unique IDs: {unique_count}")
        print(f"Data loss: {expected_count - actual_count}")
        print(f"Duplicates: {actual_count - unique_count}")

        assert actual_count == expected_count, "Data loss detected"
        assert unique_count == expected_count, "Duplicate rows detected"

    @pytest.mark.timeout(10)
    def test_rapid_repeated_execution(self):
        """Test repeated execution to expose intermittent failures."""
        failures = []

        for i in range(20):
            processor = DataProcessor()
            results = processor.process_file('test_data_medium.csv', num_workers=8)

            expected = 100000
            actual = len(results)

            if actual != expected:
                failures.append({
                    'iteration': i,
                    'expected': expected,
                    'actual': actual,
                    'missing': expected - actual
                })

        if failures:
            print(f"\n=== FAILURES: {len(failures)}/20 ===")
            for f in failures:
                print(f)

        assert len(failures) == 0, \
            f"Intermittent failures detected in {len(failures)}/20 runs"

    def test_concurrent_state_corruption(self):
        """Test if concurrent access corrupts internal state."""
        processor = DataProcessor()

        def process_and_check():
            results = processor.process_file('test_data_small.csv', num_workers=4)
            return len(results)

        # Run multiple processing tasks concurrently
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_and_check) for _ in range(5)]
            counts = [f.result() for f in futures]

        print(f"\nCounts from concurrent runs: {counts}")

        # Each run should process exactly 1000 rows
        assert all(c == 1000 for c in counts), \
            "State corruption detected: processor reused across concurrent runs"
```

### Step 3: Execute & Monitor

Run tests with race detection and profiling:

```bash
# Run baseline test (should pass)
pytest tests/test_race_conditions.py::TestConcurrencyBugs::test_sequential_baseline -v

# Run parallel tests (will expose race)
pytest tests/test_race_conditions.py::TestConcurrencyBugs::test_parallel_processing_medium_dataset -v -s

# Run stress test with thread sanitizer
pytest tests/test_race_conditions.py -m stress -v -s

# Profile execution
python -m pyinstrument -r html -o profile.html tests/manual_stress_test.py
```

**Test Output (Exposes Race):**
```
tests/test_race_conditions.py::TestConcurrencyBugs::test_parallel_processing_medium_dataset

Expected: 100000, Got: 94738
Missing: 5262
Duplicates: 0

FAILED - AssertionError: Race condition detected: missing 5262 rows

===========================
tests/test_race_conditions.py::TestConcurrencyBugs::test_rapid_repeated_execution

=== FAILURES: 7/20 ===
{'iteration': 2, 'expected': 100000, 'actual': 96841, 'missing': 3159}
{'iteration': 5, 'expected': 100000, 'actual': 98023, 'missing': 1977}
{'iteration': 8, 'expected': 100000, 'actual': 95456, 'missing': 4544}
{'iteration': 11, 'expected': 100000, 'actual': 99201, 'missing': 799}
{'iteration': 14, 'expected': 100000, 'actual': 97834, 'missing': 2166}
{'iteration': 16, 'expected': 100000, 'actual': 96122, 'missing': 3878}
{'iteration': 19, 'expected': 100000, 'actual': 98567, 'missing': 1433}

FAILED - Intermittent failures detected in 7/20 runs
```

### Step 4: Analyze Results

**Add Race Detection Instrumentation:**

```python
# instrumented_processor.py
import threading
import time
import traceback

class InstrumentedDataProcessor:
    def __init__(self):
        self.results = []
        self.processed_chunks = 0
        self.lock = threading.Lock()

        # Race detection
        self.access_log = []
        self.access_lock = threading.Lock()

    def _log_access(self, operation, thread_id, data_size):
        """Log all shared state accesses."""
        with self.access_lock:
            self.access_log.append({
                'timestamp': time.time(),
                'operation': operation,
                'thread': thread_id,
                'size': data_size,
                'stack': ''.join(traceback.format_stack())
            })

    def process_chunk(self, chunk_data):
        thread_id = threading.get_ident()

        processed = [self.transform_row(row) for row in chunk_data]

        # Log before access
        self._log_access('BEFORE_EXTEND', thread_id, len(self.results))

        # RACE CONDITION HERE
        self.results.extend(processed)

        # Log after access
        self._log_access('AFTER_EXTEND', thread_id, len(self.results))

        with self.lock:
            self.processed_chunks += 1
```

**Analyze Access Patterns:**

```python
# tools/analyze_race.py
def analyze_race_condition(access_log):
    """Analyze access log for overlapping writes."""

    # Find overlapping operations
    overlaps = []

    for i, access1 in enumerate(access_log):
        if access1['operation'] != 'BEFORE_EXTEND':
            continue

        # Find corresponding AFTER for this thread
        after = None
        for j in range(i+1, len(access_log)):
            if (access_log[j]['thread'] == access1['thread'] and
                access_log[j]['operation'] == 'AFTER_EXTEND'):
                after = access_log[j]
                break

        if not after:
            continue

        # Check if any other thread accessed during this window
        for other in access_log[i+1:access_log.index(after)]:
            if other['thread'] != access1['thread']:
                overlaps.append({
                    'thread1': access1['thread'],
                    'thread2': other['thread'],
                    'overlap_window': after['timestamp'] - access1['timestamp']
                })

    return overlaps

# Run analysis
processor = InstrumentedDataProcessor()
processor.process_file('test_data_medium.csv', num_workers=8)

overlaps = analyze_race_condition(processor.access_log)
print(f"\n=== RACE DETECTION ===")
print(f"Total overlapping writes: {len(overlaps)}")
print(f"Unique thread conflicts: {len(set((o['thread1'], o['thread2']) for o in overlaps))}")
```

**Output Shows:**
```
=== RACE DETECTION ===
Total overlapping writes: 5,847
Unique thread conflicts: 28

CRITICAL: Multiple threads calling .extend() simultaneously without synchronization!

Example conflict:
  Thread 140234567: BEFORE_EXTEND at 1705318980.123 (results size: 45000)
  Thread 140234789: BEFORE_EXTEND at 1705318980.124 (results size: 45000) ← READ STALE SIZE
  Thread 140234567: AFTER_EXTEND at 1705318980.128 (results size: 57500)
  Thread 140234789: AFTER_EXTEND at 1705318980.129 (results size: 58200) ← OVERWROTE DATA

Expected final size: 70000 (45000 + 12500 + 12500)
Actual final size: 58200 (missing 11800 rows due to overwrite)
```

**Root Cause Identified:**

1. **Unprotected list extension**: `self.results.extend()` is not atomic and not locked
2. **Read-modify-write race**: Multiple threads read `results` size, extend, causing overwrites
3. **Lock insufficient**: Lock only protects `processed_chunks`, not `results`
4. **List is not thread-safe**: Python lists are not safe for concurrent modification

### Step 5: Debug & Fix

**Fix 1: Proper Synchronization**

```python
# processor_fixed_v1.py
class DataProcessorFixed:
    def __init__(self):
        self.results = []
        self.processed_chunks = 0
        self.results_lock = threading.Lock()  # NEW: Protect results
        self.counter_lock = threading.Lock()

    def process_chunk(self, chunk_data):
        """Process chunk with proper synchronization."""
        processed = [self.transform_row(row) for row in chunk_data]

        # Protected write to shared state
        with self.results_lock:
            self.results.extend(processed)

        with self.counter_lock:
            self.processed_chunks += 1
```

**Fix 2: Eliminate Shared Mutable State (Better Solution)**

```python
# processor_fixed_v2.py
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataProcessorImproved:
    def process_chunk(self, chunk_data):
        """Process chunk and return results (no shared state)."""
        return [self.transform_row(row) for row in chunk_data]

    def process_file(self, filepath, num_workers=4):
        """Process file using futures pattern - no shared state!"""
        chunks = self.split_file(filepath, num_workers)

        # No shared state - collect results from futures
        all_results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all chunks
            futures = {executor.submit(self.process_chunk, chunk): chunk
                      for chunk in chunks}

            # Collect results as they complete
            for future in as_completed(futures):
                chunk_results = future.result()
                all_results.extend(chunk_results)

        return all_results
```

**Fix 3: Use Thread-Safe Queue (Alternative)**

```python
# processor_fixed_v3.py
from queue import Queue
from threading import Thread

class DataProcessorQueue:
    def __init__(self):
        self.results_queue = Queue()  # Thread-safe queue
        self.processed_chunks = 0
        self.counter_lock = threading.Lock()

    def process_chunk(self, chunk_data):
        """Process chunk and push to queue."""
        processed = [self.transform_row(row) for row in chunk_data]

        # Thread-safe queue operation
        for item in processed:
            self.results_queue.put(item)

        with self.counter_lock:
            self.processed_chunks += 1

    def process_file(self, filepath, num_workers=4):
        """Process file using thread-safe queue."""
        chunks = self.split_file(filepath, num_workers)

        # Start worker threads
        threads = []
        for chunk in chunks:
            t = Thread(target=self.process_chunk, args=(chunk,))
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        # Collect results from queue
        results = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())

        return results
```

### Step 6: Verify Fix

**Run Comprehensive Verification:**

```bash
# Test all three fixed versions
for version in fixed_v1 fixed_v2 fixed_v3; do
    echo "=== Testing $version ==="
    pytest tests/test_race_conditions.py \
        --processor-version=$version \
        -v -s
done

# Stress test with ThreadSanitizer
TSAN_OPTIONS="halt_on_error=1" pytest tests/test_race_conditions.py -m stress

# Run 100 iterations to verify reliability
pytest tests/test_race_conditions.py::TestConcurrencyBugs::test_rapid_repeated_execution \
    --iterations=100 -v
```

**Success Output:**
```
=== Testing fixed_v2 (futures pattern) ===

tests/test_race_conditions.py::test_sequential_baseline PASSED
tests/test_race_conditions.py::test_parallel_processing_medium_dataset PASSED
  Expected: 100000, Got: 100000 ✓
  Missing: 0
  Duplicates: 0

tests/test_race_conditions.py::test_high_concurrency_stress PASSED
  Expected rows: 1000000
  Actual rows: 1000000 ✓
  Unique IDs: 1000000 ✓
  Data loss: 0
  Duplicates: 0

tests/test_race_conditions.py::test_rapid_repeated_execution PASSED
  FAILURES: 0/100 ✓

ThreadSanitizer: No races detected ✓

====== 4 passed in 45.2s ======
```

**Performance Comparison:**

```python
# tools/benchmark_fixes.py
import time
from processor_original import DataProcessor
from processor_fixed_v1 import DataProcessorFixed
from processor_fixed_v2 import DataProcessorImproved
from processor_fixed_v3 import DataProcessorQueue

def benchmark(processor_class, num_runs=10):
    times = []
    for _ in range(num_runs):
        processor = processor_class()
        start = time.time()
        results = processor.process_file('test_data_large.csv', num_workers=8)
        elapsed = time.time() - start
        times.append(elapsed)
        assert len(results) == 1000000  # Verify correctness

    return {
        'avg': sum(times) / len(times),
        'min': min(times),
        'max': max(times)
    }

print("=== PERFORMANCE COMPARISON ===")
print(f"Original (broken): {benchmark(DataProcessor)}")
print(f"Fixed v1 (locks): {benchmark(DataProcessorFixed)}")
print(f"Fixed v2 (futures): {benchmark(DataProcessorImproved)}")
print(f"Fixed v3 (queue): {benchmark(DataProcessorQueue)}")
```

**Results:**
```
=== PERFORMANCE COMPARISON ===
Original (broken): FAILS - data corruption
Fixed v1 (locks): {'avg': 8.42s, 'min': 8.21s, 'max': 8.67s}
Fixed v2 (futures): {'avg': 6.18s, 'min': 6.02s, 'max': 6.31s} ← BEST
Fixed v3 (queue): {'avg': 7.85s, 'min': 7.64s, 'max': 8.09s}
```

## Outcome

**What Was Discovered:**
- Unprotected access to shared mutable state caused race conditions
- List.extend() is not atomic and caused data loss when called concurrently
- Lock protected counter but not the results list
- Race only appeared under load with multiple threads
- Futures pattern is fastest and most reliable (no shared state)

**How It Helped:**
1. **Stress testing** exposed intermittent failures that normal tests missed
2. **Access logging** pinpointed exact race condition location
3. **Thread sanitizer** verified no remaining race conditions
4. **Performance benchmarking** identified best fix (futures pattern)
5. **100+ iteration testing** proved reliability

**Production Impact:**
- Data processing reliability: 95% → 100%
- Processing speed improved 27% (futures pattern)
- Zero data loss incidents post-fix
- System handles 10x larger datasets reliably

## Key Takeaways

1. **Concurrency bugs are intermittent**: Pass sometimes, fail randomly
2. **Stress testing is essential**: Normal tests can't expose race conditions
3. **Shared mutable state is dangerous**: Eliminate it when possible
4. **Locks are error-prone**: Futures pattern avoids shared state entirely
5. **Thread sanitizers catch races**: Use TSan/Helgrind for verification
6. **Verify under load**: Small datasets hide concurrency bugs
7. **Measure performance**: Best correctness fix may also be fastest

**When to Apply This Pattern:**
- Any code using threads or async concurrency
- When failures are intermittent or only under load
- Before deploying parallel/concurrent code to production
- When debugging "works sometimes" bugs
- For any shared mutable state across threads
- When processing large datasets in parallel
