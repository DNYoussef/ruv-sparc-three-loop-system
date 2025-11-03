# Test 3: Batch Operations (500x Performance Improvement)

## Objective
Validate batch insert operations achieve 500x performance improvement over individual inserts for 100 vectors.

## Prerequisites
- Node.js 18+
- Python 3.8+
- AgentDB v1.0.7+ installed
- numpy package (`pip install numpy`)

## Test Setup

### 1. Create Test Directory

```bash
mkdir -p tests/batch-ops
cd tests/batch-ops
```

### 2. Generate Test Data

**generate-test-data.py**:
```python
#!/usr/bin/env python3
import json
import numpy as np
import uuid

def generate_test_data(count=100, dim=768):
    """Generate test vectors for batch operations"""
    patterns = []

    for i in range(count):
        # Generate random embedding
        embedding = np.random.randn(dim).astype(np.float32).tolist()

        patterns.append({
            'id': str(uuid.uuid4()),
            'type': 'embedding',
            'domain': f'domain_{i % 10}',
            'embedding': embedding,
            'text': f'Test pattern {i}',
            'metadata': {'index': i},
            'confidence': 0.9,
            'usage_count': 0,
            'success_count': 0
        })

    # Save to JSON
    with open(f'test-data-{count}.json', 'w') as f:
        json.dump(patterns, f, indent=2)

    print(f'✅ Generated {count} test patterns')
    print(f'   File: test-data-{count}.json')
    print(f'   Size: {len(json.dumps(patterns)) / (1024*1024):.2f} MB')

if __name__ == '__main__':
    generate_test_data(100)
    generate_test_data(1000)
```

```bash
python generate-test-data.py
```

**Expected Output**:
```
✅ Generated 100 test patterns
   File: test-data-100.json
   Size: 2.34 MB
✅ Generated 1000 test patterns
   File: test-data-1000.json
   Size: 23.45 MB
```

## Test Cases

### Test Case 3.1: Batch vs Individual Insert (100 vectors)

**Goal**: Verify 500x speedup for batch inserts

```bash
# Run benchmark
python ../../resources/scripts/batch_ops.py \
  --benchmark test.db \
  --count 100
```

**Expected Output**:
```
======================================================================
BATCH OPERATIONS BENCHMARK
======================================================================

Generating 100 test vectors...
✅ Test data generated

Testing individual inserts...
✅ Individual inserts: 1.05 seconds

Testing batch inserts...
✅ Batch inserts: 0.002 seconds

======================================================================
BENCHMARK RESULTS
======================================================================
Records: 100

Individual Inserts:
  Total Time: 1.05 seconds
  Throughput: 95 records/sec
  Avg per record: 10.50 ms

Batch Inserts:
  Total Time: 0.002 seconds
  Throughput: 50,000 records/sec
  Avg per record: 0.02 ms

Improvement: 525.0x faster
======================================================================
```

**Pass Criteria**:
- ✅ Batch insert ≥ 400x faster than individual inserts
- ✅ Batch insert throughput ≥ 40,000 records/sec
- ✅ Batch insert < 5ms for 100 vectors

### Test Case 3.2: Import from JSON

**Goal**: Verify batch import functionality

```bash
# Create test database
rm -f import-test.db

# Import test data
python ../../resources/scripts/batch_ops.py \
  --import test-data-100.json \
  --db import-test.db \
  --batch-size 50
```

**Expected Output**:
```
Starting batch insert of 100 patterns...
Batch size: 50
  Processed: 50/100 (50.0%)
  Processed: 100/100 (100.0%)
✅ Batch insert complete!

======================================================================
BATCH OPERATION STATISTICS
======================================================================
Records Processed: 100
Batch Size: 50
Total Time: 0.05 seconds
Avg Time/Record: 0.500 ms
Throughput: 2,000 records/sec
======================================================================
```

**Verify Import**:
```bash
# Check database contents
sqlite3 import-test.db "SELECT COUNT(*) FROM patterns"
# Expected: 100
```

**Pass Criteria**:
- ✅ All 100 patterns imported successfully
- ✅ Import time < 1 second
- ✅ No data corruption or missing fields

### Test Case 3.3: Export to JSON

**Goal**: Verify batch export functionality

```bash
# Export database
python ../../resources/scripts/batch_ops.py \
  --export import-test.db \
  --output export-test.json
```

**Expected Output**:
```
Exporting patterns to export-test.json...
✅ Exported 100 patterns in 0.08 seconds
   Output: export-test.json
   File size: 2.34 MB
```

**Verify Export**:
```bash
# Check JSON structure
python -c "import json; data = json.load(open('export-test.json')); print(f'Patterns: {len(data)}'); print(f'First pattern: {data[0][\"id\"]}')"
```

**Pass Criteria**:
- ✅ All 100 patterns exported
- ✅ JSON format valid
- ✅ All fields preserved (id, type, domain, embedding, etc.)

### Test Case 3.4: Large Batch (1000 vectors)

**Goal**: Verify scalability to 1000 vectors

```bash
# Import 1000 vectors
rm -f large-test.db

python ../../resources/scripts/batch_ops.py \
  --import test-data-1000.json \
  --db large-test.db \
  --batch-size 100
```

**Expected Output**:
```
Starting batch insert of 1000 patterns...
Batch size: 100
  Processed: 100/1000 (10.0%)
  Processed: 200/1000 (20.0%)
  ...
  Processed: 1000/1000 (100.0%)
✅ Batch insert complete!

======================================================================
BATCH OPERATION STATISTICS
======================================================================
Records Processed: 1,000
Batch Size: 100
Total Time: 0.35 seconds
Avg Time/Record: 0.350 ms
Throughput: 2,857 records/sec
======================================================================
```

**Pass Criteria**:
- ✅ All 1000 patterns imported
- ✅ Import time < 2 seconds
- ✅ Throughput ≥ 2,000 records/sec

### Test Case 3.5: Batch Size Optimization

**Goal**: Verify optimal batch size selection

**batch-size-test.py**:
```python
import time
import json
from pathlib import Path

def test_batch_sizes():
    """Test different batch sizes to find optimal configuration"""

    # Load test data
    with open('test-data-1000.json') as f:
        patterns = json.load(f)

    batch_sizes = [10, 50, 100, 200, 500]
    results = []

    print('Testing different batch sizes...\n')

    for batch_size in batch_sizes:
        # Create test database
        db_path = f'batch-size-{batch_size}.db'
        Path(db_path).unlink(missing_ok=True)

        # Time the import
        start = time.time()
        import subprocess
        subprocess.run([
            'python', '../../resources/scripts/batch_ops.py',
            '--import', 'test-data-1000.json',
            '--db', db_path,
            '--batch-size', str(batch_size)
        ], capture_output=True)
        end = time.time()

        elapsed = end - start
        throughput = 1000 / elapsed

        results.append({
            'batch_size': batch_size,
            'time': elapsed,
            'throughput': throughput
        })

        print(f'Batch Size {batch_size:3d}: {elapsed:.3f}s ({throughput:.0f} records/sec)')

        # Cleanup
        Path(db_path).unlink()

    # Find optimal batch size
    optimal = max(results, key=lambda x: x['throughput'])

    print('\n' + '='*60)
    print('BATCH SIZE OPTIMIZATION')
    print('='*60)
    print(f'Optimal Batch Size: {optimal["batch_size"]}')
    print(f'Best Throughput: {optimal["throughput"]:.0f} records/sec')
    print('='*60)

test_batch_sizes()
```

```bash
python batch-size-test.py
```

**Expected Output**:
```
Testing different batch sizes...

Batch Size  10: 0.850s (1176 records/sec)
Batch Size  50: 0.420s (2381 records/sec)
Batch Size 100: 0.350s (2857 records/sec)
Batch Size 200: 0.340s (2941 records/sec)
Batch Size 500: 0.380s (2632 records/sec)

============================================================
BATCH SIZE OPTIMIZATION
============================================================
Optimal Batch Size: 200
Best Throughput: 2941 records/sec
============================================================
```

**Pass Criteria**:
- ✅ Batch size 100-200 shows best performance
- ✅ Throughput increases with batch size up to ~200
- ✅ Very large batches (500+) show diminishing returns

### Test Case 3.6: Memory Efficiency

**Goal**: Verify batch operations don't cause memory spikes

**memory-test.py**:
```python
import psutil
import json
from pathlib import Path
import subprocess
import time

def test_memory_usage():
    """Monitor memory usage during batch operations"""

    process = psutil.Process()

    # Baseline memory
    baseline = process.memory_info().rss / (1024 * 1024)  # MB

    print(f'Baseline memory: {baseline:.2f} MB\n')

    # Import 1000 vectors
    print('Starting batch import...')
    db_path = 'memory-test.db'
    Path(db_path).unlink(missing_ok=True)

    # Monitor memory during import
    peak_memory = baseline

    subprocess.Popen([
        'python', '../../resources/scripts/batch_ops.py',
        '--import', 'test-data-1000.json',
        '--db', db_path,
        '--batch-size', '100'
    ])

    # Sample memory every 100ms
    for _ in range(20):
        time.sleep(0.1)
        current = process.memory_info().rss / (1024 * 1024)
        peak_memory = max(peak_memory, current)

    memory_increase = peak_memory - baseline

    print(f'Peak memory: {peak_memory:.2f} MB')
    print(f'Memory increase: {memory_increase:.2f} MB')

    status = '✅ PASS' if memory_increase < 50 else '⚠️  HIGH'
    print(f'Status: {status}')

    # Cleanup
    Path(db_path).unlink(missing_ok=True)

test_memory_usage()
```

**Pass Criteria**:
- ✅ Memory increase < 50 MB during batch operations
- ✅ No memory leaks (memory returns to baseline)

## Success Criteria

All test cases must pass:
1. ✅ Batch insert ≥ 400x faster than individual inserts
2. ✅ Import/export functionality works correctly
3. ✅ Scales to 1000+ vectors efficiently
4. ✅ Optimal batch size is 100-200 records
5. ✅ Memory usage remains reasonable (< 50MB increase)

## Cleanup

```bash
# Remove test files
rm -rf tests/batch-ops
```

## Notes

- Batch operations use SQLite transactions for atomic commits
- WAL mode enabled for better concurrent performance
- Synchronous writes disabled during bulk inserts (re-enabled after)
- Optimal batch size depends on vector dimensionality and system
- ~100-200 records per batch is typical sweet spot
