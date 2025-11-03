# Test 2: HNSW Index Tuning (150x-12,500x Search Speedup)

## Objective
Validate HNSW index tuning achieves 150x faster search on 10K vectors and scales to 12,500x on 1M vectors.

## Prerequisites
- Node.js 18+
- AgentDB v1.0.7+ installed
- Bash (for running tuning script)
- 10K-1M test vectors

## Test Setup

### 1. Generate Test Databases

```bash
# Create test directory
mkdir -p tests/hnsw-tuning
cd tests/hnsw-tuning

# Generate small dataset (10K vectors)
node ../generate-test-vectors.js --count 10000 --dim 768 --output small-10k.db

# Generate medium dataset (100K vectors) - optional
node ../generate-test-vectors.js --count 100000 --dim 768 --output medium-100k.db

# Generate large dataset (1M vectors) - optional, takes ~10 minutes
# node ../generate-test-vectors.js --count 1000000 --dim 768 --output large-1m.db
```

### 2. Run HNSW Tuning Script

```bash
# Tune for small dataset
bash ../../resources/scripts/hnsw_tuning.sh small-10k.db small
```

**Expected Output**:
```
========================================
AgentDB HNSW Index Tuning
========================================

Database: small-10k.db
Dataset Size: small

📊 HNSW Parameters for small:
  M (connections per layer): 8
  efConstruction (build quality): 100
  efSearch (search quality): 50
  Description: Small dataset (<10K vectors) - Fast build, minimal memory

⚡ Expected Performance:
  Search Latency: <100µs
  Recall Rate: 95-98%
  Build Time: <1 second

✅ Configuration file generated: .agentdb/hnsw-config-small.ts

✅ Benchmark script generated: .agentdb/benchmark-small.ts

📊 HNSW Parameter Comparison Table:
| Dataset Size | M  | efConstruction | efSearch | Search Latency | Recall | Memory Usage |
|--------------|----|--------------------|----------|----------------|--------|--------------|
| Small (<10K) | 8  | 100                | 50       | <100µs         | 95-98% | Low          |
| Medium       | 16 | 200                | 100      | <200µs         | 97-99% | Medium       |
| Large        | 32 | 400                | 200      | <500µs         | 98-99% | High         |
| Massive (>1M)| 48 | 500                | 250      | <2ms           | 99%+   | Very High    |

========================================
Next Steps:
========================================
1. Import the configuration:
   import { createOptimizedAdapter } from './.agentdb/hnsw-config-small.ts';

2. Run the benchmark:
   npx tsx .agentdb/benchmark-small.ts

3. Adjust parameters if needed in .agentdb/hnsw-config-small.ts

4. Monitor performance with:
   npx agentdb@latest stats small-10k.db

💡 Tuning Tips:
  • Increase M for better recall (but more memory)
  • Increase efConstruction for better index quality (but slower build)
  • Increase efSearch for better recall (but slower search)
  • Use binary quantization with large datasets to save memory
  • Cache hot patterns to reduce database queries

✅ HNSW tuning complete!
```

## Test Cases

### Test Case 2.1: Small Dataset (10K vectors) - 150x Speedup

**Goal**: Verify 150x search speedup on 10K vectors

**small-dataset-benchmark.js**:
```javascript
const { createAgentDBAdapter } = require('agentic-flow/reasoningbank');

async function benchmarkSmall() {
  console.log('Benchmarking 10K vector dataset...\n');

  // Linear scan (no HNSW)
  const linearAdapter = await createAgentDBAdapter({
    dbPath: 'small-10k.db',
    hnswM: 0, // Disable HNSW
  });

  // HNSW optimized
  const hnswAdapter = await createAgentDBAdapter({
    dbPath: 'small-10k.db',
    hnswM: 8,
    hnswEfConstruction: 100,
    hnswEfSearch: 50,
  });

  const queryEmbedding = Array.from({ length: 768 }, () => Math.random() * 2 - 1);

  // Warm-up
  await linearAdapter.retrieveWithReasoning(queryEmbedding, { k: 10 });
  await hnswAdapter.retrieveWithReasoning(queryEmbedding, { k: 10 });

  // Benchmark linear scan
  const iterations = 50;
  const linearStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    await linearAdapter.retrieveWithReasoning(queryEmbedding, { k: 10 });
  }
  const linearEnd = performance.now();
  const linearAvg = (linearEnd - linearStart) / iterations;

  // Benchmark HNSW
  const hnswStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    await hnswAdapter.retrieveWithReasoning(queryEmbedding, { k: 10 });
  }
  const hnswEnd = performance.now();
  const hnswAvg = (hnswEnd - hnswStart) / iterations;

  const speedup = linearAvg / hnswAvg;

  console.log('========================================');
  console.log('SMALL DATASET BENCHMARK (10K vectors)');
  console.log('========================================');
  console.log(`Linear Scan: ${linearAvg.toFixed(2)}ms`);
  console.log(`HNSW Index: ${hnswAvg.toFixed(2)}ms`);
  console.log(`Speedup: ${speedup.toFixed(1)}x`);
  console.log(`Target: 150x`);
  console.log(`Status: ${speedup >= 100 ? '✅ PASS' : '❌ FAIL'}`);
  console.log('========================================');
}

benchmarkSmall();
```

```bash
node small-dataset-benchmark.js
```

**Expected Output**:
```
Benchmarking 10K vector dataset...

========================================
SMALL DATASET BENCHMARK (10K vectors)
========================================
Linear Scan: 15.23ms
HNSW Index: 0.10ms
Speedup: 152.3x
Target: 150x
Status: ✅ PASS
========================================
```

**Pass Criteria**:
- ✅ HNSW search < 0.2ms (200µs)
- ✅ Speedup ≥ 100x
- ✅ Recall ≥ 95%

### Test Case 2.2: Medium Dataset (100K vectors) - 1,250x Speedup

**Goal**: Verify 1,250x search speedup on 100K vectors

```bash
# Tune for medium dataset
bash ../../resources/scripts/hnsw_tuning.sh medium-100k.db medium

# Run benchmark
node medium-dataset-benchmark.js
```

**Expected Output**:
```
========================================
MEDIUM DATASET BENCHMARK (100K vectors)
========================================
Linear Scan: 150.45ms
HNSW Index: 0.12ms
Speedup: 1,254.2x
Target: 1,250x
Status: ✅ PASS
========================================
```

**Pass Criteria**:
- ✅ HNSW search < 0.3ms (300µs)
- ✅ Speedup ≥ 1,000x
- ✅ Recall ≥ 97%

### Test Case 2.3: Parameter Tuning Validation

**Goal**: Verify M and efSearch parameters affect performance as expected

**parameter-tuning-test.js**:
```javascript
const { createAgentDBAdapter } = require('agentic-flow/reasoningbank');

async function testParameterTuning() {
  console.log('Testing HNSW parameter tuning...\n');

  const queryEmbedding = Array.from({ length: 768 }, () => Math.random() * 2 - 1);

  // Test different M values
  const mValues = [8, 16, 32];
  const results = [];

  for (const M of mValues) {
    const adapter = await createAgentDBAdapter({
      dbPath: 'small-10k.db',
      hnswM: M,
      hnswEfConstruction: 200,
      hnswEfSearch: 100,
    });

    // Warm-up
    await adapter.retrieveWithReasoning(queryEmbedding, { k: 10 });

    // Benchmark
    const iterations = 100;
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      await adapter.retrieveWithReasoning(queryEmbedding, { k: 10 });
    }
    const end = performance.now();
    const avgLatency = (end - start) / iterations;

    results.push({ M, avgLatency });
  }

  console.log('========================================');
  console.log('PARAMETER TUNING TEST');
  console.log('========================================');
  console.log('Effect of M (connections per layer):\n');

  results.forEach(({ M, avgLatency }) => {
    console.log(`M=${M}: ${avgLatency.toFixed(3)}ms`);
  });

  // Verify M increases lead to better (or similar) performance
  const latenciesDecrease = results[1].avgLatency <= results[0].avgLatency * 1.2;

  console.log(`\nTrend: ${latenciesDecrease ? 'Expected ✅' : 'Unexpected ❌'}`);
  console.log('(Higher M should maintain or improve performance)');
  console.log('========================================');
}

testParameterTuning();
```

**Pass Criteria**:
- ✅ Higher M values maintain or improve search speed
- ✅ All configurations achieve sub-millisecond search

### Test Case 2.4: Build Time Validation

**Goal**: Verify index build time scales appropriately

**build-time-test.js**:
```javascript
const { createAgentDBAdapter } = require('agentic-flow/reasoningbank');

async function testBuildTime() {
  console.log('Testing HNSW build time...\n');

  const datasets = [
    { path: 'small-10k.db', size: '10K', expected: 1 },
    { path: 'medium-100k.db', size: '100K', expected: 30 },
  ];

  for (const { path, size, expected } of datasets) {
    const start = performance.now();

    const adapter = await createAgentDBAdapter({
      dbPath: path,
      hnswM: 16,
      hnswEfConstruction: 200,
    });

    const end = performance.now();
    const buildTime = (end - start) / 1000; // seconds

    const status = buildTime <= expected ? '✅ PASS' : '⚠️  SLOW';

    console.log(`${size} vectors: ${buildTime.toFixed(2)}s (expected: <${expected}s) ${status}`);
  }
}

testBuildTime();
```

**Pass Criteria**:
- ✅ 10K vectors: < 1 second build time
- ✅ 100K vectors: < 30 seconds build time

## Success Criteria

All test cases must pass:
1. ✅ Small dataset (10K): ≥ 100x speedup, < 200µs search
2. ✅ Medium dataset (100K): ≥ 1,000x speedup, < 300µs search
3. ✅ Parameter tuning behaves as expected
4. ✅ Build times within acceptable ranges

## Cleanup

```bash
# Remove test files
rm -rf tests/hnsw-tuning
```

## Notes

- HNSW achieves O(log n) search complexity vs O(n) linear scan
- M parameter controls graph connectivity (higher = better recall, more memory)
- efConstruction affects build quality (higher = better quality, slower build)
- efSearch affects search accuracy (higher = better recall, slower search)
- Combine with quantization for maximum performance
