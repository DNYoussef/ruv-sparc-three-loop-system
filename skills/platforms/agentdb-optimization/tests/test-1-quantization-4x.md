# Test 1: Scalar Quantization (4x Memory Reduction)

## Objective
Validate scalar quantization reduces memory usage by 4x while maintaining 98-99% accuracy.

## Prerequisites
- Node.js 18+
- AgentDB v1.0.7+ installed
- Test database with 10,000 vectors (768-dimensional)

## Test Setup

### 1. Generate Test Database

```bash
# Create test directory
mkdir -p tests/quantization
cd tests/quantization

# Generate test vectors
node generate-test-vectors.js --count 10000 --dim 768 --output test-vectors.db
```

**generate-test-vectors.js**:
```javascript
const { createAgentDBAdapter } = require('agentic-flow/reasoningbank');
const fs = require('fs');

async function generateTestVectors(count, dim) {
  const adapter = await createAgentDBAdapter({
    dbPath: 'test-vectors.db',
    quantizationType: 'none', // Unquantized baseline
  });

  console.log(`Generating ${count} test vectors (${dim} dimensions)...`);

  for (let i = 0; i < count; i++) {
    // Generate random embedding
    const embedding = Array.from({ length: dim }, () => Math.random() * 2 - 1);

    await adapter.insertPattern({
      id: `test-${i}`,
      type: 'embedding',
      domain: `domain-${i % 10}`,
      pattern_data: JSON.stringify({
        embedding,
        text: `Test vector ${i}`,
        metadata: { index: i }
      }),
      confidence: 0.9,
      usage_count: 0,
      success_count: 0,
      created_at: Date.now(),
      last_used: Date.now(),
    });

    if ((i + 1) % 1000 === 0) {
      console.log(`  Generated ${i + 1}/${count} vectors`);
    }
  }

  console.log('✅ Test vectors generated');
}

generateTestVectors(10000, 768);
```

### 2. Run Scalar Quantization

```bash
# Quantize database
python ../resources/scripts/quantize_vectors.py \
  --type scalar \
  --input test-vectors.db \
  --output test-vectors-scalar.db
```

**Expected Output**:
```
========================================
QUANTIZATION STATISTICS
========================================
Quantization Type: SCALAR
Vectors Processed: 10,000
Original Size: 29.30 MB
Quantized Size: 7.32 MB
Memory Reduction: 4.0x
Processing Time: 2.45 seconds
========================================

✅ Success! Quantized database saved to: test-vectors-scalar.db
```

## Test Cases

### Test Case 1.1: Memory Reduction Validation

**Goal**: Verify 4x memory reduction

```bash
# Check file sizes
ls -lh test-vectors*.db

# Expected:
# test-vectors.db:        29.30 MB (unquantized)
# test-vectors-scalar.db:  7.32 MB (scalar quantized)
```

**Pass Criteria**:
- ✅ Quantized database is 3.5x - 4.5x smaller
- ✅ Both databases contain same number of vectors

### Test Case 1.2: Search Accuracy Validation

**Goal**: Verify 98-99% search accuracy preserved

**accuracy-test.js**:
```javascript
const { createAgentDBAdapter } = require('agentic-flow/reasoningbank');

async function testAccuracy() {
  // Load both databases
  const originalAdapter = await createAgentDBAdapter({
    dbPath: 'test-vectors.db',
    quantizationType: 'none',
  });

  const quantizedAdapter = await createAgentDBAdapter({
    dbPath: 'test-vectors-scalar.db',
    quantizationType: 'scalar',
  });

  console.log('Testing search accuracy...\n');

  // Run 100 random queries
  const numQueries = 100;
  const k = 10;
  let totalRecall = 0;

  for (let i = 0; i < numQueries; i++) {
    // Generate random query
    const queryEmbedding = Array.from({ length: 768 }, () => Math.random() * 2 - 1);

    // Search both databases
    const originalResults = await originalAdapter.retrieveWithReasoning(queryEmbedding, { k });
    const quantizedResults = await quantizedAdapter.retrieveWithReasoning(queryEmbedding, { k });

    // Calculate recall@k
    const originalIds = new Set(originalResults.map(r => r.id));
    const quantizedIds = new Set(quantizedResults.map(r => r.id));

    const intersection = [...originalIds].filter(id => quantizedIds.has(id));
    const recall = intersection.length / k;

    totalRecall += recall;

    if ((i + 1) % 10 === 0) {
      console.log(`  Completed ${i + 1}/${numQueries} queries`);
    }
  }

  const avgRecall = totalRecall / numQueries;

  console.log('\n========================================');
  console.log('ACCURACY TEST RESULTS');
  console.log('========================================');
  console.log(`Queries: ${numQueries}`);
  console.log(`Recall@${k}: ${(avgRecall * 100).toFixed(2)}%`);
  console.log(`Target: 98-99%`);
  console.log(`Status: ${avgRecall >= 0.98 ? '✅ PASS' : '❌ FAIL'}`);
  console.log('========================================');
}

testAccuracy();
```

```bash
node accuracy-test.js
```

**Expected Output**:
```
Testing search accuracy...

  Completed 10/100 queries
  Completed 20/100 queries
  ...
  Completed 100/100 queries

========================================
ACCURACY TEST RESULTS
========================================
Queries: 100
Recall@10: 98.70%
Target: 98-99%
Status: ✅ PASS
========================================
```

**Pass Criteria**:
- ✅ Average recall ≥ 98%
- ✅ No queries with recall < 90%

### Test Case 1.3: Search Performance

**Goal**: Verify 3x search speed improvement

**performance-test.js**:
```javascript
const { createAgentDBAdapter } = require('agentic-flow/reasoningbank');

async function testPerformance() {
  const originalAdapter = await createAgentDBAdapter({
    dbPath: 'test-vectors.db',
    quantizationType: 'none',
  });

  const quantizedAdapter = await createAgentDBAdapter({
    dbPath: 'test-vectors-scalar.db',
    quantizationType: 'scalar',
  });

  const queryEmbedding = Array.from({ length: 768 }, () => Math.random() * 2 - 1);

  // Warm-up
  await originalAdapter.retrieveWithReasoning(queryEmbedding, { k: 10 });
  await quantizedAdapter.retrieveWithReasoning(queryEmbedding, { k: 10 });

  // Benchmark original
  const iterations = 100;
  const originalStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    await originalAdapter.retrieveWithReasoning(queryEmbedding, { k: 10 });
  }
  const originalEnd = performance.now();
  const originalAvg = (originalEnd - originalStart) / iterations;

  // Benchmark quantized
  const quantizedStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    await quantizedAdapter.retrieveWithReasoning(queryEmbedding, { k: 10 });
  }
  const quantizedEnd = performance.now();
  const quantizedAvg = (quantizedEnd - quantizedStart) / iterations;

  const speedup = originalAvg / quantizedAvg;

  console.log('========================================');
  console.log('PERFORMANCE TEST RESULTS');
  console.log('========================================');
  console.log(`Original (no quantization): ${originalAvg.toFixed(2)}ms`);
  console.log(`Quantized (scalar): ${quantizedAvg.toFixed(2)}ms`);
  console.log(`Speedup: ${speedup.toFixed(1)}x`);
  console.log(`Target: 3x`);
  console.log(`Status: ${speedup >= 2.5 ? '✅ PASS' : '❌ FAIL'}`);
  console.log('========================================');
}

testPerformance();
```

**Pass Criteria**:
- ✅ Speedup ≥ 2.5x
- ✅ Quantized search < 1ms per query

## Success Criteria

All test cases must pass:
1. ✅ Memory reduction: 3.5x - 4.5x
2. ✅ Search accuracy: ≥ 98% recall@10
3. ✅ Search performance: ≥ 2.5x speedup

## Cleanup

```bash
# Remove test files
rm -rf tests/quantization
```

## Notes

- Scalar quantization uses uint8 (1 byte) instead of float32 (4 bytes)
- Min/max normalization preserves relative distances
- Best for production applications requiring high accuracy
- ~1-2% accuracy loss is typical and acceptable
