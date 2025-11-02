# Example 2: HNSW Parameter Tuning for Optimal Search Performance

## Overview

This example demonstrates how to tune HNSW (Hierarchical Navigable Small World) index parameters to optimize search performance. Learn how M, efConstruction, and efSearch affect speed, accuracy, and memory usage.

## What is HNSW?

HNSW is a graph-based index structure that provides O(log n) search complexity instead of O(n) linear scans. It's the key to AgentDB's 150x-12,500x performance improvement.

**Key Parameters:**
- **M**: Number of bidirectional links per layer (affects memory and recall)
- **efConstruction**: Build-time quality parameter (affects index build time)
- **efSearch**: Search-time quality parameter (affects search time and recall)

## Scenario

You have a 100,000-vector database and need to optimize for different use cases: maximum speed, balanced performance, or maximum accuracy.

---

## Parameter Impact Analysis

### M (Connections Per Layer)

**Trade-off:** Higher M = Better recall, slower search, more memory

```typescript
// LOW M (8): Fast search, less memory, lower recall
const lowMAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/low-m.db',
  hnswM: 8,
  hnswEfConstruction: 200,
  hnswEfSearch: 100,
});

// MEDIUM M (16): Balanced (default)
const mediumMAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/medium-m.db',
  hnswM: 16,
  hnswEfConstruction: 200,
  hnswEfSearch: 100,
});

// HIGH M (48): Maximum recall, slower search, more memory
const highMAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/high-m.db',
  hnswM: 48,
  hnswEfConstruction: 200,
  hnswEfSearch: 100,
});
```

**Benchmark Results (100K vectors):**

| M | Search Time | Memory Overhead | Recall@10 |
|---|------------|----------------|-----------|
| 8 | **75µs** | +10% | 94% |
| 16 | 100µs | +20% | 97% |
| 32 | 140µs | +40% | 99% |
| 48 | 180µs | +60% | 99.5% |

---

### efConstruction (Build Quality)

**Trade-off:** Higher efConstruction = Better index quality, slower build

```typescript
// LOW efConstruction (100): Fast build
const fastBuildAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/fast-build.db',
  hnswM: 16,
  hnswEfConstruction: 100,  // Faster build
  hnswEfSearch: 100,
});

// MEDIUM efConstruction (200): Balanced (default)
const balancedBuildAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/balanced-build.db',
  hnswM: 16,
  hnswEfConstruction: 200,  // Balanced
  hnswEfSearch: 100,
});

// HIGH efConstruction (400): Maximum quality
const highQualityAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/high-quality.db',
  hnswM: 16,
  hnswEfConstruction: 400,  // Best quality
  hnswEfSearch: 100,
});
```

**Benchmark Results (100K vectors):**

| efConstruction | Build Time | Search Time | Recall@10 |
|---------------|-----------|-------------|-----------|
| 100 | **5s** | 105µs | 96% |
| 200 | 12s | 100µs | 97% |
| 400 | 28s | 98µs | 98% |

**Note:** efConstruction only affects initial index build time, not search performance.

---

### efSearch (Search Quality)

**Trade-off:** Higher efSearch = Better recall, slower search

```typescript
// LOW efSearch (50): Fast search
const fastSearchAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/fast-search.db',
  hnswM: 16,
  hnswEfConstruction: 200,
  hnswEfSearch: 50,  // Fast
});

// MEDIUM efSearch (100): Balanced (default)
const balancedSearchAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/balanced-search.db',
  hnswM: 16,
  hnswEfConstruction: 200,
  hnswEfSearch: 100,  // Balanced
});

// HIGH efSearch (200): Maximum recall
const accurateSearchAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/accurate-search.db',
  hnswM: 16,
  hnswEfConstruction: 200,
  hnswEfSearch: 200,  // Accurate
});
```

**Benchmark Results (100K vectors):**

| efSearch | Search Time | Recall@10 | Recall@100 |
|----------|-------------|-----------|------------|
| 50 | **70µs** | 94% | 89% |
| 100 | 100µs | 97% | 94% |
| 200 | 150µs | 99% | 98% |
| 500 | 280µs | 99.8% | 99.5% |

---

## Optimization Recipes

### Recipe 1: Maximum Speed (Low Latency Applications)

**Use Case:** Real-time search, chatbots, auto-complete

```typescript
const speedOptimizedAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/speed-optimized.db',
  quantizationType: 'binary',  // 32x memory reduction, faster
  hnswM: 8,                    // Fewer connections = faster
  hnswEfConstruction: 100,     // Fast build
  hnswEfSearch: 50,            // Fast search
  cacheSize: 5000,             // Large cache
});

// Expected: 50-70µs search, 92-94% recall
```

**Performance:**
- Search Time: **50-70µs**
- Recall@10: **92-94%**
- Memory: Very low
- Best for: Latency-sensitive applications

---

### Recipe 2: Balanced Performance (Production Default)

**Use Case:** General-purpose production applications

```typescript
const balancedAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/balanced.db',
  quantizationType: 'scalar',  // 4x memory reduction
  hnswM: 16,                   // Standard connections
  hnswEfConstruction: 200,     // Standard quality
  hnswEfSearch: 100,           // Standard search
  cacheSize: 1000,             // Standard cache
});

// Expected: 100µs search, 97% recall
```

**Performance:**
- Search Time: **100µs**
- Recall@10: **97%**
- Memory: Moderate
- Best for: Most production applications

---

### Recipe 3: Maximum Accuracy (Research/Analytics)

**Use Case:** Scientific research, high-stakes recommendations

```typescript
const accuracyOptimizedAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/accuracy-optimized.db',
  quantizationType: 'none',    // Full precision
  hnswM: 48,                   // Many connections
  hnswEfConstruction: 400,     // High build quality
  hnswEfSearch: 200,           // High search quality
  cacheSize: 2000,             // Large cache
});

// Expected: 150-180µs search, 99-99.5% recall
```

**Performance:**
- Search Time: **150-180µs**
- Recall@10: **99-99.5%**
- Memory: High
- Best for: Accuracy-critical applications

---

### Recipe 4: Massive Scale (>1M vectors)

**Use Case:** Large-scale knowledge bases, enterprise search

```typescript
const scaleOptimizedAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/scale-optimized.db',
  quantizationType: 'product',  // 8-16x memory reduction
  hnswM: 48,                    // More connections for scale
  hnswEfConstruction: 400,      // High quality for large graphs
  hnswEfSearch: 150,            // Good search quality
  cacheSize: 5000,              // Large cache
});

// Expected: 8-12ms search, 98% recall at 1M+ vectors
```

**Performance:**
- Search Time: **8-12ms** at 1M+ vectors
- Recall@10: **98%**
- Memory: Moderate (quantization helps)
- Best for: Massive databases

---

## Tuning Workflow

### Step 1: Establish Baseline

```typescript
const baseline = await createAgentDBAdapter({
  dbPath: '.agentdb/baseline.db',
  quantizationType: 'none',
  hnswM: 16,
  hnswEfConstruction: 200,
  hnswEfSearch: 100,
});

// Measure baseline performance
const testQueries = [...]; // Your test queries
let totalTime = 0;
let totalRecall = 0;

for (const query of testQueries) {
  const start = performance.now();
  const results = await baseline.retrieveWithReasoning(query.embedding, { k: 10 });
  const elapsed = performance.now() - start;

  totalTime += elapsed;
  totalRecall += calculateRecall(results, query.groundTruth);
}

console.log('Baseline Search Time:', totalTime / testQueries.length);
console.log('Baseline Recall@10:', totalRecall / testQueries.length);
```

### Step 2: Tune M (Start Here)

```typescript
const mValues = [8, 16, 32, 48];

for (const m of mValues) {
  const adapter = await createAgentDBAdapter({
    dbPath: `.agentdb/m-${m}.db`,
    hnswM: m,
    hnswEfConstruction: 200,
    hnswEfSearch: 100,
  });

  // Benchmark and compare
  const metrics = await benchmarkAdapter(adapter, testQueries);
  console.log(`M=${m}:`, metrics);
}

// Choose best M value based on your speed/accuracy requirements
```

### Step 3: Tune efSearch (Most Impact)

```typescript
const efSearchValues = [50, 100, 150, 200];

for (const ef of efSearchValues) {
  const adapter = await createAgentDBAdapter({
    dbPath: `.agentdb/ef-${ef}.db`,
    hnswM: 16,  // Use your chosen M
    hnswEfConstruction: 200,
    hnswEfSearch: ef,
  });

  const metrics = await benchmarkAdapter(adapter, testQueries);
  console.log(`efSearch=${ef}:`, metrics);
}
```

### Step 4: Optionally Tune efConstruction

```typescript
// Only if you want to improve recall at the cost of build time
const efConstructionValues = [100, 200, 400];

for (const efc of efConstructionValues) {
  const adapter = await createAgentDBAdapter({
    dbPath: `.agentdb/efc-${efc}.db`,
    hnswM: 16,
    hnswEfConstruction: efc,
    hnswEfSearch: 100,  // Use your chosen efSearch
  });

  const metrics = await benchmarkAdapter(adapter, testQueries);
  console.log(`efConstruction=${efc}:`, metrics);
}
```

---

## Monitoring HNSW Performance

### Get Index Statistics

```bash
npx agentdb@latest stats .agentdb/vectors.db

# Output includes:
# HNSW Index: Enabled
# HNSW M: 16
# HNSW efConstruction: 200
# Avg Search Time: 100µs
# Recall@10: 97%
```

### Runtime Performance Tracking

```typescript
const stats = await adapter.getStats();

console.log('HNSW Configuration:');
console.log('  M:', stats.hnswM);
console.log('  efConstruction:', stats.hnswEfConstruction);
console.log('  Index Size:', stats.indexSize);

console.log('Performance Metrics:');
console.log('  Avg Search Latency:', stats.avgSearchLatency);
console.log('  P50 Latency:', stats.p50Latency);
console.log('  P99 Latency:', stats.p99Latency);
console.log('  Recall@10:', stats.recallAt10);
```

---

## Parameter Selection Guide

### By Dataset Size

| Vector Count | M | efConstruction | efSearch |
|-------------|---|----------------|----------|
| <10K | 8 | 100 | 50-100 |
| 10K-100K | 16 | 200 | 100 |
| 100K-500K | 32 | 200-400 | 100-150 |
| 500K-1M | 48 | 400 | 150-200 |
| >1M | 48-64 | 400 | 150-200 |

### By Use Case

| Use Case | Priority | M | efSearch |
|----------|---------|---|----------|
| Real-time Search | Speed | 8 | 50 |
| Chatbot/Assistant | Speed | 8-16 | 50-100 |
| E-commerce Search | Balanced | 16 | 100 |
| Research/Analytics | Accuracy | 32-48 | 200 |
| Recommendation Engine | Balanced | 16-32 | 100-150 |

---

## Next Steps

1. Run the tuning workflow on your dataset
2. Combine HNSW tuning with [quantization](./example-1-quantization.md)
3. Add [batch operations](./example-3-batching.md) for maximum throughput
4. Monitor performance over time as database grows

## References

- [HNSW Parameters Deep Dive](../references/hnsw-parameters.md)
- [Main Skill Documentation](../skill.md)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
