# AgentDB Performance Optimization - Silver Tier Documentation

## Overview

This skill provides comprehensive performance optimization techniques for AgentDB vector databases, achieving 150x-12,500x performance improvements through quantization, HNSW indexing, caching strategies, and batch operations.

**Performance Highlights:**
- **150x faster** pattern search (100µs vs 15ms)
- **4-32x memory reduction** with quantization
- **500x faster** batch operations (2ms vs 1s for 100 vectors)
- **12,500x faster** large-scale queries (8ms vs 100s at 1M vectors)

## Quick Start

### Installation

```bash
# Install AgentDB via agentic-flow
npm install agentic-flow

# Run performance benchmarks
npx agentdb@latest benchmark
```

### Basic Optimization Setup

```typescript
import { createAgentDBAdapter } from 'agentic-flow/reasoningbank';

const adapter = await createAgentDBAdapter({
  dbPath: '.agentdb/optimized.db',
  quantizationType: 'binary',   // 32x memory reduction
  cacheSize: 1000,               // 1000-pattern cache
  enableLearning: true,
  enableReasoning: true,
});

// Ultra-fast vector search (100µs)
const results = await adapter.retrieveWithReasoning(queryEmbedding, {
  k: 10,
});
```

## Optimization Techniques

### 1. Quantization (4-32x Memory Reduction)

Quantization reduces memory usage by compressing vector representations while maintaining accuracy.

**Available Methods:**
- **Binary Quantization** (32x reduction): Best for large-scale deployments, 95-98% accuracy
- **Scalar Quantization** (4x reduction): Balanced performance/accuracy, 98-99% accuracy
- **Product Quantization** (8-16x reduction): High-dimensional vectors, 93-97% accuracy
- **No Quantization** (full precision): Maximum accuracy for small datasets

**Example:** [example-1-quantization.md](./examples/example-1-quantization.md)

### 2. HNSW Indexing (150x-12,500x Speedup)

Hierarchical Navigable Small World (HNSW) provides O(log n) search complexity instead of O(n) linear scans.

**Key Parameters:**
- **M**: Connections per layer (8-48 depending on scale)
- **efConstruction**: Build quality (100-400)
- **efSearch**: Search quality (50-200)

**Example:** [example-2-hnsw-tuning.md](./examples/example-2-hnsw-tuning.md)

### 3. Batch Operations (500x Throughput Improvement)

Batch processing dramatically reduces overhead for multiple operations.

**Benefits:**
- 500x faster inserts (2ms vs 1s for 100 vectors)
- Automatic transaction optimization
- Reduced database I/O

**Example:** [example-3-batching.md](./examples/example-3-batching.md)

### 4. Caching Strategies (>80% Hit Rate)

In-memory LRU cache for frequently accessed patterns reduces database queries.

**Configuration:**
- Small apps: 100-500 patterns
- Medium apps: 500-2000 patterns
- Large apps: 2000-5000 patterns

**Target:** >80% cache hit rate for optimal performance

## Optimization Recipes

### Maximum Speed (Sacrifice Accuracy)

```typescript
const adapter = await createAgentDBAdapter({
  quantizationType: 'binary',
  cacheSize: 5000,
  hnswM: 8,
  hnswEfSearch: 50,
});
// Expected: <50µs search, 90-95% accuracy
```

### Balanced Performance

```typescript
const adapter = await createAgentDBAdapter({
  quantizationType: 'scalar',
  cacheSize: 1000,
  hnswM: 16,
  hnswEfSearch: 100,
});
// Expected: <100µs search, 98-99% accuracy
```

### Maximum Accuracy

```typescript
const adapter = await createAgentDBAdapter({
  quantizationType: 'none',
  cacheSize: 2000,
  hnswM: 32,
  hnswEfSearch: 200,
});
// Expected: <200µs search, 100% accuracy
```

### Memory-Constrained (Mobile/Edge)

```typescript
const adapter = await createAgentDBAdapter({
  quantizationType: 'binary',
  cacheSize: 100,
  hnswM: 8,
});
// Expected: <100µs search, ~10MB for 100K vectors
```

## Scaling Guidelines

| Scale | Vector Count | Quantization | Cache | HNSW M | Memory Usage |
|-------|-------------|--------------|-------|--------|-------------|
| Small | <10K | None | 500 | 8 | ~30MB |
| Medium | 10K-100K | Scalar (4x) | 1000 | 16 | ~75MB |
| Large | 100K-1M | Binary (32x) | 2000 | 32 | ~96MB |
| Massive | >1M | Product (8-16x) | 5000 | 48 | ~200MB |

## Performance Benchmarks

**Test System:** AMD Ryzen 9 5950X, 64GB RAM

| Operation | Vector Count | Unoptimized | Optimized | Improvement |
|-----------|-------------|-------------|-----------|-------------|
| Search | 10K | 15ms | 100µs | **150x** |
| Search | 100K | 150ms | 120µs | **1,250x** |
| Search | 1M | 100s | 8ms | **12,500x** |
| Batch Insert (100) | - | 1s | 2ms | **500x** |
| Memory Usage | 1M | 3GB | 96MB | **32x** |

## Examples

1. **[Quantization Walkthrough](./examples/example-1-quantization.md)** - Reduce memory by 4-32x
2. **[HNSW Parameter Tuning](./examples/example-2-hnsw-tuning.md)** - Optimize search speed vs accuracy
3. **[Batch Operations](./examples/example-3-batching.md)** - 500x throughput improvement

## References

- **[Quantization Techniques](./references/quantization-techniques.md)** - Deep dive into compression methods
- **[HNSW Parameters](./references/hnsw-parameters.md)** - Complete parameter tuning guide
- **[Optimization Workflow](./graphviz/workflow.dot)** - Visual optimization pipeline

## Monitoring & Troubleshooting

### Get Statistics

```bash
npx agentdb@latest stats .agentdb/vectors.db
```

### Runtime Metrics

```typescript
const stats = await adapter.getStats();
console.log('Cache Hit Rate:', stats.cacheHitRate); // Target: >80%
console.log('Avg Search Latency:', stats.avgSearchLatency); // Target: <1ms
console.log('Database Size:', stats.dbSize);
```

### Common Issues

**High Memory Usage:**
```typescript
// Enable quantization
quantizationType: 'binary'  // 32x reduction
```

**Slow Searches:**
```typescript
// Increase cache, reduce search quality
cacheSize: 2000,
hnswEfSearch: 50  // Faster, slight accuracy loss
```

**Low Accuracy:**
```typescript
// Use lighter quantization
quantizationType: 'scalar',  // 4x reduction, 98-99% accuracy
hnswEfSearch: 200            // Higher search quality
```

## Learn More

- **GitHub:** https://github.com/ruvnet/agentic-flow/tree/main/packages/agentdb
- **Website:** https://agentdb.ruv.io
- **Main Skill:** [skill.md](./skill.md)

## Skill Tier: Silver

**Files:** 7+ comprehensive documentation files
- README.md (this file)
- skill.md (main skill)
- 3 examples
- 2 references
- 1 workflow diagram
