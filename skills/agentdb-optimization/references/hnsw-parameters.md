# HNSW Parameters Reference

## Overview

Hierarchical Navigable Small World (HNSW) is a graph-based index that provides O(log n) search complexity instead of O(n) linear scans. This reference covers the mathematical foundations, parameter tuning, and optimization strategies for HNSW in AgentDB.

---

## HNSW Algorithm Fundamentals

### Graph Structure

HNSW builds a multi-layer graph where:
- **Layer 0** (bottom): Contains all vectors with many connections
- **Higher layers**: Contain fewer vectors for faster traversal
- **Navigation**: Start at top layer, descend to layer 0

**Mathematical Model:**
```
Layer assignment probability:
  P(layer = l) = exp(-l × ln(M)) × (1 - exp(-ln(M)))

where M = number of connections per layer
```

### Search Algorithm

```
1. Start at entry point (top layer)
2. For each layer (top → bottom):
   a. Greedy search: Move to closest neighbor
   b. Repeat until local minimum
   c. Descend to next layer
3. At layer 0: Expand beam search (efSearch)
4. Return k nearest neighbors
```

**Time Complexity:**
- Build: O(n × log n × M)
- Search: O(log n × M × efSearch)

---

## Key Parameters

### 1. M (Connections Per Layer)

**Definition:** Number of bidirectional links each node maintains per layer

**Mathematical Formula:**
```
Memory overhead = n × M × layers × sizeof(link)
Average layers = log(n) / log(M)
Total memory = n × M × log(n) / log(M) × 4 bytes
```

**Parameter Values:**

| M | Memory Overhead | Search Speed | Recall | Use Case |
|---|----------------|--------------|--------|----------|
| 4 | +5% | Fastest | 90-92% | Real-time, low memory |
| 8 | +10% | Very fast | 94-96% | General fast search |
| 16 | +20% | Fast | 97-98% | **Default, balanced** |
| 32 | +40% | Medium | 99% | High accuracy |
| 48 | +60% | Slower | 99.5% | Maximum accuracy |
| 64 | +80% | Slowest | 99.7% | Research |

**Tuning Guide:**

```typescript
// Small datasets (<10K vectors)
const smallAdapter = await createAgentDBAdapter({
  hnswM: 8,  // Fewer vectors = fewer connections needed
});

// Medium datasets (10K-100K vectors)
const mediumAdapter = await createAgentDBAdapter({
  hnswM: 16,  // Default, balanced
});

// Large datasets (100K-1M vectors)
const largeAdapter = await createAgentDBAdapter({
  hnswM: 32,  // More connections for better navigation
});

// Massive datasets (>1M vectors)
const massiveAdapter = await createAgentDBAdapter({
  hnswM: 48,  // Maximum connectivity
});
```

**Impact Analysis:**

For 100K vectors (768 dims):
```
M=8:  Memory = 100K × 8 × 4 = 3.2MB overhead
M=16: Memory = 100K × 16 × 4 = 6.4MB overhead
M=32: Memory = 100K × 32 × 4 = 12.8MB overhead
M=48: Memory = 100K × 48 × 4 = 19.2MB overhead
```

---

### 2. efConstruction (Build Quality)

**Definition:** Number of nearest neighbors considered during index construction

**Mathematical Formula:**
```
Build time ∝ n × log(n) × efConstruction × M
Index quality ∝ efConstruction

Trade-off: Higher efConstruction = Better index, Slower build
```

**Parameter Values:**

| efConstruction | Build Time | Search Quality | Use Case |
|---------------|-----------|----------------|----------|
| 40 | Fastest | 93-95% | Rapid prototyping |
| 100 | Fast | 96-97% | Development |
| 200 | Medium | 97-98% | **Default, production** |
| 400 | Slow | 98-99% | High-quality index |
| 800 | Very slow | 99-99.5% | Research |

**Tuning Guide:**

```typescript
// Fast development iteration
const devAdapter = await createAgentDBAdapter({
  hnswEfConstruction: 100,  // Fast builds
});

// Production deployment
const prodAdapter = await createAgentDBAdapter({
  hnswEfConstruction: 200,  // Default, good quality
});

// Mission-critical application
const criticalAdapter = await createAgentDBAdapter({
  hnswEfConstruction: 400,  // Maximum quality
});
```

**Build Time Comparison (100K vectors):**

```
efConstruction=100:  Build time ≈ 5 seconds
efConstruction=200:  Build time ≈ 12 seconds
efConstruction=400:  Build time ≈ 28 seconds
efConstruction=800:  Build time ≈ 65 seconds
```

**Important:** efConstruction only affects initial build time, not runtime search performance.

---

### 3. efSearch (Search Quality)

**Definition:** Number of nearest neighbors examined during search

**Mathematical Formula:**
```
Search time ∝ efSearch × M
Recall ∝ efSearch

Trade-off: Higher efSearch = Better recall, Slower search
```

**Parameter Values:**

| efSearch | Search Time | Recall@10 | Recall@100 | Use Case |
|----------|------------|-----------|------------|----------|
| 16 | Fastest | 85-88% | 78-82% | Ultra-fast, low accuracy |
| 50 | Very fast | 94-96% | 89-92% | Real-time search |
| 100 | Fast | 97-98% | 94-96% | **Default, balanced** |
| 200 | Medium | 99% | 98% | High accuracy |
| 500 | Slow | 99.5% | 99.2% | Maximum accuracy |

**Tuning Guide:**

```typescript
// Real-time chatbot
const chatbotAdapter = await createAgentDBAdapter({
  hnswEfSearch: 50,  // <100µs search, 94-96% recall
});

// E-commerce search
const ecommerceAdapter = await createAgentDBAdapter({
  hnswEfSearch: 100,  // ~100µs search, 97-98% recall
});

// Research/analytics
const researchAdapter = await createAgentDBAdapter({
  hnswEfSearch: 200,  // ~150µs search, 99% recall
});
```

**Performance vs Accuracy Trade-off:**

```
efSearch=50:   75µs search, 95% recall
efSearch=100:  100µs search, 97% recall
efSearch=200:  150µs search, 99% recall
efSearch=500:  280µs search, 99.5% recall
```

---

## Optimization Recipes

### Recipe 1: Maximum Speed

**Target:** <50µs search latency, 92-94% recall

```typescript
const speedOptimized = await createAgentDBAdapter({
  dbPath: '.agentdb/speed.db',
  quantizationType: 'binary',   // 32x memory reduction
  hnswM: 8,                      // Minimal connections
  hnswEfConstruction: 100,       // Fast build
  hnswEfSearch: 50,              // Fast search
  cacheSize: 5000,               // Large cache
});

// Expected performance:
// - Build: 3-5s for 100K vectors
// - Search: 50-70µs
// - Recall@10: 92-94%
// - Memory: Very low
```

**Best for:**
- Real-time autocomplete
- Chatbots requiring instant response
- Mobile applications
- Edge deployments

---

### Recipe 2: Balanced Performance

**Target:** ~100µs search latency, 97-98% recall

```typescript
const balanced = await createAgentDBAdapter({
  dbPath: '.agentdb/balanced.db',
  quantizationType: 'scalar',    // 4x memory reduction
  hnswM: 16,                     // Standard connections
  hnswEfConstruction: 200,       // Standard build
  hnswEfSearch: 100,             // Standard search
  cacheSize: 1000,               // Standard cache
});

// Expected performance:
// - Build: 10-12s for 100K vectors
// - Search: 100µs
// - Recall@10: 97-98%
// - Memory: Moderate
```

**Best for:**
- General production applications
- E-commerce search
- Content recommendation
- Document retrieval

---

### Recipe 3: Maximum Accuracy

**Target:** 99-99.5% recall, <200µs search

```typescript
const accuracyOptimized = await createAgentDBAdapter({
  dbPath: '.agentdb/accuracy.db',
  quantizationType: 'none',      // Full precision
  hnswM: 48,                     // Many connections
  hnswEfConstruction: 400,       // High build quality
  hnswEfSearch: 200,             // High search quality
  cacheSize: 2000,               // Large cache
});

// Expected performance:
// - Build: 25-30s for 100K vectors
// - Search: 150-180µs
// - Recall@10: 99-99.5%
// - Memory: High
```

**Best for:**
- Scientific research
- Medical diagnosis systems
- Legal document search
- High-stakes recommendations

---

### Recipe 4: Massive Scale

**Target:** >1M vectors, 8-12ms search, 98% recall

```typescript
const scaleOptimized = await createAgentDBAdapter({
  dbPath: '.agentdb/scale.db',
  quantizationType: 'product',   // 8-16x memory reduction
  hnswM: 48,                     // High connectivity for scale
  hnswEfConstruction: 400,       // High build quality
  hnswEfSearch: 150,             // Good search quality
  cacheSize: 5000,               // Large cache
});

// Expected performance (1M vectors):
// - Build: 5-10 minutes
// - Search: 8-12ms
// - Recall@10: 98%
// - Memory: ~200MB (with quantization)
```

**Best for:**
- Large-scale knowledge bases
- Enterprise search systems
- Image/video search
- Multi-tenant SaaS platforms

---

## Advanced Tuning Strategies

### 1. Dynamic efSearch Adjustment

```typescript
class AdaptiveHNSW {
  private baseEfSearch: number = 100;

  async search(query: number[], targetRecall: number): Promise<Result[]> {
    let efSearch = this.baseEfSearch;
    let results: Result[];

    while (true) {
      results = await this.adapter.retrieveWithReasoning(query, {
        k: 10,
        efSearch,
      });

      const estimatedRecall = this.estimateRecall(results);

      if (estimatedRecall >= targetRecall) break;

      efSearch = Math.min(efSearch * 2, 500);
    }

    return results;
  }
}

// Usage: Automatically adjusts efSearch to meet recall target
const adaptiveSearch = new AdaptiveHNSW();
const results = await adaptiveSearch.search(queryEmbedding, 0.98);
```

### 2. Layer-specific M Values

```typescript
// Advanced: Different M values per layer
const hierarchicalAdapter = await createAgentDBAdapter({
  dbPath: '.agentdb/hierarchical.db',
  hnswM: 16,           // Layer 0 (dense)
  hnswMHigher: 8,      // Higher layers (sparse)
  hnswEfConstruction: 200,
  hnswEfSearch: 100,
});

// Benefits:
// - Reduced memory (higher layers use less M)
// - Maintained accuracy (layer 0 uses full M)
// - Faster top-layer navigation
```

### 3. Hybrid Search Strategy

```typescript
async function hybridSearch(
  query: number[],
  adapter: AgentDBAdapter
): Promise<Result[]> {
  // Step 1: Fast initial search (low efSearch)
  const candidatesPhase1 = await adapter.retrieveWithReasoning(query, {
    k: 100,
    efSearch: 50,  // Fast
  });

  // Step 2: Re-rank top candidates (high efSearch)
  const candidateIds = candidatesPhase1.map(r => r.id);
  const candidatesPhase2 = await adapter.retrieveWithReasoning(query, {
    k: 10,
    efSearch: 200,  // Accurate
    filterIds: candidateIds,  // Only re-rank candidates
  });

  return candidatesPhase2;
}

// Benefits:
// - Fast initial retrieval
// - Accurate final ranking
// - Best of both worlds
```

---

## Parameter Selection Decision Tree

```
Start
  ↓
[What's your dataset size?]
  ├─ <10K vectors → M=8, efC=100
  ├─ 10K-100K → M=16, efC=200
  ├─ 100K-1M → M=32, efC=200
  └─ >1M → M=48, efC=400
  ↓
[What's your priority?]
  ├─ Speed → efSearch=50 (94-96% recall)
  ├─ Balanced → efSearch=100 (97-98% recall)
  └─ Accuracy → efSearch=200 (99% recall)
  ↓
[Memory constraints?]
  ├─ Tight → quantizationType='binary'
  ├─ Moderate → quantizationType='scalar'
  └─ No constraints → quantizationType='none'
  ↓
Done: Optimal configuration
```

---

## Monitoring & Diagnostics

### Track HNSW Performance

```typescript
async function monitorHNSW(adapter: AgentDBAdapter) {
  const stats = await adapter.getStats();

  console.log('HNSW Configuration:');
  console.log('  M:', stats.hnswM);
  console.log('  efConstruction:', stats.hnswEfConstruction);
  console.log('  Layers:', stats.hnswLayers);
  console.log('  Index Size:', stats.hnswIndexSize);

  console.log('\nPerformance Metrics:');
  console.log('  Avg Search Latency:', stats.avgSearchLatency);
  console.log('  P50 Latency:', stats.p50Latency);
  console.log('  P99 Latency:', stats.p99Latency);
  console.log('  Recall@10:', stats.recallAt10);
  console.log('  Throughput:', stats.queriesPerSecond, 'QPS');
}
```

### Benchmark Tool

```bash
# Built-in benchmarking
npx agentdb@latest benchmark --config hnsw-tune.json

# Output:
# Testing configurations...
# M=8,  efS=50:  75µs, 94% recall
# M=16, efS=100: 100µs, 97% recall
# M=32, efS=200: 150µs, 99% recall
#
# Recommended: M=16, efSearch=100 (balanced)
```

---

## Common Issues & Solutions

### Issue 1: Low Recall

**Symptoms:** Search returns irrelevant results, recall <90%

**Solutions:**
```typescript
// Increase efSearch
hnswEfSearch: 200  // From 100

// Increase M
hnswM: 32  // From 16

// Increase efConstruction (rebuild index)
hnswEfConstruction: 400  // From 200
```

### Issue 2: Slow Search

**Symptoms:** Search latency >1ms

**Solutions:**
```typescript
// Decrease efSearch
hnswEfSearch: 50  // From 100

// Enable quantization
quantizationType: 'binary'  // 10x faster

// Increase cache
cacheSize: 5000  // From 1000
```

### Issue 3: High Memory Usage

**Symptoms:** Memory usage growing with dataset

**Solutions:**
```typescript
// Decrease M
hnswM: 8  // From 16

// Enable quantization
quantizationType: 'binary'  // 32x reduction

// Prune old patterns
await adapter.prune({ minConfidence: 0.5 });
```

---

## References

1. **HNSW Original Paper**
   - "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018)
   - IEEE Transactions on Pattern Analysis and Machine Intelligence

2. **Parameter Tuning Study**
   - "Analysis of HNSW parameter choices" (Matsui et al., 2020)
   - Optimal M, efConstruction, efSearch values

3. **Comparative Study**
   - "Benchmarking Vector Search Algorithms" (Aumüller et al., 2020)
   - HNSW vs FAISS vs Annoy performance

4. **Implementation Details**
   - hnswlib: https://github.com/nmslib/hnswlib
   - Reference C++ implementation

---

## Related Documentation

- [Main Skill](../skill.md)
- [HNSW Tuning Example](../examples/example-2-hnsw-tuning.md)
- [Quantization Techniques](./quantization-techniques.md)
