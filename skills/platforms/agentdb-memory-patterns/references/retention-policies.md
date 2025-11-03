# Retention Policies and Memory Optimization

## Overview

This document covers strategies for managing memory capacity, implementing garbage collection, and optimizing long-term storage in AgentDB-based systems.

## Retention Policy Types

### 1. Time-Based Retention (TTL)

**Description**: Automatically expire memories after a fixed duration.

**Configuration**:
```typescript
interface TTLPolicy {
  shortTerm: 24 * 60 * 60 * 1000;      // 24 hours
  midTerm: 7 * 24 * 60 * 60 * 1000;    // 7 days
  longTerm: 30 * 24 * 60 * 60 * 1000;  // 30 days
  permanent: Infinity;                  // Never expire
}
```

**Implementation**:
```typescript
class TTLMemoryManager {
  async pruneExpired(memoryType: 'short' | 'mid' | 'long') {
    const ttl = this.getTTL(memoryType);
    const cutoffTime = Date.now() - ttl;

    await db.deletePatterns({
      domain: memoryType,
      created_at: { $lt: cutoffTime }
    });
  }

  async scheduleExpiration() {
    setInterval(() => this.pruneExpired('short'), 60 * 60 * 1000);  // Hourly
    setInterval(() => this.pruneExpired('mid'), 24 * 60 * 60 * 1000); // Daily
    setInterval(() => this.pruneExpired('long'), 7 * 24 * 60 * 60 * 1000); // Weekly
  }
}
```

**Use Cases**:
- Session data (short-term)
- Cache invalidation (mid-term)
- Audit logs (long-term)
- Temporary experiments

**Pros**:
- Simple to implement
- Predictable memory usage
- Automatic cleanup

**Cons**:
- Deletes important data if not accessed
- Doesn't consider usage patterns
- Fixed retention regardless of value

---

### 2. Importance-Based Retention

**Description**: Retain high-value memories regardless of age.

**Configuration**:
```typescript
interface ImportancePolicy {
  minConfidence: number;       // 0-1 threshold
  minUsageCount: number;       // Access frequency
  minSuccessCount: number;     // Success rate
  importanceScore: (pattern: Pattern) => number;
}
```

**Importance Scoring**:
```typescript
function calculateImportance(pattern: Pattern): number {
  const confidenceWeight = 0.4;
  const usageWeight = 0.3;
  const successWeight = 0.2;
  const recencyWeight = 0.1;

  const normalizedUsage = Math.min(1, pattern.usage_count / 100);
  const successRate = pattern.success_count / Math.max(1, pattern.usage_count);
  const recencyScore = 1 - (Date.now() - pattern.last_used) / (365 * 24 * 60 * 60 * 1000);

  return (
    confidenceWeight * pattern.confidence +
    usageWeight * normalizedUsage +
    successWeight * successRate +
    recencyWeight * Math.max(0, recencyScore)
  );
}
```

**Implementation**:
```typescript
class ImportanceBasedRetention {
  async prune(maxPatterns: number, minImportance: number = 0.3) {
    // Get all patterns with importance scores
    const patterns = await db.getAllPatterns();
    const scored = patterns.map(p => ({
      pattern: p,
      importance: calculateImportance(p)
    }));

    // Sort by importance (descending)
    scored.sort((a, b) => b.importance - a.importance);

    // Keep top N patterns and those above threshold
    const toKeep = scored
      .slice(0, maxPatterns)
      .filter(s => s.importance >= minImportance)
      .map(s => s.pattern.id);

    // Delete the rest
    const toDelete = scored
      .filter(s => !toKeep.includes(s.pattern.id))
      .map(s => s.pattern.id);

    await db.deletePatterns({ id: { $in: toDelete } });

    return {
      kept: toKeep.length,
      deleted: toDelete.length,
      averageImportance: toKeep.reduce((sum, id) =>
        sum + scored.find(s => s.pattern.id === id)!.importance, 0
      ) / toKeep.length
    };
  }
}
```

**Use Cases**:
- Knowledge bases (retain valuable facts)
- Learned patterns (keep successful strategies)
- User preferences (preserve important settings)

**Pros**:
- Retains high-value data
- Adapts to usage patterns
- Quality over quantity

**Cons**:
- More complex to implement
- Requires importance metric design
- May retain outdated but frequently used data

---

### 3. Capacity-Based Retention (LRU/LFU)

**Description**: Maintain fixed capacity using eviction policies.

**LRU (Least Recently Used)**:
```typescript
class LRURetention {
  private maxSize: number;

  async enforce() {
    const count = await db.countPatterns();

    if (count > this.maxSize) {
      const toRemove = count - this.maxSize;

      // Delete least recently used
      await db.deletePatterns({
        orderBy: 'last_used ASC',
        limit: toRemove
      });
    }
  }

  async onAccess(patternId: string) {
    // Update last_used timestamp
    await db.updatePattern(patternId, {
      last_used: Date.now()
    });
  }
}
```

**LFU (Least Frequently Used)**:
```typescript
class LFURetention {
  private maxSize: number;

  async enforce() {
    const count = await db.countPatterns();

    if (count > this.maxSize) {
      const toRemove = count - this.maxSize;

      // Delete least frequently used
      await db.deletePatterns({
        orderBy: 'usage_count ASC',
        limit: toRemove
      });
    }
  }

  async onAccess(patternId: string) {
    // Increment usage_count
    await db.updatePattern(patternId, {
      usage_count: '+1',
      last_used: Date.now()
    });
  }
}
```

**Use Cases**:
- Cache management
- Fixed-size memory systems
- Resource-constrained environments

**Pros**:
- Bounded memory usage
- Predictable performance
- Well-studied algorithms

**Cons**:
- May evict important data
- Doesn't consider data value
- Cold start problem (LFU)

---

### 4. Hybrid Retention Policies

**Description**: Combine multiple strategies for optimal retention.

**Configuration**:
```typescript
interface HybridPolicy {
  shortTerm: {
    type: 'ttl';
    duration: 24 * 60 * 60 * 1000; // 24h
  };
  midTerm: {
    type: 'lru';
    maxSize: 10000;
  };
  longTerm: {
    type: 'importance';
    minConfidence: 0.5;
    maxPatterns: 50000;
  };
}
```

**Implementation**:
```typescript
class HybridRetention {
  async applyPolicy(domain: string, policy: any) {
    switch (policy.type) {
      case 'ttl':
        await this.pruneTTL(domain, policy.duration);
        break;
      case 'lru':
        await this.enforceLRU(domain, policy.maxSize);
        break;
      case 'importance':
        await this.pruneByImportance(domain, policy);
        break;
      case 'hybrid':
        // Apply multiple policies in sequence
        await this.pruneTTL(domain, policy.ttl);
        await this.enforceLRU(domain, policy.maxSize);
        await this.pruneByImportance(domain, policy);
        break;
    }
  }

  async optimizeAllMemory() {
    await this.applyPolicy('short-term', { type: 'ttl', duration: 24 * 60 * 60 * 1000 });
    await this.applyPolicy('mid-term', { type: 'lru', maxSize: 10000 });
    await this.applyPolicy('long-term', {
      type: 'importance',
      minConfidence: 0.5,
      maxPatterns: 50000
    });
  }
}
```

---

## Memory Consolidation Strategies

### 1. Deduplication

**Goal**: Remove duplicate or near-duplicate memories.

```typescript
async function deduplicateMemories(threshold: number = 0.95) {
  const patterns = await db.getAllPatterns();
  const duplicates = new Map<string, string[]>();

  // Find similar patterns
  for (let i = 0; i < patterns.length; i++) {
    const similar = await db.searchPatterns(patterns[i].embedding, {
      k: 10,
      threshold
    });

    for (const s of similar) {
      if (s.id !== patterns[i].id) {
        if (!duplicates.has(patterns[i].id)) {
          duplicates.set(patterns[i].id, []);
        }
        duplicates.get(patterns[i].id)!.push(s.id);
      }
    }
  }

  // Merge duplicates
  for (const [keepId, mergeIds] of duplicates.entries()) {
    const keep = patterns.find(p => p.id === keepId)!;
    const merge = patterns.filter(p => mergeIds.includes(p.id));

    // Aggregate statistics
    const merged = {
      ...keep,
      usage_count: keep.usage_count + merge.reduce((sum, m) => sum + m.usage_count, 0),
      success_count: keep.success_count + merge.reduce((sum, m) => sum + m.success_count, 0),
      confidence: Math.max(keep.confidence, ...merge.map(m => m.confidence))
    };

    await db.updatePattern(keepId, merged);
    await db.deletePatterns({ id: { $in: mergeIds } });
  }
}
```

### 2. Semantic Clustering

**Goal**: Group related memories and summarize.

```typescript
async function clusterAndSummarize(k: number = 100) {
  const patterns = await db.getAllPatterns();
  const embeddings = patterns.map(p => p.embedding);

  // K-means clustering
  const clusters = kMeansClustering(embeddings, k);

  // Summarize each cluster
  for (const cluster of clusters) {
    const clusterPatterns = cluster.indices.map(i => patterns[i]);

    // Create cluster summary
    const summary = {
      centroid: cluster.centroid,
      size: clusterPatterns.length,
      avgConfidence: clusterPatterns.reduce((sum, p) => sum + p.confidence, 0) / clusterPatterns.length,
      representative: clusterPatterns.reduce((best, p) =>
        p.confidence > best.confidence ? p : best
      )
    };

    // Store cluster metadata
    await db.storeFact('cluster-metadata', `cluster-${cluster.id}`, summary);
  }
}
```

### 3. Hierarchical Compression

**Goal**: Build hierarchical summaries for efficient retrieval.

```typescript
interface MemoryHierarchy {
  level0: Pattern[];          // Raw memories
  level1: Summary[];          // Daily summaries
  level2: Summary[];          // Weekly summaries
  level3: Summary[];          // Monthly summaries
}

async function buildHierarchy(patterns: Pattern[]): Promise<MemoryHierarchy> {
  // Group by day
  const byDay = groupByTimeWindow(patterns, 24 * 60 * 60 * 1000);
  const level1 = byDay.map(day => summarize(day));

  // Group by week
  const byWeek = groupByTimeWindow(level1, 7 * 24 * 60 * 60 * 1000);
  const level2 = byWeek.map(week => summarize(week));

  // Group by month
  const byMonth = groupByTimeWindow(level2, 30 * 24 * 60 * 60 * 1000);
  const level3 = byMonth.map(month => summarize(month));

  return {
    level0: patterns,
    level1,
    level2,
    level3
  };
}

function summarize(patterns: Pattern[]): Summary {
  return {
    count: patterns.length,
    timeRange: {
      start: Math.min(...patterns.map(p => p.created_at)),
      end: Math.max(...patterns.map(p => p.created_at))
    },
    centroid: computeCentroid(patterns.map(p => p.embedding)),
    statistics: {
      avgConfidence: patterns.reduce((sum, p) => sum + p.confidence, 0) / patterns.length,
      totalUsage: patterns.reduce((sum, p) => sum + p.usage_count, 0)
    }
  };
}
```

---

## Automated Maintenance Schedules

### Hourly Tasks
```typescript
async function hourlyMaintenance() {
  // 1. Prune expired short-term memories
  await ttlManager.pruneExpired('short');

  // 2. Update access statistics
  await updateAccessMetrics();

  // 3. Compact hot cache
  await compactCache();
}
```

### Daily Tasks
```typescript
async function dailyMaintenance() {
  // 1. Prune expired mid-term memories
  await ttlManager.pruneExpired('mid');

  // 2. Deduplicate similar patterns
  await deduplicateMemories(0.95);

  // 3. Update importance scores
  await recalculateImportance();

  // 4. Generate daily summaries
  await generateDailySummary();
}
```

### Weekly Tasks
```typescript
async function weeklyMaintenance() {
  // 1. Full consolidation
  await memory.consolidate({
    strategy: 'hybrid',
    minConfidence: 0.3,
    maxPatterns: 50000
  });

  // 2. Cluster and summarize
  await clusterAndSummarize(100);

  // 3. Build memory hierarchy
  await buildMemoryHierarchy();

  // 4. Optimize indexes
  await db.optimizeIndexes();

  // 5. Generate weekly report
  await generateWeeklyReport();
}
```

### Monthly Tasks
```typescript
async function monthlyMaintenance() {
  // 1. Deep analysis and archival
  await archiveOldMemories(90 * 24 * 60 * 60 * 1000); // 90 days

  // 2. Performance benchmarks
  await runPerformanceBenchmarks();

  // 3. Capacity planning
  await analyzeGrowthTrends();

  // 4. Backup and export
  await exportMemorySnapshot();
}
```

---

## Performance Metrics

### Memory Usage Tracking
```typescript
interface MemoryMetrics {
  totalPatterns: number;
  totalSize: number;          // bytes
  byDomain: Map<string, number>;
  avgPatternSize: number;
  compressionRatio: number;
}

async function getMemoryMetrics(): Promise<MemoryMetrics> {
  const stats = await db.getStatistics();

  return {
    totalPatterns: stats.count,
    totalSize: stats.sizeBytes,
    byDomain: stats.byDomain,
    avgPatternSize: stats.sizeBytes / stats.count,
    compressionRatio: stats.quantizationType === 'binary' ? 32 :
                      stats.quantizationType === 'scalar' ? 4 : 1
  };
}
```

### Retention Effectiveness
```typescript
interface RetentionMetrics {
  retentionRate: number;      // % of patterns retained
  hitRate: number;            // % of queries finding results
  avgImportance: number;      // Average importance of retained patterns
  storageEfficiency: number;  // bytes per valuable pattern
}

async function measureRetentionEffectiveness(): Promise<RetentionMetrics> {
  const beforeCount = await db.countPatterns();
  await applyRetentionPolicies();
  const afterCount = await db.countPatterns();

  const hitRate = await measureHitRate(1000); // Test 1000 queries
  const patterns = await db.getAllPatterns();
  const avgImportance = patterns.reduce((sum, p) =>
    sum + calculateImportance(p), 0
  ) / patterns.length;

  const metrics = await getMemoryMetrics();

  return {
    retentionRate: afterCount / beforeCount,
    hitRate,
    avgImportance,
    storageEfficiency: metrics.totalSize / afterCount
  };
}
```

---

## Best Practices

1. **Start Conservative**: Begin with longer retention, then optimize
2. **Monitor Metrics**: Track hit rate and storage efficiency
3. **Gradual Pruning**: Prune 10-20% at a time, not all at once
4. **Preserve Variety**: Use MMR to maintain diverse memories
5. **Archive Before Deletion**: Export old data before pruning
6. **Test Policies**: Validate retention policies on sample data
7. **Document Decisions**: Log why patterns were pruned
8. **User Control**: Allow users to pin important memories

## Related References

- [Memory Patterns](./memory-patterns.md) - Pattern types and characteristics
- [Performance Tuning](./performance-tuning.md) - Optimization techniques
