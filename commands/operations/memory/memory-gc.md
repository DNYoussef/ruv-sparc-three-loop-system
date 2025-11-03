---
name: memory-gc
description: Garbage collection and memory defragmentation with optimization
version: 2.0.0
category: memory
complexity: medium
tags: [memory, garbage-collection, optimization, defragmentation, cleanup]
author: ruv-SPARC Memory Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [memory-stats, memory-clear]
chains_with: [memory-clear, memory-stats, memory-export]
evidence_based_techniques: [self-consistency, plan-and-solve]
---

# /memory-gc - Garbage Collection & Defragmentation

## Overview

The `/memory-gc` command performs garbage collection, defragmentation, and optimization of memory storage. It reclaims unused space, consolidates fragmented data, and optimizes internal data structures for better performance.

## Purpose

- **Space Reclamation**: Free up unused memory
- **Defragmentation**: Consolidate fragmented storage
- **Index Optimization**: Rebuild indexes for faster access
- **Cache Cleanup**: Remove stale cache entries
- **Performance**: Improve memory access speed
- **Maintenance**: Regular housekeeping tasks

## Usage

```bash
# Basic garbage collection
npx claude-flow@alpha memory gc

# Aggressive cleanup
npx claude-flow@alpha memory gc --aggressive

# Defragmentation only
npx claude-flow@alpha memory gc --defrag-only

# Full optimization (GC + defrag + index rebuild)
npx claude-flow@alpha memory gc --full

# Dry run to preview
npx claude-flow@alpha memory gc --dry-run

# Target specific namespace
npx claude-flow@alpha memory gc --namespace "development/*"

# Schedule regular GC
npx claude-flow@alpha memory gc --schedule "daily-3am"

# GC with threshold
npx claude-flow@alpha memory gc --threshold 10MB

# Incremental GC (low impact)
npx claude-flow@alpha memory gc --incremental
```

## Parameters

### GC Modes

- `--mode <mode>` - Garbage collection mode
  - Options:
    - `standard` - Normal GC (default)
    - `aggressive` - Deep cleanup, slower
    - `incremental` - Low-impact, gradual
    - `defrag-only` - Just defragmentation

- `--aggressive` - Shortcut for aggressive mode
  - More thorough but slower

- `--defrag-only` - Only defragmentation
  - Skip garbage collection

- `--full` - Full optimization
  - GC + defrag + index rebuild

### Targeting

- `--namespace <pattern>` - Target specific namespace
  - Example: `development/*`

- `--threshold <size>` - Only run if fragmentation exceeds threshold
  - Format: `10MB`, `5%`

- `--age <duration>` - Clear entries older than duration
  - Format: `90d`, `6m`, `1y`

### Scheduling

- `--schedule <schedule>` - Schedule regular GC
  - Examples: `daily`, `weekly`, `daily-3am`

- `--cron <expression>` - Custom cron schedule
  - Example: `0 3 * * *` (daily at 3am)

### Options

- `--dry-run` - Preview GC without executing
  - Shows potential space savings

- `--verbose` - Detailed progress output

- `--stats` - Show GC statistics

- `--backup` - Backup before GC
  - Recommended for aggressive mode

## Examples

### Example 1: Basic Garbage Collection

```bash
npx claude-flow@alpha memory gc --stats

# Output:
# [Memory GC] Starting garbage collection...
# [Memory GC] Analyzing memory...
#
# PRE-GC STATE
# ────────────────────────────────────────
#   Total keys:           1,247
#   Total size:           12.4 MB
#   Fragmentation:        18.3%
#   Unused space:         2.4 MB
#
# [Memory GC] Phase 1: Mark and sweep...
# [Memory GC] Found 234 unreachable entries (1.2 MB)
# [Memory GC] Phase 2: Defragmentation...
# [Memory GC] Consolidated 89 fragments (450 KB saved)
# [Memory GC] Phase 3: Index optimization...
# [Memory GC] Rebuilt 15 indexes
# [Memory GC] Phase 4: Cache cleanup...
# [Memory GC] Cleared 45 stale cache entries (234 KB)
# [Memory GC] Complete!
#
# POST-GC STATE
# ────────────────────────────────────────
#   Total keys:           1,013
#   Total size:           10.1 MB
#   Fragmentation:        3.2%
#   Unused space:         0.3 MB
#
# STATISTICS
# ────────────────────────────────────────
#   Keys removed:         234
#   Space reclaimed:      2.3 MB (18.5%)
#   Fragmentation reduced: 15.1%
#   Execution time:       1.8s
```

### Example 2: Aggressive Cleanup

```bash
npx claude-flow@alpha memory gc --aggressive --backup --verbose

# Output:
# [Memory GC] Aggressive mode enabled
# [Memory GC] Creating backup...
# [Memory GC] Backup: .backups/pre-gc-20251101-103045.json
#
# [Memory GC] Phase 1: Deep mark and sweep...
# [Memory GC] Scanning all namespaces...
# [Memory GC] ✓ development/* (456 keys, 34 unreachable)
# [Memory GC] ✓ testing/* (234 keys, 67 unreachable)
# [Memory GC] ✓ planning/* (189 keys, 45 unreachable)
# [Memory GC] Found 234 unreachable entries
#
# [Memory GC] Phase 2: Reference analysis...
# [Memory GC] Checking circular references...
# [Memory GC] Found 12 orphaned entries
#
# [Memory GC] Phase 3: Aggressive defragmentation...
# [Memory GC] Consolidating storage...
# [Memory GC] Compacting indexes...
# [Memory GC] Optimizing metadata...
#
# [Memory GC] Phase 4: Deep cache cleanup...
# [Memory GC] Clearing all expired caches...
# [Memory GC] Rebuilding cache structures...
#
# [Memory GC] Phase 5: Index rebuild...
# [Memory GC] Rebuilding all indexes from scratch...
# [Memory GC] ✓ Rebuilt 15 indexes (450 KB saved)
#
# [Memory GC] Complete!
# [Memory GC] Space reclaimed: 3.1 MB (25.0%)
# [Memory GC] Performance improvement: ~40% faster access
```

### Example 3: Defragmentation Only

```bash
npx claude-flow@alpha memory gc --defrag-only --stats

# Output:
# [Memory GC] Defragmentation mode
# [Memory GC] Skipping garbage collection
#
# FRAGMENTATION ANALYSIS
# ────────────────────────────────────────
#   Current fragmentation:    18.3%
#   Fragmented blocks:        127
#   Largest gap:              234 KB
#   Total gaps:               2.4 MB
#
# [Memory GC] Defragmenting...
# [Memory GC] Consolidating 127 fragments...
# [Memory GC] Moving data to contiguous blocks...
# [Memory GC] Complete!
#
# POST-DEFRAG STATE
# ────────────────────────────────────────
#   Fragmentation:           3.2% (↓ 15.1%)
#   Fragmented blocks:       12 (↓ 115)
#   Largest gap:             45 KB (↓ 189 KB)
#   Total gaps:              0.3 MB (↓ 2.1 MB)
#
# Performance improvement:   ~25% faster access
```

### Example 4: Namespace-Specific GC

```bash
npx claude-flow@alpha memory gc \
  --namespace "development/temp/*" \
  --aggressive

# Output:
# [Memory GC] Targeting namespace: development/temp/*
# [Memory GC] Found 156 keys (1.2 MB)
# [Memory GC] Running aggressive cleanup...
#
# [Memory GC] Phase 1: Age-based cleanup...
# [Memory GC] Removing entries older than 7 days...
# [Memory GC] Removed 89 old entries (645 KB)
#
# [Memory GC] Phase 2: Defragmentation...
# [Memory GC] Consolidated temp storage (245 KB saved)
#
# [Memory GC] Phase 3: Index cleanup...
# [Memory GC] Rebuilt temp indexes
#
# [Memory GC] Complete!
# [Memory GC] Namespace size: 1.2 MB → 310 KB (74% reduction)
```

### Example 5: Incremental GC

```bash
npx claude-flow@alpha memory gc --incremental --verbose

# Output:
# [Memory GC] Incremental mode (low-impact)
# [Memory GC] Running in 10 small batches...
#
# [Batch 1/10] Processing 125 keys...
# [Batch 1/10] Freed 145 KB
# [Batch 2/10] Processing 125 keys...
# [Batch 2/10] Freed 89 KB
# [Batch 3/10] Processing 125 keys...
# [Batch 3/10] Freed 234 KB
# ...
# [Batch 10/10] Processing 122 keys...
# [Batch 10/10] Freed 67 KB
#
# [Memory GC] Incremental GC complete!
# [Memory GC] Total freed: 1.8 MB
# [Memory GC] Impact: Minimal (processing time distributed)
```

### Example 6: Scheduled GC

```bash
# Schedule daily GC at 3am
npx claude-flow@alpha memory gc --schedule "daily-3am"

# Output:
# [Memory GC] Scheduling daily GC at 3:00 AM
# [Memory GC] Schedule created: daily-3am-gc
# [Memory GC] Next run: 2025-11-02 03:00:00
# [Memory GC] Mode: standard
# [Memory GC] Options: --stats
#
# Scheduled jobs:
#   ✓ daily-3am-gc (next: in 16h 30m)
#
# To view logs:
#   npx claude-flow@alpha memory gc --logs daily-3am-gc
#
# To cancel schedule:
#   npx claude-flow@alpha memory gc --unschedule daily-3am-gc
```

## GC Algorithms

### 1. Mark and Sweep

```typescript
async function markAndSweep(): Promise<GCResult> {
  // Phase 1: Mark reachable entries
  const reachable = new Set<string>();
  const roots = await getRootKeys();

  for (const root of roots) {
    await markReachable(root, reachable);
  }

  // Phase 2: Sweep unreachable entries
  const allKeys = await listAllKeys();
  const unreachable = allKeys.filter(key => !reachable.has(key));

  // Phase 3: Remove unreachable
  for (const key of unreachable) {
    await remove(key);
  }

  return {
    marked: reachable.size,
    swept: unreachable.length,
    freed: calculateSize(unreachable)
  };
}
```

### 2. Reference Counting

```typescript
async function referenceCount(): Promise<Map<string, number>> {
  const refCounts = new Map<string, number>();

  // Count references to each entry
  const allEntries = await retrieveAll();

  for (const entry of allEntries) {
    // Initialize count
    if (!refCounts.has(entry.key)) {
      refCounts.set(entry.key, 0);
    }

    // Count references in value
    const refs = extractReferences(entry.value);
    for (const ref of refs) {
      refCounts.set(ref, (refCounts.get(ref) || 0) + 1);
    }
  }

  return refCounts;
}
```

### 3. Generational GC

```typescript
interface Generation {
  young: MemoryEntry[]; // Recently created
  mature: MemoryEntry[]; // Survived several GC cycles
  old: MemoryEntry[]; // Long-lived entries
}

async function generationalGC(): Promise<GCResult> {
  const generations = await categorizeByAge();

  // GC young generation frequently (most garbage here)
  const youngResult = await gcGeneration(generations.young);

  // GC mature less frequently
  const matureResult = await gcGeneration(generations.mature);

  // GC old rarely (usually stable)
  const oldResult = await gcGeneration(generations.old);

  return mergeResults([youngResult, matureResult, oldResult]);
}
```

## Implementation Details

### Full GC Process

```typescript
interface GCOptions {
  mode?: 'standard' | 'aggressive' | 'incremental' | 'defrag-only';
  namespace?: string;
  threshold?: string;
  backup?: boolean;
  dryRun?: boolean;
}

async function memoryGC(options: GCOptions): Promise<GCResult> {
  // 1. Pre-GC analysis
  const preStats = await collectStats();

  // 2. Backup if requested
  if (options.backup) {
    await createBackup();
  }

  // 3. Run GC phases based on mode
  let result: GCResult;

  switch (options.mode) {
    case 'aggressive':
      result = await aggressiveGC(options);
      break;
    case 'incremental':
      result = await incrementalGC(options);
      break;
    case 'defrag-only':
      result = await defragmentation(options);
      break;
    default:
      result = await standardGC(options);
  }

  // 4. Post-GC analysis
  const postStats = await collectStats();

  // 5. Calculate improvements
  return {
    ...result,
    improvements: calculateImprovements(preStats, postStats)
  };
}
```

### Defragmentation

```typescript
async function defragmentation(): Promise<DefragResult> {
  // 1. Analyze fragmentation
  const fragments = await analyzeFragmentation();

  // 2. Plan consolidation
  const plan = createConsolidationPlan(fragments);

  // 3. Execute moves
  for (const move of plan) {
    await moveData(move.from, move.to);
  }

  // 4. Update indexes
  await rebuildIndexes();

  return {
    fragmentsConsolidated: fragments.length,
    spaceReclaimed: calculateSpaceSaved(plan)
  };
}
```

## Integration with Other Commands

### Chains With

**Preparation**:
- `/memory-stats` - Analyze before GC
- `/memory-export` - Backup before aggressive GC

**After GC**:
- `/memory-stats` - Verify improvements
- `/memory-clear` - Additional cleanup
- `/metrics-export` - Track GC metrics

**Workflow**:
```bash
# Weekly maintenance workflow
npx claude-flow@alpha memory stats --health-check
npx claude-flow@alpha memory export --backup "weekly-backup.json"
npx claude-flow@alpha memory gc --aggressive --stats
npx claude-flow@alpha memory clear --namespace "temp/*" --age 7d
npx claude-flow@alpha memory stats --compare pre-gc
```

## Best Practices

### 1. Regular Scheduling
```bash
# Schedule weekly GC
npx claude-flow@alpha memory gc --schedule "weekly-sunday-3am" --aggressive
```

### 2. Backup Before Aggressive GC
```bash
# Always backup for aggressive mode
npx claude-flow@alpha memory gc --aggressive --backup
```

### 3. Use Incremental for Production
```bash
# Low-impact GC for production
npx claude-flow@alpha memory gc --incremental
```

## See Also

- `/memory-clear` - Clear memory by namespace
- `/memory-stats` - Memory statistics
- `/memory-export` - Export backups
- `/state-checkpoint` - State checkpoints

---

**Version**: 2.0.0
**Last Updated**: 2025-11-01
**Maintained By**: ruv-SPARC Memory Team
