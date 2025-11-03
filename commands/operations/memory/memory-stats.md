---
name: memory-stats
description: Detailed memory usage statistics and analysis
version: 2.0.0
category: memory
complexity: low
tags: [memory, statistics, analysis, monitoring, optimization]
author: ruv-SPARC Memory Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [memory-retrieve]
chains_with: [memory-gc, memory-clear, metrics-export]
evidence_based_techniques: [self-consistency]
---

# /memory-stats - Memory Usage Statistics

## Overview

The `/memory-stats` command provides comprehensive memory usage statistics, analysis, and recommendations for optimization. It helps identify memory bottlenecks, track growth trends, and optimize memory allocation.

## Purpose

- **Usage Analysis**: Understand memory consumption patterns
- **Optimization**: Identify cleanup opportunities
- **Monitoring**: Track memory growth over time
- **Debugging**: Find memory leaks and bloat
- **Capacity Planning**: Forecast future memory needs
- **Cost Analysis**: Understand storage costs

## Usage

```bash
# Basic statistics
npx claude-flow@alpha memory stats

# Detailed breakdown
npx claude-flow@alpha memory stats --detailed

# Statistics for specific namespace
npx claude-flow@alpha memory stats --namespace "development/*"

# Compare with baseline
npx claude-flow@alpha memory stats --compare baseline

# Export statistics
npx claude-flow@alpha memory stats --export stats.json

# Show growth trends
npx claude-flow@alpha memory stats --trends --period 30d

# Identify large entries
npx claude-flow@alpha memory stats --top-keys 20

# Memory health check
npx claude-flow@alpha memory stats --health-check

# Real-time monitoring
npx claude-flow@alpha memory stats --watch --interval 5s
```

## Parameters

### Display Options

- `--detailed` - Show detailed breakdown
  - Includes per-namespace statistics

- `--summary` - Summary only (default)
  - High-level overview

- `--format <format>` - Output format
  - Options: `table`, `json`, `yaml`, `csv`

### Filtering

- `--namespace <pattern>` - Filter by namespace
  - Supports wildcards: `development/*`

- `--tags <filter>` - Filter by metadata tags
  - Format: `key:value`

### Analysis

- `--trends` - Show growth trends
  - Requires `--period`

- `--period <duration>` - Analysis period
  - Format: `7d`, `30d`, `90d`

- `--compare <baseline>` - Compare with baseline
  - Baseline: checkpoint name or file

- `--top-keys <count>` - Show largest keys
  - Default: 10

- `--health-check` - Run health analysis
  - Identifies issues and recommendations

### Output

- `--export <path>` - Export to file
  - Formats: `.json`, `.yaml`, `.csv`

- `--watch` - Real-time monitoring
  - Requires `--interval`

- `--interval <duration>` - Update interval
  - Format: `5s`, `1m`, `5m`

## Examples

### Example 1: Basic Statistics

```bash
npx claude-flow@alpha memory stats

# Output:
# ════════════════════════════════════════════════════════════
#                    MEMORY STATISTICS
# ════════════════════════════════════════════════════════════
#
# Total Keys:           1,247
# Total Size:           12.4 MB
# Namespaces:           15
#
# By Namespace:
#   development/*       456 keys (5.2 MB) - 41.9%
#   testing/*           234 keys (2.1 MB) - 16.9%
#   planning/*          189 keys (1.8 MB) - 14.5%
#   research/*          145 keys (1.4 MB) - 11.3%
#   production/*        89 keys (0.9 MB) - 7.1%
#   other               134 keys (1.0 MB) - 8.3%
#
# Storage Status:
#   Used:               12.4 MB
#   Available:          987.6 MB
#   Utilization:        1.2%
#
# Health:               ✓ Good
# Recommendations:      No action needed
```

### Example 2: Detailed Breakdown

```bash
npx claude-flow@alpha memory stats --detailed

# Output:
# ════════════════════════════════════════════════════════════
#                 DETAILED MEMORY STATISTICS
# ════════════════════════════════════════════════════════════
#
# OVERVIEW
# ────────────────────────────────────────────────────────────
# Total Keys:           1,247
# Total Size:           12.4 MB
# Average Key Size:     10.2 KB
# Median Key Size:      4.5 KB
# Largest Key:          234 KB (development/coder/large-dataset)
#
# NAMESPACE BREAKDOWN
# ────────────────────────────────────────────────────────────
# development/coder
#   Keys:               234 keys
#   Size:               3.2 MB
#   Avg Size:           14.0 KB
#   Oldest:             45 days
#   Newest:             2 hours
#   Growth (7d):        +12 keys (+145 KB)
#
# development/tester
#   Keys:               156 keys
#   Size:               1.4 MB
#   Avg Size:           9.2 KB
#   Oldest:             30 days
#   Newest:             1 day
#   Growth (7d):        +8 keys (+89 KB)
#
# AGE DISTRIBUTION
# ────────────────────────────────────────────────────────────
#   < 24 hours:         234 keys (2.1 MB) - 18.8%
#   1-7 days:           456 keys (4.5 MB) - 36.6%
#   7-30 days:          389 keys (3.8 MB) - 31.2%
#   30-90 days:         123 keys (1.5 MB) - 9.9%
#   > 90 days:          45 keys (0.5 MB) - 3.6%
#
# SIZE DISTRIBUTION
# ────────────────────────────────────────────────────────────
#   < 1 KB:             345 keys (234 KB) - 27.7%
#   1-10 KB:            567 keys (3.2 MB) - 45.5%
#   10-100 KB:          289 keys (7.8 MB) - 23.2%
#   100 KB - 1 MB:      42 keys (1.1 MB) - 3.4%
#   > 1 MB:             4 keys (89 KB) - 0.3%
```

### Example 3: Growth Trends

```bash
npx claude-flow@alpha memory stats --trends --period 30d

# Output:
# ════════════════════════════════════════════════════════════
#                    GROWTH TRENDS (30 days)
# ════════════════════════════════════════════════════════════
#
# OVERALL GROWTH
# ────────────────────────────────────────────────────────────
#   Start (30d ago):    987 keys (8.9 MB)
#   Current:            1,247 keys (12.4 MB)
#   Change:             +260 keys (+3.5 MB)
#   Growth Rate:        +26.4% keys, +39.3% size
#   Daily Average:      +8.7 keys/day (+120 KB/day)
#
# GROWTH BY NAMESPACE
# ────────────────────────────────────────────────────────────
#   development/*:      +145 keys (+2.1 MB) - Fastest growing
#   testing/*:          +67 keys (+0.8 MB)
#   planning/*:         +34 keys (+0.4 MB)
#   research/*:         +14 keys (+0.2 MB)
#
# FORECAST (next 30 days)
# ────────────────────────────────────────────────────────────
#   Projected:          1,507 keys (15.9 MB)
#   Growth:             +260 keys (+3.5 MB)
#   Confidence:         High (based on linear trend)
#
# RECOMMENDATIONS
# ────────────────────────────────────────────────────────────
#   ⚠ development/* growing rapidly
#      Consider: periodic cleanup of temp/* and cache/*
#   ✓ Overall growth is healthy
#   ✓ No immediate action needed
```

### Example 4: Health Check

```bash
npx claude-flow@alpha memory stats --health-check

# Output:
# ════════════════════════════════════════════════════════════
#                    MEMORY HEALTH CHECK
# ════════════════════════════════════════════════════════════
#
# HEALTH SCORE: 82/100 (Good)
# ────────────────────────────────────────────────────────────
#
# ✓ PASSED CHECKS
# ────────────────────────────────────────────────────────────
#   ✓ Total size within limits (12.4 MB < 100 MB)
#   ✓ No memory leaks detected
#   ✓ Average key size reasonable (10.2 KB)
#   ✓ Fragmentation low (8%)
#   ✓ Growth rate sustainable (+8.7 keys/day)
#
# ⚠ WARNINGS
# ────────────────────────────────────────────────────────────
#   ⚠ 45 keys older than 90 days (0.5 MB)
#      Recommendation: Review and archive old data
#   ⚠ 4 keys larger than 100 KB (total: 1.1 MB)
#      Recommendation: Consider compression or splitting
#
# ❌ ISSUES
# ────────────────────────────────────────────────────────────
#   None detected
#
# OPTIMIZATION OPPORTUNITIES
# ────────────────────────────────────────────────────────────
#   1. Clear old temp data (estimated savings: 245 KB)
#      Command: npx claude-flow@alpha memory clear --namespace "*/temp/*" --age 7d
#
#   2. Run garbage collection (estimated savings: 180 KB)
#      Command: npx claude-flow@alpha memory gc
#
#   3. Compress large keys (estimated savings: 450 KB)
#      Command: npx claude-flow@alpha memory compress --threshold 50KB
#
#   Total potential savings: 875 KB (7.1%)
```

### Example 5: Top Keys Analysis

```bash
npx claude-flow@alpha memory stats --top-keys 20

# Output:
# ════════════════════════════════════════════════════════════
#                    TOP 20 LARGEST KEYS
# ════════════════════════════════════════════════════════════
#
# Rank  Size      Key                                      Age
# ────────────────────────────────────────────────────────────
#  1.   234 KB    development/coder/large-dataset          15d
#  2.   189 KB    testing/results/integration-suite        7d
#  3.   156 KB    research/findings/api-patterns           30d
#  4.   134 KB    development/coder/auth-implementation    3d
#  5.   128 KB    planning/architecture/system-design      45d
#  6.   112 KB    testing/results/e2e-suite                2d
#  7.   98 KB     development/tester/coverage-report       1d
#  8.   87 KB     research/benchmarks/performance          14d
#  9.   76 KB     production/config/feature-flags          60d
# 10.   65 KB     development/coder/api-contracts          5d
# ...
#
# Total (top 20): 2.8 MB (22.6% of total memory)
#
# RECOMMENDATIONS
# ────────────────────────────────────────────────────────────
#   • Consider archiving planning/architecture/system-design (45d old)
#   • Review production/config/feature-flags (60d old, rarely accessed)
#   • Compress research/findings/api-patterns (high compression potential)
```

### Example 6: Real-Time Monitoring

```bash
npx claude-flow@alpha memory stats --watch --interval 5s

# Output:
# ════════════════════════════════════════════════════════════
#              REAL-TIME MEMORY MONITORING
# ════════════════════════════════════════════════════════════
# Press Ctrl+C to stop
#
# [10:30:00] Keys: 1,247 | Size: 12.4 MB | Rate: +2 keys/min
# [10:30:05] Keys: 1,247 | Size: 12.4 MB | Rate: +0 keys/min
# [10:30:10] Keys: 1,248 | Size: 12.5 MB | Rate: +1 keys/min ↑
# [10:30:15] Keys: 1,248 | Size: 12.5 MB | Rate: +0 keys/min
# [10:30:20] Keys: 1,249 | Size: 12.6 MB | Rate: +1 keys/min ↑
#
# Latest Activity:
#   [10:30:20] Added: development/coder/task-456 (145 KB)
#   [10:30:10] Added: testing/tester/results-789 (98 KB)
```

## Implementation Details

### Statistics Collection

```typescript
interface MemoryStats {
  total_keys: number;
  total_size: number;
  avg_key_size: number;
  median_key_size: number;
  namespaces: NamespaceStats[];
  age_distribution: AgeDistribution;
  size_distribution: SizeDistribution;
  growth_trends?: GrowthTrends;
  health_score?: number;
  recommendations?: Recommendation[];
}

async function collectStats(options: StatsOptions): Promise<MemoryStats> {
  const keys = await listAllKeys(options.namespace);
  const entries = await retrieveAll(keys);

  return {
    total_keys: keys.length,
    total_size: calculateTotalSize(entries),
    avg_key_size: calculateAverage(entries),
    median_key_size: calculateMedian(entries),
    namespaces: groupByNamespace(entries),
    age_distribution: analyzeAge(entries),
    size_distribution: analyzeSize(entries),
    ...(options.trends && { growth_trends: analyzeTrends(entries, options.period) }),
    ...(options.healthCheck && { ...runHealthCheck(entries) })
  };
}
```

## Integration with Other Commands

### Chains With

**Before Cleanup**:
- Use `/memory-stats` to identify cleanup targets
- Then `/memory-clear` to remove stale data

**Monitoring**:
- Use `/memory-stats --watch` for continuous monitoring
- Export to `/metrics-export` for long-term tracking

**Optimization**:
- Analyze with `/memory-stats --health-check`
- Then `/memory-gc` to optimize

## See Also

- `/memory-gc` - Garbage collection
- `/memory-clear` - Clear memory
- `/metrics-export` - Export metrics
- `/agent-benchmark` - Agent performance metrics

---

**Version**: 2.0.0
**Last Updated**: 2025-11-01
**Maintained By**: ruv-SPARC Memory Team
