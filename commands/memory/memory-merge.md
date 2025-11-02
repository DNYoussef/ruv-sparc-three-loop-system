---
name: memory-merge
description: Merge memory from multiple sources with conflict resolution and deduplication
version: 2.0.0
category: memory
complexity: high
tags: [memory, merge, consolidation, deduplication, integration]
author: ruv-SPARC Memory Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [memory-import, memory-export, memory-store]
chains_with: [memory-import, memory-validate, state-merge]
evidence_based_techniques: [self-consistency, program-of-thought, plan-and-solve]
---

# /memory-merge - Multi-Source Memory Consolidation

## Overview

The `/memory-merge` command consolidates memory from multiple sources with intelligent conflict resolution, deduplication, and validation. It enables merging distributed memory, consolidating backups, and integrating data from multiple agents or environments.

## Purpose

- **Multi-Source Integration**: Combine memory from different sources
- **Distributed Coordination**: Merge memory from distributed agents
- **Backup Consolidation**: Merge multiple backup files
- **Cross-Environment Sync**: Synchronize memory across environments
- **Deduplication**: Eliminate duplicate entries
- **Conflict Resolution**: Intelligent merging of conflicting values

## Usage

```bash
# Merge two memory sources
npx claude-flow@alpha memory merge \
  --sources "backup1.json,backup2.json" \
  --output "merged.json"

# Merge with conflict resolution strategy
npx claude-flow@alpha memory merge \
  --sources "dev.json,staging.json,prod.json" \
  --strategy newest \
  --output "consolidated.json"

# Merge and deduplicate
npx claude-flow@alpha memory merge \
  --sources "source1.json,source2.json" \
  --deduplicate \
  --output "deduplicated.json"

# Merge directly into memory (no file output)
npx claude-flow@alpha memory merge \
  --sources "backup1.json,backup2.json" \
  --apply

# Merge with priority order
npx claude-flow@alpha memory merge \
  --sources "prod.json,staging.json,dev.json" \
  --priority-order \
  --output "merged.json"

# Three-way merge with base
npx claude-flow@alpha memory merge \
  --base "original.json" \
  --sources "branch1.json,branch2.json" \
  --strategy three-way \
  --output "merged.json"

# Merge with custom conflict resolver
npx claude-flow@alpha memory merge \
  --sources "source1.json,source2.json" \
  --resolver "custom-resolver.js" \
  --output "merged.json"

# Dry run to preview merge
npx claude-flow@alpha memory merge \
  --sources "backup1.json,backup2.json" \
  --dry-run
```

## Parameters

### Input Options

- `--sources <paths>` - Source files (comma-separated)
  - Example: `--sources "file1.json,file2.json,file3.json"`
  - Minimum 2 sources required

- `--base <path>` - Base file for three-way merge
  - Used with `--strategy three-way`

- `--current-memory` - Include current memory as source
  - Merges with existing in-memory data

### Merge Strategies

- `--strategy <strategy>` - Merge strategy
  - Options:
    - `newest` - Keep newest by timestamp (default)
    - `oldest` - Keep oldest by timestamp
    - `priority-order` - Use source order as priority
    - `union` - Keep all unique values
    - `intersection` - Keep only common values
    - `three-way` - Three-way merge with base
    - `custom` - Use custom resolver

- `--priority-order` - Use source order as priority
  - First source has highest priority

### Conflict Resolution

- `--on-conflict <action>` - How to handle conflicts
  - Options: `auto`, `merge`, `prompt`, `fail`
  - Default: `auto` (uses strategy)

- `--resolver <path>` - Custom conflict resolver script
  - JavaScript file exporting resolver function

- `--merge-depth <number>` - Deep merge depth for objects
  - Default: Infinite
  - Example: `--merge-depth 3`

### Deduplication

- `--deduplicate` - Remove duplicate entries
  - Based on key and value hash

- `--dedupe-strategy <strategy>` - Deduplication strategy
  - Options: `exact`, `semantic`, `fuzzy`
  - Default: `exact`

- `--similarity-threshold <percent>` - Fuzzy match threshold
  - Range: 0-100
  - Default: 90

### Output Options

- `--output <path>` - Output merged file
  - Format auto-detected from extension

- `--apply` - Apply merge directly to memory
  - No file output

- `--format <format>` - Output format
  - Options: `json`, `yaml`

- `--backup` - Backup current memory before applying

### Validation Options

- `--validate` - Validate merge result
  - Check integrity and consistency

- `--dry-run` - Preview merge without applying
  - Shows conflicts and resolution plan

- `--verbose` - Detailed merge progress

- `--stats` - Show merge statistics

## Examples

### Example 1: Basic Two-Source Merge

```bash
# Merge development and staging memory
npx claude-flow@alpha memory merge \
  --sources "dev-memory.json,staging-memory.json" \
  --strategy newest \
  --output "merged-memory.json" \
  --stats

# Output:
# [Memory Merge] Loading sources...
# [Memory Merge] ✓ dev-memory.json (234 keys)
# [Memory Merge] ✓ staging-memory.json (189 keys)
# [Memory Merge] Total unique keys: 367
# [Memory Merge] Conflicts detected: 56
# [Memory Merge] Applying strategy: newest
# [Memory Merge] Resolving conflicts...
# [Memory Merge] Writing merged data to: merged-memory.json
# [Memory Merge] Complete!
#
# Statistics:
#   Sources: 2
#   Total keys: 423 (367 unique)
#   Duplicates: 56
#   Conflicts resolved: 56 (newest timestamp)
#   Output size: 1.2 MB
```

### Example 2: Priority-Based Merge

```bash
# Merge with priority order (production > staging > development)
npx claude-flow@alpha memory merge \
  --sources "prod.json,staging.json,dev.json" \
  --priority-order \
  --output "consolidated.json" \
  --verbose

# Output:
# [Memory Merge] Priority order: prod.json > staging.json > dev.json
# [Memory Merge] Loading sources...
# [Memory Merge] Source 1 (priority 1): prod.json (156 keys)
# [Memory Merge] Source 2 (priority 2): staging.json (234 keys)
# [Memory Merge] Source 3 (priority 3): dev.json (312 keys)
# [Memory Merge] Merging...
# [Memory Merge] ✓ Key: config/database/host
#   - Conflict: prod.json (prod-db.example.com) vs staging.json (staging-db.example.com)
#   - Resolution: Using prod.json (priority 1)
# [Memory Merge] ✓ Key: config/feature-flags/new-ui
#   - No conflict
# ...
# [Memory Merge] Merged 702 keys (387 unique)
```

### Example 3: Three-Way Merge

```bash
# Three-way merge with common base
npx claude-flow@alpha memory merge \
  --base "original.json" \
  --sources "branch-a.json,branch-b.json" \
  --strategy three-way \
  --output "merged.json"

# Output:
# [Memory Merge] Three-way merge mode
# [Memory Merge] Base: original.json (100 keys)
# [Memory Merge] Branch A: branch-a.json (15 changes)
# [Memory Merge] Branch B: branch-b.json (23 changes)
# [Memory Merge] Analyzing changes...
# [Memory Merge] A-only changes: 8
# [Memory Merge] B-only changes: 15
# [Memory Merge] Conflicts: 7
# [Memory Merge] Resolving conflicts...
# [Memory Merge] ✓ Auto-resolved: 4
# [Memory Merge] ⚠ Manual resolution needed: 3
#
# Conflicts requiring manual resolution:
#   1. config/api/timeout (A: 5000ms, B: 10000ms)
#   2. config/cache/ttl (A: 3600s, B: 7200s)
#   3. feature/new-auth (A: enabled, B: disabled)
#
# Use --on-conflict prompt to resolve interactively
```

### Example 4: Deduplication with Fuzzy Matching

```bash
# Merge with semantic deduplication
npx claude-flow@alpha memory merge \
  --sources "backup1.json,backup2.json,backup3.json" \
  --deduplicate \
  --dedupe-strategy fuzzy \
  --similarity-threshold 95 \
  --output "deduplicated.json" \
  --stats

# Output:
# [Memory Merge] Loading 3 sources...
# [Memory Merge] Total entries: 1,247
# [Memory Merge] Running fuzzy deduplication (threshold: 95%)...
# [Memory Merge] Found 234 potential duplicates
# [Memory Merge] Analyzing similarity...
# [Memory Merge] ✓ Exact duplicates: 89
# [Memory Merge] ✓ Fuzzy matches: 145 (95%+ similar)
# [Memory Merge] Deduplicated: 234 entries removed
# [Memory Merge] Final count: 1,013 unique entries
#
# Statistics:
#   Original entries: 1,247
#   Exact duplicates: 89
#   Fuzzy duplicates: 145
#   Unique entries: 1,013
#   Space saved: 18.8%
```

### Example 5: Custom Conflict Resolver

```bash
# Use custom resolver for domain-specific logic
npx claude-flow@alpha memory merge \
  --sources "team-a.json,team-b.json" \
  --resolver "resolvers/team-merge.js" \
  --output "team-merged.json"

# Custom resolver (resolvers/team-merge.js):
module.exports = function(key, values, metadata) {
  // Custom logic: prefer completed tasks
  if (key.includes('/task-')) {
    const completed = values.find(v => v.status === 'completed');
    if (completed) return completed;
  }

  // Default: newest timestamp
  return values.sort((a, b) =>
    b.metadata.updated_at - a.metadata.updated_at
  )[0];
};

# Output:
# [Memory Merge] Using custom resolver: resolvers/team-merge.js
# [Memory Merge] Merging 2 sources...
# [Memory Merge] Applying custom resolution logic...
# [Memory Merge] ✓ Resolved 47 conflicts using custom logic
# [Memory Merge] Complete!
```

### Example 6: Apply Merge Directly to Memory

```bash
# Merge and apply directly without file output
npx claude-flow@alpha memory merge \
  --sources "update1.json,update2.json" \
  --current-memory \
  --apply \
  --backup \
  --verbose

# Output:
# [Memory Merge] Including current memory as source
# [Memory Merge] Creating backup before merge...
# [Memory Merge] Backup: .backups/pre-merge-20251101-103045.json
# [Memory Merge] Loading sources...
# [Memory Merge] Source 1: update1.json (45 keys)
# [Memory Merge] Source 2: update2.json (67 keys)
# [Memory Merge] Source 3: current memory (1,247 keys)
# [Memory Merge] Merging...
# [Memory Merge] Applying to memory...
# [Memory Merge] ✓ Updated 112 keys
# [Memory Merge] ✓ Added 0 new keys
# [Memory Merge] Complete!
```

## Merge Strategies Explained

### 1. Newest Strategy
Keeps the value with the newest timestamp:
```typescript
function newestStrategy(conflicts: Conflict[]): Resolution[] {
  return conflicts.map(c => ({
    key: c.key,
    value: c.values.sort((a, b) =>
      b.metadata.updated_at - a.metadata.updated_at
    )[0]
  }));
}
```

### 2. Priority Order Strategy
Uses source order as priority:
```typescript
function priorityOrderStrategy(
  conflicts: Conflict[],
  sourceOrder: string[]
): Resolution[] {
  return conflicts.map(c => {
    const sourceIndex = c.values.map(v =>
      sourceOrder.indexOf(v.source)
    );
    const highestPriority = Math.min(...sourceIndex);
    return {
      key: c.key,
      value: c.values[sourceIndex.indexOf(highestPriority)]
    };
  });
}
```

### 3. Three-Way Merge Strategy
Merges based on changes from common base:
```typescript
function threeWayMerge(
  base: MemoryEntry,
  branchA: MemoryEntry,
  branchB: MemoryEntry
): Resolution {
  const aChanged = !deepEqual(base, branchA);
  const bChanged = !deepEqual(base, branchB);

  if (!aChanged && !bChanged) return base;
  if (aChanged && !bChanged) return branchA;
  if (!aChanged && bChanged) return branchB;

  // Both changed - conflict
  if (deepEqual(branchA, branchB)) return branchA;

  // Manual resolution needed
  throw new MergeConflict('Both branches modified', {
    base,
    branchA,
    branchB
  });
}
```

### 4. Union Strategy
Combines all unique values:
```typescript
function unionStrategy(conflicts: Conflict[]): Resolution[] {
  return conflicts.map(c => ({
    key: c.key,
    value: {
      merged: true,
      sources: c.values.map(v => ({
        source: v.source,
        value: v.value
      }))
    }
  }));
}
```

## Implementation Details

### Merge Algorithm

```typescript
interface MergeOptions {
  sources: string[];
  base?: string;
  strategy?: MergeStrategy;
  onConflict?: 'auto' | 'merge' | 'prompt' | 'fail';
  deduplicate?: boolean;
  output?: string;
  apply?: boolean;
  validate?: boolean;
}

async function memoryMerge(options: MergeOptions): Promise<MergeResult> {
  // 1. Load all sources
  const sources = await loadSources(options.sources);

  // 2. Load base if three-way merge
  const base = options.base ? await loadBase(options.base) : null;

  // 3. Identify conflicts
  const conflicts = identifyConflicts(sources, base);

  // 4. Apply merge strategy
  const resolved = await applyStrategy(conflicts, options.strategy);

  // 5. Deduplicate if requested
  if (options.deduplicate) {
    resolved = await deduplicate(resolved, options);
  }

  // 6. Validate result
  if (options.validate) {
    await validateMerge(resolved);
  }

  // 7. Output or apply
  if (options.apply) {
    return await applyToMemory(resolved, options);
  } else {
    return await writeOutput(resolved, options.output);
  }
}
```

### Conflict Detection

```typescript
function identifyConflicts(sources: MemoryData[]): Conflict[] {
  const keyMap = new Map<string, MemoryEntry[]>();

  // Group entries by key
  for (const source of sources) {
    for (const [key, entry] of Object.entries(source.data)) {
      if (!keyMap.has(key)) {
        keyMap.set(key, []);
      }
      keyMap.get(key)!.push({ ...entry, source: source.name });
    }
  }

  // Find conflicts (keys with multiple different values)
  const conflicts: Conflict[] = [];
  for (const [key, entries] of keyMap) {
    if (entries.length > 1) {
      const uniqueValues = deduplicateByValue(entries);
      if (uniqueValues.length > 1) {
        conflicts.push({ key, values: uniqueValues });
      }
    }
  }

  return conflicts;
}
```

## Integration with Other Commands

### Chains With

**Before Merge**:
- `/memory-export` - Export sources for merging
- `/memory-stats` - Analyze source data
- `/state-checkpoint` - Create checkpoint before merge

**After Merge**:
- `/memory-validate` - Validate merged result
- `/memory-import` - Import merged data
- `/memory-stats` - Verify merge results

**Workflow**:
```bash
# Multi-source consolidation workflow
npx claude-flow@alpha memory export --namespace "dev/*" --output "dev.json"
npx claude-flow@alpha memory export --namespace "staging/*" --output "staging.json"
npx claude-flow@alpha memory merge --sources "dev.json,staging.json" --output "merged.json"
npx claude-flow@alpha memory import --file "merged.json" --validate
```

## Best Practices

### 1. Always Backup Before Applying
```bash
# ✅ GOOD: Create backup
npx claude-flow@alpha memory merge --sources "s1.json,s2.json" --apply --backup

# ❌ BAD: No backup
npx claude-flow@alpha memory merge --sources "s1.json,s2.json" --apply
```

### 2. Use Dry Run First
```bash
# ✅ GOOD: Preview merge
npx claude-flow@alpha memory merge --sources "s1.json,s2.json" --dry-run
npx claude-flow@alpha memory merge --sources "s1.json,s2.json" --apply

# ❌ BAD: Direct merge
npx claude-flow@alpha memory merge --sources "s1.json,s2.json" --apply
```

### 3. Choose Appropriate Strategy
```bash
# ✅ GOOD: Explicit strategy
npx claude-flow@alpha memory merge --sources "prod.json,staging.json" --strategy newest

# ❌ BAD: Default strategy without consideration
npx claude-flow@alpha memory merge --sources "prod.json,staging.json"
```

## See Also

- `/memory-import` - Import memory snapshots
- `/memory-export` - Export memory snapshots
- `/memory-validate` - Validate memory data
- `/state-merge` - Merge full state
- `/memory-clear` - Clear memory by namespace

---

**Version**: 2.0.0
**Last Updated**: 2025-11-01
**Maintained By**: ruv-SPARC Memory Team
