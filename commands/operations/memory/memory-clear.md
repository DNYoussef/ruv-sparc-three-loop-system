---
name: memory-clear
description: Clear memory by namespace with selective deletion and safety checks
version: 2.0.0
category: memory
complexity: medium
tags: [memory, cleanup, state-management, garbage-collection]
author: ruv-SPARC Memory Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [memory-store, memory-retrieve, memory-stats]
chains_with: [memory-gc, memory-export, state-checkpoint]
evidence_based_techniques: [self-consistency, plan-and-solve]
---

# /memory-clear - Selective Memory Cleanup

## Overview

The `/memory-clear` command provides selective memory deletion with namespace filtering, safety checks, and backup options. It ensures controlled memory cleanup while preventing accidental data loss through confirmation prompts and automatic backups.

## Purpose

- **Selective Deletion**: Clear specific namespaces without affecting others
- **Safety First**: Automatic backups before deletion
- **Pattern Matching**: Support for wildcards and regex patterns
- **Audit Trail**: Log all deletions for recovery and analysis
- **Resource Recovery**: Free up memory and storage space
- **State Management**: Clean up stale or obsolete data

## Usage

```bash
# Clear specific namespace
npx claude-flow@alpha memory clear --namespace "development/coder/*"

# Clear with pattern matching
npx claude-flow@alpha memory clear --pattern "*/temp/*" --confirm

# Clear by age (older than 7 days)
npx claude-flow@alpha memory clear --age "7d" --namespace "research/*"

# Dry run (preview what would be deleted)
npx claude-flow@alpha memory clear --namespace "planning/*" --dry-run

# Clear with automatic backup
npx claude-flow@alpha memory clear --namespace "testing/*" --backup "backups/testing-backup.json"

# Clear all except specific namespaces
npx claude-flow@alpha memory clear --all --except "production/*,critical/*"

# Force clear without confirmation
npx claude-flow@alpha memory clear --namespace "temp/*" --force

# Clear by metadata tags
npx claude-flow@alpha memory clear --tags "status:stale,priority:low"
```

## Parameters

### Required

- `--namespace <pattern>` - Namespace pattern to clear (supports wildcards)
  - Examples: `development/*`, `*/temp/*`, `testing/unit/*`
  - Can use multiple patterns: `--namespace "dev/*" --namespace "test/*"`

### Optional

- `--pattern <regex>` - Regular expression pattern for matching keys
  - Example: `--pattern "task-[0-9]+-.*"`

- `--age <duration>` - Clear memories older than specified duration
  - Format: `7d` (days), `24h` (hours), `30m` (minutes)
  - Example: `--age "30d"` clears memories older than 30 days

- `--tags <tag-filter>` - Filter by metadata tags
  - Format: `key:value,key:value`
  - Example: `--tags "status:stale,type:temp"`

- `--dry-run` - Preview what would be deleted without actually deleting
  - Shows count and list of keys that match criteria

- `--backup <path>` - Create backup before deletion
  - Default: Auto-generates timestamped backup in `.backups/`
  - Example: `--backup "my-backup.json"`

- `--confirm` - Require explicit confirmation before deletion
  - Prompts user with deletion summary

- `--force` - Skip confirmation prompts (use with caution)
  - For automated scripts and CI/CD

- `--all` - Clear all memories (requires confirmation or --force)

- `--except <namespaces>` - Exclude specific namespaces when using --all
  - Format: Comma-separated list
  - Example: `--except "production/*,critical/*"`

- `--verbose` - Show detailed deletion progress

- `--stats` - Show statistics after deletion

## Examples

### Example 1: Clear Temporary Development Data

```bash
# Clear all temporary development artifacts
npx claude-flow@alpha memory clear \
  --namespace "development/*/temp/*" \
  --backup "backups/dev-temp-backup.json" \
  --verbose

# Output:
# [Memory Clear] Scanning namespace: development/*/temp/*
# [Memory Clear] Found 47 keys matching pattern
# [Memory Clear] Creating backup: backups/dev-temp-backup.json
# [Memory Clear] Backup created successfully (47 entries, 234 KB)
# [Memory Clear] Deleting 47 keys...
# [Memory Clear] ✓ Deleted development/coder/temp/cache-123
# [Memory Clear] ✓ Deleted development/tester/temp/results-456
# ...
# [Memory Clear] Successfully deleted 47 keys
# [Memory Clear] Memory freed: 234 KB
```

### Example 2: Clean Up Stale Test Results

```bash
# Remove test results older than 7 days
npx claude-flow@alpha memory clear \
  --namespace "testing/*" \
  --age "7d" \
  --tags "status:completed" \
  --stats

# Output:
# [Memory Clear] Filtering by age: older than 7 days
# [Memory Clear] Filtering by tags: status:completed
# [Memory Clear] Found 23 stale test results
# [Memory Clear] Total size: 145 KB
# [Memory Clear] Oldest entry: 2025-10-15 (17 days old)
# [Memory Clear] Newest entry: 2025-10-25 (7 days old)
#
# Delete 23 entries? (y/N): y
#
# [Memory Clear] Deleted 23 entries
# [Memory Clear] Memory freed: 145 KB
```

### Example 3: Dry Run Preview

```bash
# Preview what would be deleted without actually deleting
npx claude-flow@alpha memory clear \
  --namespace "planning/*" \
  --pattern "task-[0-9]+-draft" \
  --dry-run

# Output:
# [Memory Clear] DRY RUN MODE - No data will be deleted
# [Memory Clear] Matching keys:
#   1. planning/planner/task-123-draft (12 KB, age: 5d)
#   2. planning/planner/task-456-draft (8 KB, age: 3d)
#   3. planning/researcher/task-789-draft (15 KB, age: 1d)
#
# [Memory Clear] Would delete: 3 keys
# [Memory Clear] Total size: 35 KB
# [Memory Clear] Run without --dry-run to execute deletion
```

### Example 4: Selective Clear with Exclusions

```bash
# Clear all development namespaces except active projects
npx claude-flow@alpha memory clear \
  --namespace "development/*" \
  --except "development/*/active/*,development/*/production/*" \
  --backup "backups/dev-cleanup.json" \
  --confirm

# Output:
# [Memory Clear] Scanning: development/*
# [Memory Clear] Excluding: development/*/active/*, development/*/production/*
# [Memory Clear] Found 156 keys (42 excluded)
#
# Deletion Summary:
#   Total keys: 156
#   Total size: 1.2 MB
#   Namespaces affected: 8
#   Backup: backups/dev-cleanup.json
#
# Proceed with deletion? (y/N): y
#
# [Memory Clear] Creating backup...
# [Memory Clear] Deleting 156 keys...
# [Memory Clear] Complete! Memory freed: 1.2 MB
```

### Example 5: Force Clear for Automation

```bash
# Automated cleanup script (no prompts)
npx claude-flow@alpha memory clear \
  --namespace "temp/*" \
  --age "1d" \
  --force \
  --backup ".backups/auto-cleanup-$(date +%Y%m%d).json" \
  --stats

# Output:
# [Memory Clear] Auto-backup created: .backups/auto-cleanup-20251101.json
# [Memory Clear] Deleted 89 keys (425 KB)
#
# Statistics:
#   Keys deleted: 89
#   Memory freed: 425 KB
#   Oldest entry deleted: 2025-10-25 (7d)
#   Namespaces cleared: temp/cache, temp/sessions, temp/logs
#   Execution time: 0.34s
```

## Implementation Details

### Memory Clearing Algorithm

```typescript
interface ClearOptions {
  namespace?: string[];
  pattern?: string;
  age?: string;
  tags?: Record<string, string>;
  dryRun?: boolean;
  backup?: string;
  force?: boolean;
  all?: boolean;
  except?: string[];
  verbose?: boolean;
  stats?: boolean;
}

async function memoryClear(options: ClearOptions): Promise<ClearResult> {
  // 1. Build filter criteria
  const filters = buildFilters(options);

  // 2. Scan and match keys
  const matchedKeys = await scanMemory(filters);

  // 3. Apply exclusions
  const keysToDelete = applyExclusions(matchedKeys, options.except);

  // 4. Safety checks
  if (!options.force) {
    const confirmed = await confirmDeletion(keysToDelete);
    if (!confirmed) return { cancelled: true };
  }

  // 5. Create backup
  if (options.backup !== false) {
    await createBackup(keysToDelete, options.backup);
  }

  // 6. Execute deletion
  const result = await executeDeletes(keysToDelete, options.verbose);

  // 7. Generate statistics
  if (options.stats) {
    return generateStats(result);
  }

  return result;
}
```

### Filter Building

```typescript
function buildFilters(options: ClearOptions): MemoryFilter {
  const filters: MemoryFilter = {};

  // Namespace pattern matching
  if (options.namespace) {
    filters.namespace = options.namespace.map(ns =>
      convertWildcardToRegex(ns)
    );
  }

  // Age filtering
  if (options.age) {
    const threshold = parseAge(options.age);
    filters.timestamp = { before: threshold };
  }

  // Tag filtering
  if (options.tags) {
    filters.metadata = options.tags;
  }

  // Pattern matching
  if (options.pattern) {
    filters.pattern = new RegExp(options.pattern);
  }

  return filters;
}
```

### Backup Creation

```typescript
async function createBackup(keys: string[], backupPath?: string): Promise<string> {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const path = backupPath || `.backups/memory-clear-${timestamp}.json`;

  const data = await retrieveAll(keys);
  const backup = {
    timestamp,
    keys: keys.length,
    data,
    metadata: {
      version: '2.0.0',
      source: 'memory-clear',
      can_restore: true
    }
  };

  await fs.writeFile(path, JSON.stringify(backup, null, 2));
  return path;
}
```

## Safety Features

### 1. Automatic Backups
All deletions create automatic backups unless explicitly disabled:
```bash
# Automatic timestamped backup
npx claude-flow@alpha memory clear --namespace "dev/*"
# Creates: .backups/memory-clear-2025-11-01T10-30-45.json

# Custom backup location
npx claude-flow@alpha memory clear --namespace "dev/*" --backup "my-backup.json"

# Disable backup (not recommended)
npx claude-flow@alpha memory clear --namespace "temp/*" --no-backup --force
```

### 2. Confirmation Prompts
Interactive confirmation shows deletion summary:
```
Deletion Summary:
  Keys to delete: 47
  Total size: 234 KB
  Namespaces: development/coder/temp, development/tester/temp
  Oldest entry: 2025-10-15 (17 days old)
  Backup: .backups/memory-clear-2025-11-01.json

Proceed with deletion? (y/N):
```

### 3. Dry Run Mode
Preview deletions without executing:
```bash
npx claude-flow@alpha memory clear --namespace "planning/*" --dry-run
```

### 4. Exclusion Lists
Protect critical namespaces:
```bash
npx claude-flow@alpha memory clear --all \
  --except "production/*,critical/*,config/*"
```

## Integration with Other Commands

### Chains With

**Before Clearing**:
- `/memory-export` - Full backup before large-scale clearing
- `/memory-stats` - Analyze memory usage to identify cleanup targets
- `/state-checkpoint` - Create state checkpoint before clearing

**After Clearing**:
- `/memory-gc` - Run garbage collection to reclaim space
- `/memory-stats` - Verify memory freed
- `/metrics-export` - Export cleanup metrics

**Recovery**:
- `/memory-import` - Restore from backup if needed
- `/state-restore` - Restore full state from checkpoint

### Workflow Example

```bash
# 1. Analyze current memory usage
npx claude-flow@alpha memory stats --detailed

# 2. Preview cleanup
npx claude-flow@alpha memory clear --namespace "temp/*" --dry-run

# 3. Create checkpoint
npx claude-flow@alpha state checkpoint --name "before-cleanup"

# 4. Execute cleanup
npx claude-flow@alpha memory clear --namespace "temp/*" --backup "temp-backup.json"

# 5. Run garbage collection
npx claude-flow@alpha memory gc --aggressive

# 6. Verify results
npx claude-flow@alpha memory stats --compare before-cleanup
```

## Best Practices

### 1. Always Use Backups
```bash
# ✅ GOOD: Automatic backup
npx claude-flow@alpha memory clear --namespace "dev/*"

# ❌ BAD: No backup
npx claude-flow@alpha memory clear --namespace "dev/*" --no-backup --force
```

### 2. Use Dry Run First
```bash
# ✅ GOOD: Preview first
npx claude-flow@alpha memory clear --namespace "planning/*" --dry-run
# Then execute after verification

# ❌ BAD: Direct deletion without preview
npx claude-flow@alpha memory clear --namespace "planning/*" --force
```

### 3. Be Specific with Patterns
```bash
# ✅ GOOD: Specific pattern
npx claude-flow@alpha memory clear --namespace "testing/unit/temp/*"

# ❌ BAD: Too broad
npx claude-flow@alpha memory clear --namespace "testing/*" --force
```

### 4. Use Age-Based Cleanup
```bash
# ✅ GOOD: Clear old data
npx claude-flow@alpha memory clear --namespace "logs/*" --age "30d"

# ❌ BAD: Clear everything
npx claude-flow@alpha memory clear --namespace "logs/*" --force
```

### 5. Protect Production Data
```bash
# ✅ GOOD: Exclude production
npx claude-flow@alpha memory clear --all --except "production/*,critical/*"

# ❌ BAD: No exclusions
npx claude-flow@alpha memory clear --all --force
```

## Error Handling

```bash
# Permission errors
# Error: Insufficient permissions to delete namespace 'production/*'
# Solution: Use --except to exclude protected namespaces

# Backup failures
# Error: Cannot create backup at 'readonly/path'
# Solution: Specify writable backup location

# Pattern errors
# Error: Invalid regex pattern: '[unclosed'
# Solution: Use valid regex or wildcard patterns

# Age parsing errors
# Error: Invalid age format: '7 days'. Use format like '7d', '24h', '30m'
# Solution: Use correct duration format
```

## Performance Considerations

- **Large Deletions**: Use `--verbose` to monitor progress
- **Pattern Matching**: Specific patterns are faster than broad wildcards
- **Backup Size**: Large backups may take time to create
- **Concurrent Access**: Clears may be queued if memory is in use

## See Also

- `/memory-gc` - Garbage collection and defragmentation
- `/memory-export` - Export memory snapshots
- `/memory-import` - Import memory snapshots
- `/memory-stats` - Memory usage statistics
- `/state-checkpoint` - Create state checkpoints
- `/state-restore` - Restore from checkpoints

## Troubleshooting

### Issue: Deletion Too Slow
```bash
# Use more specific patterns
npx claude-flow@alpha memory clear --namespace "specific/path/*" --pattern "exact-match"

# Delete in batches
for ns in dev test staging; do
  npx claude-flow@alpha memory clear --namespace "$ns/*" --force
done
```

### Issue: Cannot Restore from Backup
```bash
# Verify backup integrity
npx claude-flow@alpha memory import --file "backup.json" --validate

# Restore specific keys
npx claude-flow@alpha memory import --file "backup.json" --namespace "specific/*"
```

### Issue: Accidentally Deleted Critical Data
```bash
# Restore from automatic backup
npx claude-flow@alpha memory import --file ".backups/memory-clear-2025-11-01.json"

# Or from state checkpoint
npx claude-flow@alpha state restore --checkpoint "before-cleanup"
```

---

**Version**: 2.0.0
**Last Updated**: 2025-11-01
**Maintained By**: ruv-SPARC Memory Team
