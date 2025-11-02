---
name: memory-import
description: Import memory snapshots from JSON/YAML with conflict resolution
version: 2.0.0
category: memory
complexity: medium
tags: [memory, restore, import, migration, recovery]
author: ruv-SPARC Memory Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [memory-store, memory-export]
chains_with: [memory-merge, state-restore, memory-validate]
evidence_based_techniques: [self-consistency, plan-and-solve]
---

# /memory-import - Memory Snapshot Import

## Overview

The `/memory-import` command imports memory snapshots from JSON/YAML files with conflict resolution, validation, and selective restore capabilities. It supports restoring backups, migrating data, and merging memory from multiple sources.

## Purpose

- **Disaster Recovery**: Restore from backups after data loss
- **Migration**: Transfer memory between environments
- **Collaboration**: Import shared memory snapshots
- **Testing**: Load test data and fixtures
- **Rollback**: Revert to previous memory state
- **Merge**: Combine memory from multiple sources

## Usage

```bash
# Import from JSON file
npx claude-flow@alpha memory import --file "memory-backup.json"

# Import with conflict resolution
npx claude-flow@alpha memory import \
  --file "backup.json" \
  --on-conflict merge

# Import specific namespace only
npx claude-flow@alpha memory import \
  --file "full-backup.json" \
  --namespace "development/*"

# Import from compressed file
npx claude-flow@alpha memory import \
  --file "backup.json.gz" \
  --decompress gzip

# Import from encrypted file
npx claude-flow@alpha memory import \
  --file "secure.json.enc" \
  --decrypt aes256 \
  --password-file ".secrets/import-key"

# Dry run (validate without importing)
npx claude-flow@alpha memory import \
  --file "backup.json" \
  --dry-run

# Import with validation
npx claude-flow@alpha memory import \
  --file "backup.json" \
  --validate \
  --strict

# Import from stdin
cat backup.json | npx claude-flow@alpha memory import --stdin

# Import and merge multiple files
npx claude-flow@alpha memory import \
  --files "backup1.json,backup2.json,backup3.json" \
  --merge-strategy newest

# Selective import by tags
npx claude-flow@alpha memory import \
  --file "backup.json" \
  --tags "status:active,priority:high"
```

## Parameters

### Input Options

- `--file <path>` - Input file path
  - Supports: `.json`, `.yaml`, `.yml`, `.gz`, `.enc`

- `--files <paths>` - Multiple files (comma-separated)
  - Import and merge multiple snapshots

- `--stdin` - Read from stdin instead of file
  - Useful for piping data

- `--format <format>` - Input format (auto-detected if not specified)
  - Options: `json`, `yaml`, `auto` (default)

### Decompression/Decryption

- `--decompress <method>` - Decompression method
  - Options: `gzip`, `bzip2`, `xz`, `auto` (default)

- `--decrypt <method>` - Decryption method
  - Options: `aes256`, `aes128`, `none` (default)
  - Requires: `--password` or `--password-file`

- `--password <password>` - Decryption password
  - Not recommended (use --password-file instead)

- `--password-file <path>` - Decryption password file
  - More secure than --password

### Filtering Options

- `--namespace <pattern>` - Import specific namespace(s) only
  - Supports wildcards: `development/*`
  - Multiple: `--namespace "dev/*" --namespace "test/*"`

- `--tags <filter>` - Filter by metadata tags
  - Format: `key:value,key:value`
  - Example: `--tags "status:active,priority:high"`

- `--exclude-namespace <pattern>` - Exclude specific namespaces
  - Example: `--exclude-namespace "temp/*,cache/*"`

- `--keys <pattern>` - Import specific keys only
  - Supports regex: `--keys "task-[0-9]+-.*"`

### Conflict Resolution

- `--on-conflict <strategy>` - How to handle conflicts
  - Options:
    - `skip` - Skip conflicting keys (keep existing)
    - `overwrite` - Overwrite existing with imported
    - `merge` - Merge values (default)
    - `fail` - Abort on first conflict
    - `prompt` - Ask for each conflict

- `--merge-strategy <strategy>` - How to merge values
  - Options:
    - `newest` - Keep newest by timestamp (default)
    - `oldest` - Keep oldest by timestamp
    - `imported` - Prefer imported values
    - `existing` - Prefer existing values
    - `deep-merge` - Deep merge objects

### Validation Options

- `--validate` - Validate import before applying
  - Checks format, schema, and data integrity

- `--strict` - Strict validation (fail on warnings)
  - Use for production imports

- `--dry-run` - Validate without importing
  - Preview what would be imported

### Additional Options

- `--backup` - Create backup before importing
  - Auto-generates backup file

- `--backup-file <path>` - Custom backup path

- `--verbose` - Show detailed progress

- `--stats` - Show import statistics

## Examples

### Example 1: Simple Restore from Backup

```bash
# Restore from backup with automatic backup of current state
npx claude-flow@alpha memory import \
  --file "backups/memory-20251101.json" \
  --backup \
  --verbose

# Output:
# [Memory Import] Reading: backups/memory-20251101.json
# [Memory Import] Format: JSON
# [Memory Import] Keys in file: 1,247
# [Memory Import] Creating backup of current state...
# [Memory Import] Backup created: .backups/pre-import-20251101-103045.json
# [Memory Import] Validating import data...
# [Memory Import] ✓ Validation passed
# [Memory Import] Importing 1,247 keys...
# [Memory Import] ✓ development/coder/task-123
# [Memory Import] ✓ testing/tester/results-456
# ...
# [Memory Import] Import complete!
#
# Statistics:
#   Keys imported: 1,247
#   Keys skipped: 0
#   Keys merged: 0
#   Conflicts: 0
#   Execution time: 2.3s
```

### Example 2: Import with Conflict Resolution

```bash
# Import with merge strategy for conflicts
npx claude-flow@alpha memory import \
  --file "shared-memory.json" \
  --on-conflict merge \
  --merge-strategy newest \
  --stats

# Output:
# [Memory Import] Processing: shared-memory.json
# [Memory Import] Keys in file: 234
# [Memory Import] Checking for conflicts...
# [Memory Import] Found 47 conflicting keys
# [Memory Import] Applying merge strategy: newest
# [Memory Import]
# Conflict Resolution Summary:
#   Total conflicts: 47
#   Kept existing: 12 (older timestamps)
#   Kept imported: 35 (newer timestamps)
#   Deep merged: 0
#
# [Memory Import] Importing...
# [Memory Import] Complete!
#
# Statistics:
#   Keys imported: 187 (new)
#   Keys merged: 47 (conflicts)
#   Total: 234 keys processed
```

### Example 3: Selective Namespace Import

```bash
# Import only development namespace from full backup
npx claude-flow@alpha memory import \
  --file "full-backup.json" \
  --namespace "development/*" \
  --exclude-namespace "development/temp/*" \
  --verbose

# Output:
# [Memory Import] Filtering by namespace: development/*
# [Memory Import] Excluding: development/temp/*
# [Memory Import] Matched 156 keys (84 excluded)
# [Memory Import] Importing 156 keys...
# [Memory Import] ✓ development/coder/auth-api
# [Memory Import] ✓ development/coder/user-service
# [Memory Import] ✗ development/temp/cache (excluded)
# ...
# [Memory Import] Imported 156 keys
```

### Example 4: Encrypted Import

```bash
# Import from encrypted backup
npx claude-flow@alpha memory import \
  --file "secure/production.json.enc" \
  --decrypt aes256 \
  --password-file ".secrets/import-key" \
  --decompress gzip \
  --validate

# Output:
# [Memory Import] Reading encrypted file...
# [Memory Import] Loading decryption key from: .secrets/import-key
# [Memory Import] Decrypting with AES-256...
# [Memory Import] Decompressing with gzip...
# [Memory Import] Validating data integrity...
# [Memory Import] ✓ Validation passed
# [Memory Import] Importing 234 keys...
# [Memory Import] Complete!
```

### Example 5: Dry Run Validation

```bash
# Validate import without actually importing
npx claude-flow@alpha memory import \
  --file "untrusted-backup.json" \
  --dry-run \
  --strict \
  --verbose

# Output:
# [Memory Import] DRY RUN MODE - No data will be imported
# [Memory Import] Validating: untrusted-backup.json
# [Memory Import] Format: JSON
# [Memory Import] Version: 2.0.0
# [Memory Import] Keys: 1,247
# [Memory Import] Running strict validation...
# [Memory Import] ✓ Schema valid
# [Memory Import] ✓ All keys have valid namespaces
# [Memory Import] ✓ All timestamps valid
# [Memory Import] ✓ All metadata valid
# [Memory Import] ✓ No duplicate keys
# [Memory Import] ⚠ Warning: 3 keys have future timestamps
# [Memory Import] ✗ FAILED: Strict mode requires no warnings
#
# Validation failed. Fix warnings and re-run.
# Use --validate (without --strict) to import with warnings.
```

### Example 6: Merge Multiple Sources

```bash
# Import and merge from multiple backup files
npx claude-flow@alpha memory import \
  --files "backup1.json,backup2.json,backup3.json" \
  --merge-strategy newest \
  --on-conflict merge \
  --stats

# Output:
# [Memory Import] Merging 3 files...
# [Memory Import] Processing backup1.json (412 keys)
# [Memory Import] Processing backup2.json (356 keys)
# [Memory Import] Processing backup3.json (289 keys)
# [Memory Import] Total unique keys: 867
# [Memory Import] Conflicts across files: 190
# [Memory Import] Applying merge strategy: newest
# [Memory Import] Importing merged data...
# [Memory Import] Complete!
#
# Statistics:
#   Files merged: 3
#   Unique keys: 867
#   Cross-file conflicts: 190
#   Final keys imported: 867
```

## Import Validation

### Schema Validation

```typescript
interface ImportSchema {
  version: string;
  exported_at: string;
  keys: number;
  data: Record<string, MemoryEntry>;
  metadata?: {
    source: string;
    can_import: boolean;
    compression?: string;
    encryption?: string;
  };
}

function validateImport(data: unknown): ValidationResult {
  // Check schema
  if (!isValidSchema(data)) {
    return { valid: false, error: 'Invalid schema' };
  }

  // Check version compatibility
  if (!isCompatibleVersion(data.version)) {
    return { valid: false, error: 'Incompatible version' };
  }

  // Validate each entry
  for (const [key, entry] of Object.entries(data.data)) {
    if (!isValidEntry(key, entry)) {
      return { valid: false, error: `Invalid entry: ${key}` };
    }
  }

  return { valid: true };
}
```

### Conflict Detection

```typescript
async function detectConflicts(
  importData: ImportData,
  existingKeys: string[]
): Promise<Conflict[]> {
  const conflicts: Conflict[] = [];

  for (const key of Object.keys(importData.data)) {
    if (existingKeys.includes(key)) {
      const existing = await retrieve(key);
      const imported = importData.data[key];

      conflicts.push({
        key,
        existing: existing.value,
        imported: imported.value,
        existingTimestamp: existing.metadata.updated_at,
        importedTimestamp: imported.metadata.updated_at
      });
    }
  }

  return conflicts;
}
```

### Merge Strategies

```typescript
function mergeValues(
  existing: any,
  imported: any,
  strategy: MergeStrategy
): any {
  switch (strategy) {
    case 'newest':
      return existing.timestamp > imported.timestamp ? existing : imported;

    case 'oldest':
      return existing.timestamp < imported.timestamp ? existing : imported;

    case 'deep-merge':
      return deepMerge(existing.value, imported.value);

    case 'imported':
      return imported;

    case 'existing':
      return existing;

    default:
      throw new Error(`Unknown merge strategy: ${strategy}`);
  }
}
```

## Implementation Details

### Import Algorithm

```typescript
interface ImportOptions {
  file?: string;
  files?: string[];
  stdin?: boolean;
  format?: 'json' | 'yaml' | 'auto';
  decompress?: 'gzip' | 'bzip2' | 'xz' | 'auto';
  decrypt?: 'aes256' | 'aes128' | 'none';
  namespace?: string[];
  onConflict?: 'skip' | 'overwrite' | 'merge' | 'fail' | 'prompt';
  mergeStrategy?: 'newest' | 'oldest' | 'imported' | 'existing' | 'deep-merge';
  validate?: boolean;
  strict?: boolean;
  dryRun?: boolean;
  backup?: boolean;
}

async function memoryImport(options: ImportOptions): Promise<ImportResult> {
  // 1. Read and decrypt/decompress
  let data = await readImportData(options);

  // 2. Validate format
  if (options.validate) {
    const validation = await validateImport(data, options.strict);
    if (!validation.valid) {
      throw new Error(`Validation failed: ${validation.error}`);
    }
  }

  // 3. Apply filters
  data = applyFilters(data, options);

  // 4. Detect conflicts
  const conflicts = await detectConflicts(data, await listAllKeys());

  // 5. Resolve conflicts
  const resolved = await resolveConflicts(conflicts, options);

  // 6. Create backup if requested
  if (options.backup) {
    await createBackup();
  }

  // 7. Import data
  if (!options.dryRun) {
    return await executeImport(resolved);
  }

  return { dryRun: true, wouldImport: resolved.length };
}
```

## Integration with Other Commands

### Chains With

**Before Import**:
- `/memory-export` - Export current state as backup
- `/state-checkpoint` - Create checkpoint before import
- `/memory-stats` - Analyze current memory state

**After Import**:
- `/memory-validate` - Validate imported data
- `/memory-merge` - Merge with other sources
- `/memory-stats` - Verify import results

**Recovery Workflows**:
```bash
# Safe import workflow
npx claude-flow@alpha state checkpoint --name "before-import"
npx claude-flow@alpha memory import --file "backup.json" --validate --backup
npx claude-flow@alpha memory stats --compare before-import

# Rollback if needed
npx claude-flow@alpha state restore --checkpoint "before-import"
```

## Best Practices

### 1. Always Validate
```bash
# ✅ GOOD: Validate before importing
npx claude-flow@alpha memory import --file "backup.json" --validate

# ❌ BAD: Import without validation
npx claude-flow@alpha memory import --file "backup.json"
```

### 2. Create Backups
```bash
# ✅ GOOD: Backup current state
npx claude-flow@alpha memory import --file "new.json" --backup

# ❌ BAD: No backup
npx claude-flow@alpha memory import --file "new.json"
```

### 3. Use Dry Run First
```bash
# ✅ GOOD: Test first
npx claude-flow@alpha memory import --file "backup.json" --dry-run
npx claude-flow@alpha memory import --file "backup.json"

# ❌ BAD: Direct import
npx claude-flow@alpha memory import --file "backup.json"
```

### 4. Handle Conflicts Explicitly
```bash
# ✅ GOOD: Explicit conflict resolution
npx claude-flow@alpha memory import --file "backup.json" --on-conflict merge --merge-strategy newest

# ❌ BAD: Default conflict handling
npx claude-flow@alpha memory import --file "backup.json"
```

## See Also

- `/memory-export` - Export memory snapshots
- `/memory-merge` - Merge memory from multiple sources
- `/state-checkpoint` - Create state checkpoints
- `/state-restore` - Restore from checkpoints
- `/memory-validate` - Validate memory data

---

**Version**: 2.0.0
**Last Updated**: 2025-11-01
**Maintained By**: ruv-SPARC Memory Team
