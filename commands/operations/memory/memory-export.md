---
name: memory-export
description: Export memory snapshots to JSON/YAML with filtering and compression
version: 2.0.0
category: memory
complexity: medium
tags: [memory, backup, export, snapshot, portability]
author: ruv-SPARC Memory Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [memory-retrieve, memory-stats]
chains_with: [memory-import, state-checkpoint, memory-clear]
evidence_based_techniques: [self-consistency, program-of-thought]
---

# /memory-export - Memory Snapshot Export

## Overview

The `/memory-export` command exports memory snapshots to portable formats (JSON, YAML) with filtering, compression, and encryption options. It enables memory backup, migration, sharing, and version control integration.

## Purpose

- **Backup & Recovery**: Create point-in-time snapshots for disaster recovery
- **Migration**: Transfer memory between environments or systems
- **Version Control**: Track memory changes over time
- **Sharing**: Export memory for collaboration or analysis
- **Archival**: Long-term storage with compression
- **Debugging**: Export memory state for troubleshooting

## Usage

```bash
# Export all memory to JSON
npx claude-flow@alpha memory export --output "memory-snapshot.json"

# Export specific namespace
npx claude-flow@alpha memory export \
  --namespace "development/*" \
  --output "dev-memory.json"

# Export with compression
npx claude-flow@alpha memory export \
  --output "memory.json.gz" \
  --compress gzip

# Export to YAML format
npx claude-flow@alpha memory export \
  --output "memory.yaml" \
  --format yaml

# Export with filtering by tags
npx claude-flow@alpha memory export \
  --tags "project:auth,status:complete" \
  --output "auth-complete.json"

# Export with metadata only (no values)
npx claude-flow@alpha memory export \
  --metadata-only \
  --output "memory-schema.json"

# Export to stdout (for piping)
npx claude-flow@alpha memory export \
  --namespace "planning/*" \
  --format json \
  --stdout | jq '.keys | length'

# Incremental export (only changes since last export)
npx claude-flow@alpha memory export \
  --since "2025-10-25" \
  --output "incremental.json"

# Export with encryption
npx claude-flow@alpha memory export \
  --output "secure-memory.json.enc" \
  --encrypt aes256 \
  --password-file ".secrets/export-key"
```

## Parameters

### Output Options

- `--output <path>` - Output file path
  - Default: `memory-export-{timestamp}.json`
  - Supports: `.json`, `.yaml`, `.yml`, `.gz`, `.enc`

- `--stdout` - Output to stdout instead of file
  - Useful for piping to other commands

- `--format <format>` - Export format
  - Options: `json` (default), `yaml`, `json-compact`, `json-pretty`

- `--compress <method>` - Compression method
  - Options: `gzip`, `bzip2`, `xz`, `none` (default)

- `--encrypt <method>` - Encryption method
  - Options: `aes256`, `aes128`, `none` (default)
  - Requires: `--password` or `--password-file`

### Filtering Options

- `--namespace <pattern>` - Export specific namespace(s)
  - Supports wildcards: `development/*`
  - Multiple: `--namespace "dev/*" --namespace "test/*"`

- `--tags <filter>` - Filter by metadata tags
  - Format: `key:value,key:value`
  - Example: `--tags "status:active,priority:high"`

- `--pattern <regex>` - Match keys by regex pattern
  - Example: `--pattern "task-[0-9]+-.*"`

- `--since <date>` - Export changes since date
  - Formats: `2025-10-25`, `7d`, `24h`
  - Enables incremental exports

- `--until <date>` - Export changes until date
  - Useful for historical snapshots

### Content Options

- `--metadata-only` - Export metadata without values
  - Useful for schema analysis

- `--include-deleted` - Include deleted entries
  - Requires audit log enabled

- `--no-metadata` - Export values only (no metadata)
  - Minimal export format

- `--pretty` - Pretty-print JSON output
  - Default for file output, disabled for stdout

### Additional Options

- `--validate` - Validate export before writing
  - Checks data integrity

- `--split <size>` - Split large exports into chunks
  - Format: `10MB`, `500KB`
  - Creates numbered files: `export-1.json`, `export-2.json`

- `--verbose` - Show detailed progress

- `--stats` - Show export statistics

## Examples

### Example 1: Full Memory Backup

```bash
# Create complete backup with compression
npx claude-flow@alpha memory export \
  --output "backups/memory-full-$(date +%Y%m%d).json.gz" \
  --compress gzip \
  --pretty \
  --stats

# Output:
# [Memory Export] Scanning all namespaces...
# [Memory Export] Found 1,247 keys across 15 namespaces
# [Memory Export] Total size: 12.4 MB
# [Memory Export] Exporting to: backups/memory-full-20251101.json.gz
# [Memory Export] Writing JSON...
# [Memory Export] Compressing with gzip...
# [Memory Export] Export complete!
#
# Statistics:
#   Keys exported: 1,247
#   Namespaces: 15
#   Original size: 12.4 MB
#   Compressed size: 2.1 MB (83% reduction)
#   Execution time: 1.2s
#   Output: backups/memory-full-20251101.json.gz
```

### Example 2: Incremental Export

```bash
# Export only changes since last backup
npx claude-flow@alpha memory export \
  --since "7d" \
  --output "backups/incremental-$(date +%Y%m%d).json" \
  --tags "status:modified" \
  --verbose

# Output:
# [Memory Export] Incremental export since: 2025-10-25
# [Memory Export] Scanning for changes...
# [Memory Export] Found 47 modified keys
# [Memory Export] Found 12 new keys
# [Memory Export] Found 3 deleted keys (--include-deleted not set, skipping)
# [Memory Export] Total: 59 keys
# [Memory Export] Exporting...
# [Memory Export] ✓ development/coder/task-123
# [Memory Export] ✓ testing/tester/results-456
# ...
# [Memory Export] Complete! Exported 59 keys (425 KB)
```

### Example 3: Namespace-Specific Export

```bash
# Export development namespace to YAML
npx claude-flow@alpha memory export \
  --namespace "development/*" \
  --format yaml \
  --output "exports/development.yaml" \
  --pretty

# Output file (development.yaml):
---
version: "2.0.0"
exported_at: "2025-11-01T10:30:45Z"
namespace: "development/*"
keys: 156
metadata:
  source: "memory-export"
  can_import: true
data:
  development/coder/task-123:
    value:
      status: "complete"
      files: ["auth.service.ts", "auth.controller.ts"]
      timestamp: "2025-11-01T09:15:00Z"
    metadata:
      agent: "coder"
      project: "auth-system"
      tags:
        status: "complete"
        priority: "high"
```

### Example 4: Metadata Schema Export

```bash
# Export metadata structure for analysis
npx claude-flow@alpha memory export \
  --metadata-only \
  --format json-pretty \
  --output "schema/memory-structure.json"

# Output:
{
  "version": "2.0.0",
  "exported_at": "2025-11-01T10:30:45Z",
  "type": "metadata-only",
  "keys": 1247,
  "schema": {
    "development/coder/task-123": {
      "namespace": "development/coder",
      "created_at": "2025-11-01T09:15:00Z",
      "updated_at": "2025-11-01T10:20:00Z",
      "size": 12456,
      "metadata": {
        "agent": "coder",
        "project": "auth-system",
        "tags": {
          "status": "complete",
          "priority": "high"
        }
      }
    }
  }
}
```

### Example 5: Encrypted Export

```bash
# Export with encryption for sensitive data
npx claude-flow@alpha memory export \
  --namespace "production/*" \
  --output "secure/production.json.enc" \
  --encrypt aes256 \
  --password-file ".secrets/export-key" \
  --compress gzip

# Output:
# [Memory Export] Scanning namespace: production/*
# [Memory Export] Found 234 keys
# [Memory Export] Loading encryption key...
# [Memory Export] Compressing...
# [Memory Export] Encrypting with AES-256...
# [Memory Export] Export complete!
# [Memory Export] File: secure/production.json.enc
# [Memory Export] Original: 2.4 MB → Compressed: 512 KB → Encrypted: 514 KB
```

### Example 6: Piping to External Tools

```bash
# Export to stdout and analyze with jq
npx claude-flow@alpha memory export \
  --namespace "testing/*" \
  --format json \
  --stdout | jq '.data | to_entries | map(select(.value.metadata.tags.status == "failed"))'

# Export and upload to S3
npx claude-flow@alpha memory export \
  --namespace "production/*" \
  --compress gzip \
  --stdout | aws s3 cp - s3://backups/memory-$(date +%Y%m%d).json.gz

# Export and split into chunks
npx claude-flow@alpha memory export \
  --all \
  --stdout | split -b 10M - exports/chunk-
```

## Export Format

### Standard JSON Format

```json
{
  "version": "2.0.0",
  "exported_at": "2025-11-01T10:30:45Z",
  "export_type": "full",
  "keys": 1247,
  "namespaces": [
    "development/*",
    "testing/*",
    "planning/*"
  ],
  "metadata": {
    "source": "memory-export",
    "can_import": true,
    "compression": "gzip",
    "encryption": "none"
  },
  "data": {
    "development/coder/task-123": {
      "value": {
        "status": "complete",
        "files": ["auth.service.ts"]
      },
      "metadata": {
        "agent": "coder",
        "created_at": "2025-11-01T09:15:00Z",
        "updated_at": "2025-11-01T10:20:00Z",
        "tags": {
          "project": "auth-system",
          "status": "complete"
        }
      }
    }
  }
}
```

### Compact JSON Format

```json
{"v":"2.0.0","t":"2025-11-01T10:30:45Z","k":1247,"d":{"development/coder/task-123":{"v":{"status":"complete"},"m":{"agent":"coder"}}}}
```

### YAML Format

```yaml
version: "2.0.0"
exported_at: "2025-11-01T10:30:45Z"
keys: 1247
data:
  development/coder/task-123:
    value:
      status: complete
    metadata:
      agent: coder
```

## Implementation Details

### Export Algorithm

```typescript
interface ExportOptions {
  output?: string;
  stdout?: boolean;
  format?: 'json' | 'yaml' | 'json-compact' | 'json-pretty';
  compress?: 'gzip' | 'bzip2' | 'xz' | 'none';
  encrypt?: 'aes256' | 'aes128' | 'none';
  namespace?: string[];
  tags?: Record<string, string>;
  since?: string;
  metadataOnly?: boolean;
  split?: string;
  validate?: boolean;
}

async function memoryExport(options: ExportOptions): Promise<ExportResult> {
  // 1. Build filters
  const filters = buildExportFilters(options);

  // 2. Scan and collect data
  const data = await collectMemoryData(filters);

  // 3. Format data
  const formatted = formatExportData(data, options.format);

  // 4. Validate if requested
  if (options.validate) {
    await validateExport(formatted);
  }

  // 5. Compress if requested
  let output = formatted;
  if (options.compress !== 'none') {
    output = await compressData(output, options.compress);
  }

  // 6. Encrypt if requested
  if (options.encrypt !== 'none') {
    output = await encryptData(output, options.encrypt);
  }

  // 7. Split if needed
  if (options.split) {
    return await splitAndWrite(output, options);
  }

  // 8. Write output
  if (options.stdout) {
    process.stdout.write(output);
  } else {
    await writeFile(options.output, output);
  }

  return {
    keys: data.length,
    size: output.length,
    path: options.output
  };
}
```

### Incremental Export

```typescript
async function collectIncrementalData(since: Date): Promise<MemoryEntry[]> {
  const allKeys = await listAllKeys();
  const incrementalKeys = [];

  for (const key of allKeys) {
    const metadata = await getMetadata(key);

    if (metadata.updated_at > since || metadata.created_at > since) {
      incrementalKeys.push(key);
    }
  }

  return retrieveAll(incrementalKeys);
}
```

## Integration with Other Commands

### Chains With

**Before Export**:
- `/memory-stats` - Analyze memory before export
- `/memory-clear` - Clean up before export
- `/memory-gc` - Optimize memory before export

**After Export**:
- `/memory-import` - Import exported data
- `/state-checkpoint` - Create checkpoint from export
- `/metrics-export` - Export metrics alongside memory

**Backup Workflows**:
```bash
# Complete backup workflow
npx claude-flow@alpha memory stats --detailed > stats.txt
npx claude-flow@alpha memory export --output "backup-$(date +%Y%m%d).json.gz" --compress gzip
npx claude-flow@alpha memory import --file "backup-$(date +%Y%m%d).json.gz" --validate
```

## Best Practices

### 1. Regular Backups
```bash
# Daily backup script
#!/bin/bash
npx claude-flow@alpha memory export \
  --output "backups/daily-$(date +%Y%m%d).json.gz" \
  --compress gzip \
  --stats
```

### 2. Use Incremental Exports
```bash
# Weekly incremental exports
npx claude-flow@alpha memory export \
  --since "7d" \
  --output "backups/incremental-$(date +%Y%m%d).json"
```

### 3. Compress Large Exports
```bash
# Always compress for large datasets
npx claude-flow@alpha memory export \
  --output "large-export.json.gz" \
  --compress gzip
```

### 4. Encrypt Sensitive Data
```bash
# Encrypt production exports
npx claude-flow@alpha memory export \
  --namespace "production/*" \
  --encrypt aes256 \
  --password-file ".secrets/key"
```

### 5. Validate Exports
```bash
# Always validate critical exports
npx claude-flow@alpha memory export \
  --output "critical.json" \
  --validate
```

## See Also

- `/memory-import` - Import memory snapshots
- `/memory-stats` - Memory usage statistics
- `/state-checkpoint` - Create state checkpoints
- `/memory-clear` - Clear memory by namespace
- `/memory-merge` - Merge memory from multiple sources

---

**Version**: 2.0.0
**Last Updated**: 2025-11-01
**Maintained By**: ruv-SPARC Memory Team
