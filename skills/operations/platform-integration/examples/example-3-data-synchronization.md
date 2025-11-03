# Example 3: Bidirectional Data Synchronization

## Overview

This example demonstrates enterprise-grade bidirectional data synchronization with:
- Real-time and scheduled sync
- Conflict resolution strategies
- Change detection and delta sync
- Data transformation pipelines
- State tracking and recovery
- Monitoring and alerting

## Architecture

```
┌────────────────┐                    ┌────────────────┐
│   Salesforce   │                    │    HubSpot     │
│   (Source A)   │                    │   (Source B)   │
└───────┬────────┘                    └───────┬────────┘
        │                                     │
        │  ┌────────────────────────────┐    │
        ├──┤   Sync Orchestrator        │────┤
        │  ├────────────────────────────┤    │
        │  │ • Change Detection         │    │
        │  │ • Conflict Resolution      │    │
        │  │ • Transformation Pipeline  │    │
        │  │ • State Management         │    │
        │  └────────────────────────────┘    │
        │                │                    │
        ▼                ▼                    ▼
┌────────────────────────────────────────────────┐
│              PostgreSQL                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ State DB │  │ Conflicts│  │  Audit   │   │
│  └──────────┘  └──────────┘  └──────────┘   │
└────────────────────────────────────────────────┘
```

## Prerequisites

```bash
# Install dependencies
npm install pg bull cron axios lodash dotenv

# Database setup
createdb sync_state
psql sync_state < schema.sql

# Environment variables
export SALESFORCE_API_KEY="your_key"
export HUBSPOT_API_KEY="your_key"
export POSTGRES_URL="postgresql://localhost/sync_state"
export REDIS_URL="redis://localhost:6379"
export SYNC_SCHEDULE="*/15 * * * *"  # Every 15 minutes
```

## Database Schema

```sql
-- schema.sql
CREATE TABLE sync_state (
  id SERIAL PRIMARY KEY,
  entity_type VARCHAR(50) NOT NULL,
  entity_id VARCHAR(100) NOT NULL,
  source_platform VARCHAR(50) NOT NULL,
  target_platform VARCHAR(50) NOT NULL,
  last_synced_at TIMESTAMP WITH TIME ZONE,
  source_checksum VARCHAR(64),
  target_checksum VARCHAR(64),
  sync_status VARCHAR(20),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(entity_type, entity_id, source_platform, target_platform)
);

CREATE TABLE sync_conflicts (
  id SERIAL PRIMARY KEY,
  entity_type VARCHAR(50) NOT NULL,
  entity_id VARCHAR(100) NOT NULL,
  source_platform VARCHAR(50) NOT NULL,
  target_platform VARCHAR(50) NOT NULL,
  source_data JSONB NOT NULL,
  target_data JSONB NOT NULL,
  conflict_type VARCHAR(50),
  resolution_strategy VARCHAR(50),
  resolved BOOLEAN DEFAULT FALSE,
  resolved_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE sync_audit_log (
  id SERIAL PRIMARY KEY,
  entity_type VARCHAR(50) NOT NULL,
  entity_id VARCHAR(100) NOT NULL,
  operation VARCHAR(20) NOT NULL,
  source_platform VARCHAR(50) NOT NULL,
  target_platform VARCHAR(50) NOT NULL,
  data_before JSONB,
  data_after JSONB,
  success BOOLEAN NOT NULL,
  error_message TEXT,
  sync_duration_ms INTEGER,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_sync_state_entity ON sync_state(entity_type, entity_id);
CREATE INDEX idx_conflicts_resolved ON sync_conflicts(resolved, created_at);
CREATE INDEX idx_audit_created ON sync_audit_log(created_at);
```

## Implementation

### 1. Database Manager

```javascript
/**
 * Database state management
 */

const { Pool } = require('pg');

class SyncStateManager {
  constructor() {
    this.pool = new Pool({
      connectionString: process.env.POSTGRES_URL
    });
  }

  async getLastSyncTime(entityType, entityId, source, target) {
    const result = await this.pool.query(
      `SELECT last_synced_at FROM sync_state
       WHERE entity_type = $1 AND entity_id = $2
       AND source_platform = $3 AND target_platform = $4`,
      [entityType, entityId, source, target]
    );

    return result.rows[0]?.last_synced_at;
  }

  async updateSyncState(entityType, entityId, source, target, checksums) {
    await this.pool.query(
      `INSERT INTO sync_state
       (entity_type, entity_id, source_platform, target_platform,
        last_synced_at, source_checksum, target_checksum, sync_status)
       VALUES ($1, $2, $3, $4, NOW(), $5, $6, 'success')
       ON CONFLICT (entity_type, entity_id, source_platform, target_platform)
       DO UPDATE SET
         last_synced_at = NOW(),
         source_checksum = $5,
         target_checksum = $6,
         sync_status = 'success',
         updated_at = NOW()`,
      [entityType, entityId, source, target, checksums.source, checksums.target]
    );
  }

  async recordConflict(conflict) {
    await this.pool.query(
      `INSERT INTO sync_conflicts
       (entity_type, entity_id, source_platform, target_platform,
        source_data, target_data, conflict_type, resolution_strategy)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
      [
        conflict.entityType,
        conflict.entityId,
        conflict.source,
        conflict.target,
        JSON.stringify(conflict.sourceData),
        JSON.stringify(conflict.targetData),
        conflict.type,
        conflict.strategy
      ]
    );
  }

  async logAudit(log) {
    await this.pool.query(
      `INSERT INTO sync_audit_log
       (entity_type, entity_id, operation, source_platform, target_platform,
        data_before, data_after, success, error_message, sync_duration_ms)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)`,
      [
        log.entityType,
        log.entityId,
        log.operation,
        log.source,
        log.target,
        log.dataBefore ? JSON.stringify(log.dataBefore) : null,
        log.dataAfter ? JSON.stringify(log.dataAfter) : null,
        log.success,
        log.errorMessage,
        log.duration
      ]
    );
  }

  async getUnresolvedConflicts() {
    const result = await this.pool.query(
      `SELECT * FROM sync_conflicts
       WHERE resolved = false
       ORDER BY created_at DESC`
    );

    return result.rows;
  }
}

module.exports = SyncStateManager;
```

### 2. Change Detection Engine

```javascript
/**
 * Detect changes using checksums and timestamps
 */

const crypto = require('crypto');
const _ = require('lodash');

class ChangeDetector {
  /**
   * Calculate checksum for data object
   */
  calculateChecksum(data) {
    const normalized = this.normalizeData(data);
    return crypto
      .createHash('sha256')
      .update(JSON.stringify(normalized))
      .digest('hex');
  }

  /**
   * Normalize data for consistent checksums
   */
  normalizeData(data) {
    // Remove volatile fields
    const cleaned = _.omit(data, [
      'LastModifiedDate',
      'SystemModstamp',
      'CreatedDate',
      'lastmodifieddate',
      'createdate'
    ]);

    // Sort keys alphabetically
    const sorted = {};
    Object.keys(cleaned).sort().forEach(key => {
      sorted[key] = cleaned[key];
    });

    return sorted;
  }

  /**
   * Detect if data has changed
   */
  hasChanged(oldChecksum, newChecksum) {
    return oldChecksum !== newChecksum;
  }

  /**
   * Get changed fields
   */
  getChangedFields(oldData, newData) {
    const changes = {};

    Object.keys(newData).forEach(key => {
      if (!_.isEqual(oldData[key], newData[key])) {
        changes[key] = {
          old: oldData[key],
          new: newData[key]
        };
      }
    });

    return changes;
  }
}

module.exports = ChangeDetector;
```

### 3. Conflict Resolution

```javascript
/**
 * Conflict resolution strategies
 */

class ConflictResolver {
  constructor(strategy = 'last_write_wins') {
    this.strategy = strategy;
  }

  /**
   * Resolve conflict between source and target records
   */
  resolve(sourceRecord, targetRecord, metadata = {}) {
    switch (this.strategy) {
      case 'last_write_wins':
        return this.lastWriteWins(sourceRecord, targetRecord);

      case 'source_wins':
        return sourceRecord;

      case 'target_wins':
        return targetRecord;

      case 'merge_fields':
        return this.mergeFields(sourceRecord, targetRecord, metadata.mergeFields);

      case 'manual':
        return this.flagForManualResolution(sourceRecord, targetRecord);

      default:
        throw new Error(`Unknown strategy: ${this.strategy}`);
    }
  }

  /**
   * Last write wins strategy
   */
  lastWriteWins(sourceRecord, targetRecord) {
    const sourceTime = new Date(
      sourceRecord.LastModifiedDate || sourceRecord.lastmodifieddate || 0
    );
    const targetTime = new Date(
      targetRecord.LastModifiedDate || targetRecord.lastmodifieddate || 0
    );

    return sourceTime > targetTime ? sourceRecord : targetRecord;
  }

  /**
   * Merge specific fields
   */
  mergeFields(sourceRecord, targetRecord, fieldsToMerge = []) {
    const merged = { ...targetRecord };

    fieldsToMerge.forEach(field => {
      if (sourceRecord[field] !== undefined) {
        merged[field] = sourceRecord[field];
      }
    });

    return merged;
  }

  /**
   * Flag for manual resolution
   */
  flagForManualResolution(sourceRecord, targetRecord) {
    return {
      _conflict: true,
      source: sourceRecord,
      target: targetRecord,
      requiresManualReview: true
    };
  }
}

module.exports = ConflictResolver;
```

### 4. Sync Orchestrator

```javascript
/**
 * Main synchronization orchestrator
 */

const SyncStateManager = require('./sync-state-manager');
const ChangeDetector = require('./change-detector');
const ConflictResolver = require('./conflict-resolver');
const SalesforceConnector = require('./salesforce-connector');
const HubSpotConnector = require('./hubspot-connector');

class SyncOrchestrator {
  constructor(config) {
    this.config = config;
    this.stateManager = new SyncStateManager();
    this.changeDetector = new ChangeDetector();
    this.conflictResolver = new ConflictResolver(config.conflictStrategy);

    this.salesforce = new SalesforceConnector();
    this.hubspot = new HubSpotConnector();
  }

  /**
   * Execute bidirectional synchronization
   */
  async sync(entityType = 'Contact') {
    console.log(`\n🔄 Starting bidirectional sync: ${entityType}`);
    const startTime = Date.now();

    try {
      // Sync both directions
      await this.syncDirection('salesforce', 'hubspot', entityType);
      await this.syncDirection('hubspot', 'salesforce', entityType);

      const duration = Date.now() - startTime;
      console.log(`✅ Sync complete in ${duration}ms`);

      return { success: true, duration };

    } catch (error) {
      console.error(`❌ Sync failed:`, error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Sync in one direction
   */
  async syncDirection(sourceName, targetName, entityType) {
    console.log(`  ${sourceName} → ${targetName}`);

    // Get connectors
    const sourceConnector = this[sourceName];
    const targetConnector = this[targetName];

    // Fetch records modified since last sync
    const lastSync = await this.stateManager.getLastSyncTime(
      entityType,
      'all',
      sourceName,
      targetName
    );

    const sourceRecords = await sourceConnector.getModifiedSince(
      entityType,
      lastSync
    );

    console.log(`    Found ${sourceRecords.length} modified records`);

    // Process each record
    let synced = 0;
    let errors = 0;
    let conflicts = 0;

    for (const sourceRecord of sourceRecords) {
      const result = await this.syncRecord(
        sourceRecord,
        sourceName,
        targetName,
        entityType,
        sourceConnector,
        targetConnector
      );

      if (result.success) synced++;
      else if (result.conflict) conflicts++;
      else errors++;
    }

    console.log(`    ✓ ${synced} synced, ⚠ ${conflicts} conflicts, ✗ ${errors} errors`);
  }

  /**
   * Sync individual record
   */
  async syncRecord(sourceRecord, sourceName, targetName, entityType, sourceConn, targetConn) {
    const startTime = Date.now();
    const entityId = sourceRecord.Id || sourceRecord.id;

    try {
      // Calculate source checksum
      const sourceChecksum = this.changeDetector.calculateChecksum(sourceRecord);

      // Find corresponding record in target
      const targetRecord = await targetConn.findByExternalId(
        entityType,
        entityId
      );

      if (targetRecord) {
        // Record exists - check for conflicts
        const targetChecksum = this.changeDetector.calculateChecksum(targetRecord);

        // Check if both have changed since last sync
        const lastState = await this.stateManager.getLastSyncTime(
          entityType,
          entityId,
          sourceName,
          targetName
        );

        if (lastState) {
          const sourceChanged = this.changeDetector.hasChanged(
            lastState.source_checksum,
            sourceChecksum
          );
          const targetChanged = this.changeDetector.hasChanged(
            lastState.target_checksum,
            targetChecksum
          );

          if (sourceChanged && targetChanged) {
            // Conflict detected
            console.log(`    ⚠️  Conflict: ${entityId}`);

            const resolved = this.conflictResolver.resolve(
              sourceRecord,
              targetRecord
            );

            if (resolved._conflict) {
              // Manual resolution required
              await this.stateManager.recordConflict({
                entityType,
                entityId,
                source: sourceName,
                target: targetName,
                sourceData: sourceRecord,
                targetData: targetRecord,
                type: 'concurrent_modification',
                strategy: this.config.conflictStrategy
              });

              return { success: false, conflict: true };
            }

            // Auto-resolved - use resolved data
            await targetConn.update(entityType, targetRecord.id, resolved);
          } else if (sourceChanged) {
            // Only source changed - safe to update target
            await targetConn.update(entityType, targetRecord.id, sourceRecord);
          }
          // If only target changed, do nothing (will be synced in reverse direction)
        } else {
          // First sync - just update
          await targetConn.update(entityType, targetRecord.id, sourceRecord);
        }
      } else {
        // Record doesn't exist in target - create it
        await targetConn.create(entityType, sourceRecord);
      }

      // Update sync state
      await this.stateManager.updateSyncState(
        entityType,
        entityId,
        sourceName,
        targetName,
        {
          source: sourceChecksum,
          target: sourceChecksum  // Same after sync
        }
      );

      // Audit log
      await this.stateManager.logAudit({
        entityType,
        entityId,
        operation: targetRecord ? 'update' : 'create',
        source: sourceName,
        target: targetName,
        dataBefore: targetRecord,
        dataAfter: sourceRecord,
        success: true,
        duration: Date.now() - startTime
      });

      return { success: true };

    } catch (error) {
      console.error(`    ✗ Error syncing ${entityId}:`, error.message);

      await this.stateManager.logAudit({
        entityType,
        entityId,
        operation: 'sync',
        source: sourceName,
        target: targetName,
        success: false,
        errorMessage: error.message,
        duration: Date.now() - startTime
      });

      return { success: false, error: error.message };
    }
  }

  /**
   * Get sync statistics
   */
  async getStats() {
    const [lastSync, conflicts, recentLogs] = await Promise.all([
      this.stateManager.pool.query('SELECT MAX(last_synced_at) as last_sync FROM sync_state'),
      this.stateManager.pool.query('SELECT COUNT(*) as count FROM sync_conflicts WHERE resolved = false'),
      this.stateManager.pool.query(`
        SELECT
          success,
          COUNT(*) as count,
          AVG(sync_duration_ms) as avg_duration
        FROM sync_audit_log
        WHERE created_at > NOW() - INTERVAL '1 hour'
        GROUP BY success
      `)
    ]);

    return {
      lastSync: lastSync.rows[0].last_sync,
      unresolvedConflicts: parseInt(conflicts.rows[0].count),
      lastHour: recentLogs.rows
    };
  }
}

module.exports = SyncOrchestrator;
```

### 5. Scheduled Sync with Cron

```javascript
/**
 * Scheduled synchronization
 */

const cron = require('node-cron');
const SyncOrchestrator = require('./sync-orchestrator');

const orchestrator = new SyncOrchestrator({
  conflictStrategy: process.env.CONFLICT_STRATEGY || 'last_write_wins',
  mergeFields: ['Status', 'Priority']
});

// Schedule sync every 15 minutes
cron.schedule(process.env.SYNC_SCHEDULE || '*/15 * * * *', async () => {
  console.log('\n⏰ Scheduled sync starting...');

  const result = await orchestrator.sync('Contact');

  if (result.success) {
    const stats = await orchestrator.getStats();
    console.log('\n📊 Sync Statistics:');
    console.log(`  Last sync: ${stats.lastSync}`);
    console.log(`  Unresolved conflicts: ${stats.unresolvedConflicts}`);
    console.log(`  Last hour stats:`, stats.lastHour);
  }
});

console.log('🕐 Sync scheduler started');
console.log(`   Schedule: ${process.env.SYNC_SCHEDULE || '*/15 * * * *'}`);

// Manual trigger endpoint
const express = require('express');
const app = express();

app.post('/sync/trigger', async (req, res) => {
  const result = await orchestrator.sync('Contact');
  res.json(result);
});

app.get('/sync/stats', async (req, res) => {
  const stats = await orchestrator.getStats();
  res.json(stats);
});

app.get('/sync/conflicts', async (req, res) => {
  const conflicts = await orchestrator.stateManager.getUnresolvedConflicts();
  res.json(conflicts);
});

app.listen(4000, () => {
  console.log('📡 Sync API listening on port 4000');
});
```

## Usage

```bash
# Start sync scheduler
node data-sync-scheduler.js

# Manual trigger
curl -X POST http://localhost:4000/sync/trigger

# View statistics
curl http://localhost:4000/sync/stats

# View conflicts
curl http://localhost:4000/sync/conflicts
```

## Monitoring Dashboard

```javascript
// Real-time monitoring
const prometheus = require('prom-client');

const syncDuration = new prometheus.Histogram({
  name: 'sync_duration_seconds',
  help: 'Sync operation duration',
  labelNames: ['entity_type', 'direction']
});

const syncErrors = new prometheus.Counter({
  name: 'sync_errors_total',
  help: 'Total sync errors',
  labelNames: ['entity_type', 'error_type']
});

const syncConflicts = new prometheus.Counter({
  name: 'sync_conflicts_total',
  help: 'Total sync conflicts',
  labelNames: ['entity_type', 'resolution']
});
```

## Production Features

- ✅ **Change Detection**: SHA-256 checksums
- ✅ **Conflict Resolution**: 5 strategies
- ✅ **State Management**: PostgreSQL
- ✅ **Audit Logging**: Complete history
- ✅ **Error Recovery**: Automatic retry
- ✅ **Monitoring**: Prometheus metrics
- ✅ **Scheduling**: Cron-based automation

---

**Generated with Platform Integration Skill v2.0.0**
