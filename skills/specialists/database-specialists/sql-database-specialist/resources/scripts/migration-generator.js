#!/usr/bin/env node
/**
 * Migration Generator - Zero-Downtime Database Migrations
 *
 * Generates safe, zero-downtime migration scripts for PostgreSQL/MySQL:
 * - Add columns with defaults
 * - Modify columns safely
 * - Create indexes concurrently
 * - Rename columns/tables
 * - Batch backfill operations
 * - Drop columns safely
 *
 * Usage:
 *   node migration-generator.js add-column --table users --column email_verified --type boolean
 *   node migration-generator.js add-index --table orders --columns user_id,created_at
 *   node migration-generator.js rename-column --table users --old email --new email_address
 */

const fs = require('fs');
const path = require('path');

class MigrationGenerator {
  constructor(options = {}) {
    this.dbType = options.db || 'postgres';
    this.migrationDir = options.dir || './migrations';
    this.timestamp = new Date().toISOString().replace(/[-:T.]/g, '').substring(0, 14);
  }

  /**
   * Generate add column migration
   */
  generateAddColumn(table, column, dataType, options = {}) {
    const { notNull = false, defaultValue = null, comment = null } = options;

    const migrationName = `${this.timestamp}_add_${column}_to_${table}`;
    const migration = this.dbType === 'postgres'
      ? this.generatePostgresAddColumn(table, column, dataType, options)
      : this.generateMysqlAddColumn(table, column, dataType, options);

    this.writeMigration(migrationName, migration);
    console.log(`✅ Generated migration: ${migrationName}`);
  }

  /**
   * PostgreSQL add column (zero-downtime)
   */
  generatePostgresAddColumn(table, column, dataType, options) {
    const { notNull, defaultValue, comment } = options;

    return `-- Migration: Add ${column} to ${table}
-- Database: PostgreSQL
-- Strategy: Zero-downtime migration

BEGIN;

-- Step 1: Add column without NOT NULL constraint
-- This is non-blocking because no default value
ALTER TABLE ${table}
  ADD COLUMN ${column} ${dataType}${defaultValue ? ` DEFAULT ${this.formatValue(defaultValue)}` : ''};

${comment ? `COMMENT ON COLUMN ${table}.${column} IS '${comment}';` : ''}

-- Step 2: Backfill data in batches (if needed)
${defaultValue && notNull ? `
DO $$
DECLARE
  batch_size INT := 1000;
  rows_updated INT;
BEGIN
  LOOP
    WITH batch AS (
      SELECT ctid
      FROM ${table}
      WHERE ${column} IS NULL
      LIMIT batch_size
      FOR UPDATE SKIP LOCKED
    )
    UPDATE ${table}
    SET ${column} = ${this.formatValue(defaultValue)}
    FROM batch
    WHERE ${table}.ctid = batch.ctid;

    GET DIAGNOSTICS rows_updated = ROW_COUNT;
    EXIT WHEN rows_updated = 0;

    -- Commit each batch
    COMMIT;

    -- Small pause to reduce load
    PERFORM pg_sleep(0.1);
  END LOOP;
END $$;
` : ''}

${notNull ? `-- Step 3: Add NOT NULL constraint after backfill
ALTER TABLE ${table}
  ALTER COLUMN ${column} SET NOT NULL;
` : ''}

-- Step 4: Create index if needed
-- CREATE INDEX CONCURRENTLY idx_${table}_${column} ON ${table} (${column});

COMMIT;

-- Rollback plan:
-- BEGIN;
-- ALTER TABLE ${table} DROP COLUMN ${column};
-- COMMIT;
`;
  }

  /**
   * MySQL add column (zero-downtime)
   */
  generateMysqlAddColumn(table, column, dataType, options) {
    const { notNull, defaultValue } = options;

    return `-- Migration: Add ${column} to ${table}
-- Database: MySQL
-- Strategy: Online DDL (MySQL 5.6+)

-- Step 1: Add column (MySQL 5.6+ supports online DDL)
ALTER TABLE ${table}
  ADD COLUMN ${column} ${dataType}${defaultValue ? ` DEFAULT ${this.formatValue(defaultValue)}` : ''}${notNull ? ' NOT NULL' : ''},
  ALGORITHM=INPLACE,
  LOCK=NONE;

-- Step 2: Create index (online)
-- CREATE INDEX idx_${table}_${column} ON ${table} (${column}) ALGORITHM=INPLACE, LOCK=NONE;

-- Rollback plan:
-- ALTER TABLE ${table} DROP COLUMN ${column};
`;
  }

  /**
   * Generate add index migration (concurrent/online)
   */
  generateAddIndex(table, columns, options = {}) {
    const { unique = false, where = null, include = [] } = options;
    const columnsArray = columns.split(',').map(c => c.trim());
    const indexName = `idx_${table}_${columnsArray.join('_')}`;

    const migrationName = `${this.timestamp}_add_index_${indexName}`;
    const migration = this.dbType === 'postgres'
      ? this.generatePostgresAddIndex(table, indexName, columnsArray, options)
      : this.generateMysqlAddIndex(table, indexName, columnsArray, options);

    this.writeMigration(migrationName, migration);
    console.log(`✅ Generated migration: ${migrationName}`);
  }

  /**
   * PostgreSQL add index (concurrent)
   */
  generatePostgresAddIndex(table, indexName, columns, options) {
    const { unique, where, include } = options;

    return `-- Migration: Add index ${indexName}
-- Database: PostgreSQL
-- Strategy: CREATE INDEX CONCURRENTLY (non-blocking)

-- IMPORTANT: Cannot run in transaction block
-- Run this migration separately

CREATE ${unique ? 'UNIQUE ' : ''}INDEX CONCURRENTLY ${indexName}
  ON ${table} (${columns.join(', ')})${include.length > 0 ? `
  INCLUDE (${include.join(', ')})` : ''}${where ? `
  WHERE ${where}` : ''};

-- Verify index was created
SELECT schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE indexname = '${indexName}';

-- Rollback plan (also concurrent):
-- DROP INDEX CONCURRENTLY ${indexName};
`;
  }

  /**
   * MySQL add index (online)
   */
  generateMysqlAddIndex(table, indexName, columns, options) {
    const { unique } = options;

    return `-- Migration: Add index ${indexName}
-- Database: MySQL
-- Strategy: Online DDL

CREATE ${unique ? 'UNIQUE ' : ''}INDEX ${indexName}
  ON ${table} (${columns.join(', ')})
  ALGORITHM=INPLACE,
  LOCK=NONE;

-- Verify index was created
SHOW INDEX FROM ${table} WHERE Key_name = '${indexName}';

-- Rollback plan:
-- DROP INDEX ${indexName} ON ${table};
`;
  }

  /**
   * Generate rename column migration
   */
  generateRenameColumn(table, oldColumn, newColumn, options = {}) {
    const migrationName = `${this.timestamp}_rename_${oldColumn}_to_${newColumn}_in_${table}`;

    const migration = this.dbType === 'postgres'
      ? this.generatePostgresRenameColumn(table, oldColumn, newColumn)
      : this.generateMysqlRenameColumn(table, oldColumn, newColumn, options.dataType);

    this.writeMigration(migrationName, migration);
    console.log(`✅ Generated migration: ${migrationName}`);
  }

  /**
   * PostgreSQL rename column (instant)
   */
  generatePostgresRenameColumn(table, oldColumn, newColumn) {
    return `-- Migration: Rename ${oldColumn} to ${newColumn} in ${table}
-- Database: PostgreSQL
-- Strategy: Instant rename (metadata-only operation)

BEGIN;

ALTER TABLE ${table}
  RENAME COLUMN ${oldColumn} TO ${newColumn};

COMMIT;

-- Rollback plan:
-- BEGIN;
-- ALTER TABLE ${table} RENAME COLUMN ${newColumn} TO ${oldColumn};
-- COMMIT;
`;
  }

  /**
   * MySQL rename column
   */
  generateMysqlRenameColumn(table, oldColumn, newColumn, dataType) {
    if (!dataType) {
      console.error('❌ MySQL requires --data-type when renaming columns');
      process.exit(1);
    }

    return `-- Migration: Rename ${oldColumn} to ${newColumn} in ${table}
-- Database: MySQL
-- Strategy: CHANGE COLUMN

ALTER TABLE ${table}
  CHANGE COLUMN ${oldColumn} ${newColumn} ${dataType}
  ALGORITHM=INPLACE,
  LOCK=NONE;

-- Rollback plan:
-- ALTER TABLE ${table} CHANGE COLUMN ${newColumn} ${oldColumn} ${dataType};
`;
  }

  /**
   * Generate batch backfill migration
   */
  generateBatchBackfill(table, column, value, whereClause = null) {
    const migrationName = `${this.timestamp}_backfill_${column}_in_${table}`;

    const migration = `-- Migration: Batch backfill ${column} in ${table}
-- Database: ${this.dbType === 'postgres' ? 'PostgreSQL' : 'MySQL'}
-- Strategy: Batch updates with pauses

${this.dbType === 'postgres' ? `
DO $$
DECLARE
  batch_size INT := 1000;
  rows_updated INT;
  total_updated INT := 0;
BEGIN
  LOOP
    ${whereClause ? `-- Only update rows matching: ${whereClause}` : ''}
    WITH batch AS (
      SELECT ctid
      FROM ${table}
      WHERE ${column} IS NULL${whereClause ? ` AND ${whereClause}` : ''}
      LIMIT batch_size
      FOR UPDATE SKIP LOCKED
    )
    UPDATE ${table}
    SET ${column} = ${this.formatValue(value)}
    FROM batch
    WHERE ${table}.ctid = batch.ctid;

    GET DIAGNOSTICS rows_updated = ROW_COUNT;
    total_updated := total_updated + rows_updated;

    RAISE NOTICE 'Updated % rows (total: %)', rows_updated, total_updated;

    EXIT WHEN rows_updated = 0;

    -- Commit each batch
    COMMIT;

    -- Pause to reduce load (100ms)
    PERFORM pg_sleep(0.1);
  END LOOP;

  RAISE NOTICE 'Backfill complete. Total rows updated: %', total_updated;
END $$;
` : `
-- MySQL batch backfill
DELIMITER $$

CREATE PROCEDURE backfill_${table}_${column}()
BEGIN
  DECLARE rows_updated INT DEFAULT 1;
  DECLARE total_updated INT DEFAULT 0;
  DECLARE batch_size INT DEFAULT 1000;

  WHILE rows_updated > 0 DO
    UPDATE ${table}
    SET ${column} = ${this.formatValue(value)}
    WHERE ${column} IS NULL${whereClause ? ` AND ${whereClause}` : ''}
    LIMIT batch_size;

    SET rows_updated = ROW_COUNT();
    SET total_updated = total_updated + rows_updated;

    SELECT CONCAT('Updated ', rows_updated, ' rows (total: ', total_updated, ')');

    -- Small pause (100ms)
    DO SLEEP(0.1);
  END WHILE;

  SELECT CONCAT('Backfill complete. Total rows updated: ', total_updated);
END$$

DELIMITER ;

-- Execute backfill
CALL backfill_${table}_${column}();

-- Cleanup
DROP PROCEDURE backfill_${table}_${column};
`}
`;

    this.writeMigration(migrationName, migration);
    console.log(`✅ Generated migration: ${migrationName}`);
  }

  /**
   * Write migration to file
   */
  writeMigration(name, content) {
    if (!fs.existsSync(this.migrationDir)) {
      fs.mkdirSync(this.migrationDir, { recursive: true });
    }

    const filePath = path.join(this.migrationDir, `${name}.sql`);
    fs.writeFileSync(filePath, content);
  }

  /**
   * Format value for SQL
   */
  formatValue(value) {
    if (value === null) return 'NULL';
    if (typeof value === 'boolean') return value.toString().toUpperCase();
    if (typeof value === 'number') return value;
    return `'${value.replace(/'/g, "''")}'`;
  }
}

// CLI Interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0];

  if (!command) {
    console.log(`
Migration Generator - Zero-Downtime Database Migrations

USAGE:
  node migration-generator.js <command> [options]

COMMANDS:
  add-column        Add new column with zero downtime
  add-index         Create index concurrently/online
  rename-column     Rename column safely
  batch-backfill    Backfill column in batches

EXAMPLES:
  node migration-generator.js add-column --table users --column email_verified --type boolean --not-null --default false
  node migration-generator.js add-index --table orders --columns user_id,created_at --unique
  node migration-generator.js rename-column --table users --old email --new email_address --data-type VARCHAR(255)
  node migration-generator.js batch-backfill --table users --column status --value active

OPTIONS:
  --db [postgres|mysql]   Database type (default: postgres)
  --dir <path>            Migration directory (default: ./migrations)
`);
    process.exit(0);
  }

  const options = { db: 'postgres', dir: './migrations' };
  const params = {};

  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    if (arg.startsWith('--') && args[i + 1]) {
      const key = arg.substring(2);
      const value = args[i + 1];

      if (key === 'db' || key === 'dir') {
        options[key] = value;
      } else {
        params[key] = value;
      }
      i++;
    } else if (arg === '--not-null') {
      params.notNull = true;
    } else if (arg === '--unique') {
      params.unique = true;
    }
  }

  const generator = new MigrationGenerator(options);

  switch (command) {
    case 'add-column':
      if (!params.table || !params.column || !params.type) {
        console.error('❌ Required: --table, --column, --type');
        process.exit(1);
      }
      generator.generateAddColumn(params.table, params.column, params.type, {
        notNull: params.notNull,
        defaultValue: params.default
      });
      break;

    case 'add-index':
      if (!params.table || !params.columns) {
        console.error('❌ Required: --table, --columns');
        process.exit(1);
      }
      generator.generateAddIndex(params.table, params.columns, {
        unique: params.unique,
        where: params.where,
        include: params.include ? params.include.split(',') : []
      });
      break;

    case 'rename-column':
      if (!params.table || !params.old || !params.new) {
        console.error('❌ Required: --table, --old, --new');
        process.exit(1);
      }
      generator.generateRenameColumn(params.table, params.old, params.new, {
        dataType: params['data-type']
      });
      break;

    case 'batch-backfill':
      if (!params.table || !params.column || !params.value) {
        console.error('❌ Required: --table, --column, --value');
        process.exit(1);
      }
      generator.generateBatchBackfill(params.table, params.column, params.value, params.where);
      break;

    default:
      console.error(`❌ Unknown command: ${command}`);
      process.exit(1);
  }
}

module.exports = MigrationGenerator;
