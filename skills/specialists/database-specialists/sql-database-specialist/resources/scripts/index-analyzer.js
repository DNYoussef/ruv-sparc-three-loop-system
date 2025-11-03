#!/usr/bin/env node
/**
 * Index Analyzer - Database Index Effectiveness Analysis
 *
 * Analyzes database indexes to identify:
 * - Unused indexes (waste of space and write performance)
 * - Missing indexes (slow queries)
 * - Duplicate/redundant indexes
 * - Low cardinality indexes (ineffective)
 * - Bloated indexes (need rebuilding)
 * - Index fragmentation
 *
 * Usage:
 *   node index-analyzer.js --db postgres --database mydb
 *   node index-analyzer.js --db mysql --database mydb --table users
 */

const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

class IndexAnalyzer {
  constructor(options = {}) {
    this.dbType = options.db || 'postgres';
    this.database = options.database;
    this.table = options.table;
    this.findings = [];
  }

  /**
   * Analyze all indexes in database
   */
  async analyze() {
    console.log(`\n🔍 Analyzing indexes in ${this.database}...\n`);

    if (this.dbType === 'postgres') {
      await this.analyzePostgresIndexes();
    } else if (this.dbType === 'mysql') {
      await this.analyzeMysqlIndexes();
    }

    this.printFindings();
  }

  /**
   * Analyze PostgreSQL indexes
   */
  async analyzePostgresIndexes() {
    // 1. Find unused indexes
    console.log('📊 Checking for unused indexes...\n');
    const unusedQuery = `
      SELECT
        schemaname,
        tablename,
        indexname,
        pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
        idx_scan AS index_scans
      FROM pg_stat_user_indexes
      WHERE idx_scan = 0
        AND indexrelname NOT LIKE 'pg_toast%'
        ${this.table ? `AND tablename = '${this.table}'` : ''}
      ORDER BY pg_relation_size(indexrelid) DESC;
    `;

    await this.runPostgresQuery(unusedQuery, (rows) => {
      rows.forEach(row => {
        this.addFinding(
          'warning',
          `Unused index: ${row.indexname} on ${row.tablename}`,
          `${row.index_size} wasted, consider: DROP INDEX ${row.indexname};`,
          { table: row.tablename, index: row.indexname, size: row.index_size }
        );
      });
    });

    // 2. Find duplicate indexes
    console.log('📊 Checking for duplicate indexes...\n');
    const duplicateQuery = `
      SELECT
        schemaname,
        tablename,
        array_agg(indexname) AS indexes,
        indexdef
      FROM pg_indexes
      WHERE schemaname = 'public'
        ${this.table ? `AND tablename = '${this.table}'` : ''}
      GROUP BY schemaname, tablename, indexdef
      HAVING COUNT(*) > 1;
    `;

    await this.runPostgresQuery(duplicateQuery, (rows) => {
      rows.forEach(row => {
        this.addFinding(
          'critical',
          `Duplicate indexes on ${row.tablename}`,
          `Indexes: ${row.indexes}, keep only one`,
          { table: row.tablename, indexes: row.indexes }
        );
      });
    });

    // 3. Find indexes with low selectivity
    console.log('📊 Checking index selectivity...\n');
    const selectivityQuery = `
      SELECT
        schemaname,
        tablename,
        attname AS column_name,
        n_distinct,
        CASE
          WHEN n_distinct < 0 THEN ABS(n_distinct) * reltuples
          ELSE n_distinct
        END AS estimated_distinct_values,
        reltuples AS total_rows,
        CASE
          WHEN reltuples > 0 THEN
            ROUND(100.0 * (CASE WHEN n_distinct < 0 THEN ABS(n_distinct) ELSE n_distinct / reltuples END), 2)
          ELSE 0
        END AS selectivity_percent
      FROM pg_stats s
      JOIN pg_class c ON s.tablename = c.relname
      WHERE schemaname = 'public'
        ${this.table ? `AND tablename = '${this.table}'` : ''}
        AND n_distinct > -1
        AND n_distinct < 100
      ORDER BY selectivity_percent ASC;
    `;

    await this.runPostgresQuery(selectivityQuery, (rows) => {
      rows.forEach(row => {
        if (row.selectivity_percent < 1) {
          this.addFinding(
            'info',
            `Low cardinality column: ${row.tablename}.${row.column_name}`,
            `Only ${row.estimated_distinct_values} distinct values (${row.selectivity_percent}% selectivity). Index may be ineffective.`,
            { table: row.tablename, column: row.column_name }
          );
        }
      });
    });

    // 4. Find bloated indexes
    console.log('📊 Checking for bloated indexes...\n');
    const bloatQuery = `
      SELECT
        schemaname,
        tablename,
        indexname,
        pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
        ROUND(100.0 * pg_relation_size(indexrelid) / NULLIF(pg_total_relation_size(indexrelid), 0), 2) AS bloat_pct
      FROM pg_stat_user_indexes
      WHERE schemaname = 'public'
        ${this.table ? `AND tablename = '${this.table}'` : ''}
        AND pg_relation_size(indexrelid) > 10485760  -- > 10MB
      ORDER BY pg_relation_size(indexrelid) DESC;
    `;

    await this.runPostgresQuery(bloatQuery, (rows) => {
      rows.forEach(row => {
        if (row.bloat_pct > 50) {
          this.addFinding(
            'warning',
            `Bloated index: ${row.indexname} on ${row.tablename}`,
            `${row.index_size} size, ${row.bloat_pct}% bloat. Consider: REINDEX INDEX CONCURRENTLY ${row.indexname};`,
            { table: row.tablename, index: row.indexname }
          );
        }
      });
    });

    // 5. Suggest missing indexes based on seq scans
    console.log('📊 Checking for potential missing indexes...\n');
    const missingIndexQuery = `
      SELECT
        schemaname,
        tablename,
        seq_scan,
        seq_tup_read,
        idx_scan,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS table_size,
        ROUND(100.0 * seq_tup_read / NULLIF(seq_tup_read + idx_tup_fetch, 0), 2) AS seq_scan_pct
      FROM pg_stat_user_tables
      WHERE schemaname = 'public'
        ${this.table ? `AND tablename = '${this.table}'` : ''}
        AND seq_scan > 0
        AND seq_tup_read > 100000
      ORDER BY seq_tup_read DESC;
    `;

    await this.runPostgresQuery(missingIndexQuery, (rows) => {
      rows.forEach(row => {
        if (row.seq_scan_pct > 80) {
          this.addFinding(
            'critical',
            `Frequent sequential scans on ${row.tablename}`,
            `${row.seq_scan} scans, ${row.seq_scan_pct}% sequential. Add indexes on frequently queried columns.`,
            { table: row.tablename }
          );
        }
      });
    });
  }

  /**
   * Analyze MySQL indexes
   */
  async analyzeMysqlIndexes() {
    // 1. Find unused indexes
    console.log('📊 Checking for unused indexes...\n');
    const unusedQuery = `
      SELECT
        t.TABLE_SCHEMA AS database_name,
        t.TABLE_NAME AS table_name,
        s.INDEX_NAME AS index_name,
        ROUND(((s.DATA_LENGTH + s.INDEX_LENGTH) / 1024 / 1024), 2) AS size_mb
      FROM information_schema.TABLES t
      INNER JOIN information_schema.STATISTICS s
        ON t.TABLE_SCHEMA = s.TABLE_SCHEMA AND t.TABLE_NAME = s.TABLE_NAME
      LEFT JOIN performance_schema.table_io_waits_summary_by_index_usage p
        ON s.TABLE_SCHEMA = p.OBJECT_SCHEMA
        AND s.TABLE_NAME = p.OBJECT_NAME
        AND s.INDEX_NAME = p.INDEX_NAME
      WHERE t.TABLE_SCHEMA = '${this.database}'
        ${this.table ? `AND t.TABLE_NAME = '${this.table}'` : ''}
        AND s.INDEX_NAME != 'PRIMARY'
        AND (p.COUNT_STAR IS NULL OR p.COUNT_STAR = 0)
      GROUP BY s.TABLE_SCHEMA, s.TABLE_NAME, s.INDEX_NAME;
    `;

    await this.runMysqlQuery(unusedQuery, (rows) => {
      rows.forEach(row => {
        this.addFinding(
          'warning',
          `Unused index: ${row.index_name} on ${row.table_name}`,
          `${row.size_mb}MB wasted, consider: DROP INDEX ${row.index_name} ON ${row.table_name};`,
          { table: row.table_name, index: row.index_name }
        );
      });
    });

    // 2. Find duplicate indexes
    console.log('📊 Checking for duplicate indexes...\n');
    const duplicateQuery = `
      SELECT
        TABLE_NAME,
        GROUP_CONCAT(INDEX_NAME) AS duplicate_indexes,
        GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS columns
      FROM information_schema.STATISTICS
      WHERE TABLE_SCHEMA = '${this.database}'
        ${this.table ? `AND TABLE_NAME = '${this.table}'` : ''}
      GROUP BY TABLE_NAME, GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX)
      HAVING COUNT(DISTINCT INDEX_NAME) > 1;
    `;

    await this.runMysqlQuery(duplicateQuery, (rows) => {
      rows.forEach(row => {
        this.addFinding(
          'critical',
          `Duplicate indexes on ${row.TABLE_NAME}`,
          `Indexes: ${row.duplicate_indexes} on columns (${row.columns})`,
          { table: row.TABLE_NAME }
        );
      });
    });

    // 3. Check cardinality
    console.log('📊 Checking index cardinality...\n');
    const cardinalityQuery = `
      SELECT
        TABLE_NAME,
        INDEX_NAME,
        COLUMN_NAME,
        CARDINALITY,
        (SELECT TABLE_ROWS FROM information_schema.TABLES t
         WHERE t.TABLE_SCHEMA = s.TABLE_SCHEMA AND t.TABLE_NAME = s.TABLE_NAME) AS total_rows,
        ROUND(100.0 * CARDINALITY / NULLIF((SELECT TABLE_ROWS FROM information_schema.TABLES t
          WHERE t.TABLE_SCHEMA = s.TABLE_SCHEMA AND t.TABLE_NAME = s.TABLE_NAME), 0), 2) AS selectivity_pct
      FROM information_schema.STATISTICS s
      WHERE TABLE_SCHEMA = '${this.database}'
        ${this.table ? `AND TABLE_NAME = '${this.table}'` : ''}
        AND INDEX_NAME != 'PRIMARY'
        AND CARDINALITY IS NOT NULL
      HAVING selectivity_pct < 5
      ORDER BY selectivity_pct ASC;
    `;

    await this.runMysqlQuery(cardinalityQuery, (rows) => {
      rows.forEach(row => {
        this.addFinding(
          'info',
          `Low cardinality index: ${row.INDEX_NAME} on ${row.TABLE_NAME}`,
          `Only ${row.CARDINALITY} distinct values (${row.selectivity_pct}% selectivity). May be ineffective.`,
          { table: row.TABLE_NAME, index: row.INDEX_NAME }
        );
      });
    });
  }

  /**
   * Run PostgreSQL query
   */
  async runPostgresQuery(query, callback) {
    try {
      const psqlCmd = `psql -d ${this.database} -t -A -F',' -c "${query.replace(/"/g, '\\"')}"`;
      const { stdout } = await execPromise(psqlCmd);

      if (stdout.trim()) {
        const rows = this.parseCSV(stdout);
        callback(rows);
      }
    } catch (error) {
      console.error(`Warning: ${error.message}`);
    }
  }

  /**
   * Run MySQL query
   */
  async runMysqlQuery(query, callback) {
    try {
      const mysqlCmd = `mysql -D ${this.database} -e "${query.replace(/"/g, '\\"')}" --batch --skip-column-names`;
      const { stdout } = await execPromise(mysqlCmd);

      if (stdout.trim()) {
        const rows = stdout.trim().split('\n').map(line => {
          const values = line.split('\t');
          return Object.fromEntries(values.map((v, i) => [`col${i}`, v]));
        });
        callback(rows);
      }
    } catch (error) {
      console.error(`Warning: ${error.message}`);
    }
  }

  /**
   * Parse CSV output
   */
  parseCSV(csv) {
    const lines = csv.trim().split('\n');
    return lines.map(line => {
      const values = line.split(',');
      return Object.fromEntries(values.map((v, i) => [`col${i}`, v]));
    });
  }

  /**
   * Add finding
   */
  addFinding(level, issue, recommendation, metadata = {}) {
    this.findings.push({ level, issue, recommendation, metadata });

    const icon = {
      critical: '🔴',
      warning: '🟡',
      info: 'ℹ️'
    }[level];

    console.log(`${icon} ${issue}`);
    console.log(`   → ${recommendation}\n`);
  }

  /**
   * Print summary
   */
  printFindings() {
    console.log('\n═══════════════════════════════════════════════════════════');
    console.log('📊 INDEX ANALYSIS SUMMARY');
    console.log('═══════════════════════════════════════════════════════════\n');

    const critical = this.findings.filter(f => f.level === 'critical').length;
    const warnings = this.findings.filter(f => f.level === 'warning').length;
    const info = this.findings.filter(f => f.level === 'info').length;

    console.log(`🔴 Critical issues: ${critical}`);
    console.log(`🟡 Warnings:        ${warnings}`);
    console.log(`ℹ️  Info:           ${info}\n`);

    if (critical === 0 && warnings === 0) {
      console.log('✅ Index health is good!\n');
    } else {
      console.log('⚠️  Action items found. Review recommendations above.\n');
    }
  }
}

// CLI Interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const options = {};

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--db' && args[i + 1]) {
      options.db = args[i + 1];
      i++;
    } else if (args[i] === '--database' && args[i + 1]) {
      options.database = args[i + 1];
      i++;
    } else if (args[i] === '--table' && args[i + 1]) {
      options.table = args[i + 1];
      i++;
    }
  }

  if (!options.database) {
    console.error('Usage: node index-analyzer.js --db [postgres|mysql] --database <dbname> [--table <tablename>]');
    process.exit(1);
  }

  const analyzer = new IndexAnalyzer(options);
  analyzer.analyze().then(() => {
    process.exit(0);
  }).catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
  });
}

module.exports = IndexAnalyzer;
