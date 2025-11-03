#!/usr/bin/env node
/**
 * Query Optimizer - EXPLAIN Plan Analysis & Query Optimization
 *
 * Analyzes EXPLAIN plans and provides optimization recommendations:
 * - Detects sequential scans on large tables
 * - Identifies missing indexes
 * - Suggests query rewrites
 * - Analyzes JOIN strategies
 * - Detects N+1 query patterns
 * - Recommends covering indexes
 *
 * Usage:
 *   node query-optimizer.js --db postgres --explain explain.json
 *   node query-optimizer.js --db mysql --query query.sql --auto-analyze
 */

const fs = require('fs');

class QueryOptimizer {
  constructor(options = {}) {
    this.dbType = options.db || 'postgres';
    this.recommendations = [];
    this.score = 100;
  }

  /**
   * Analyze EXPLAIN plan output
   */
  analyzeExplain(explainPath) {
    const explainData = JSON.parse(fs.readFileSync(explainPath, 'utf-8'));

    console.log('\n🔍 Analyzing EXPLAIN plan...\n');

    if (this.dbType === 'postgres') {
      this.analyzePostgresExplain(explainData);
    } else if (this.dbType === 'mysql') {
      this.analyzeMysqlExplain(explainData);
    }

    this.printRecommendations();
    return this.score;
  }

  /**
   * Analyze PostgreSQL EXPLAIN (ANALYZE, BUFFERS) output
   */
  analyzePostgresExplain(explain) {
    const plan = explain[0]?.Plan || explain.Plan;

    if (!plan) {
      console.error('❌ Invalid EXPLAIN format');
      return;
    }

    this.traversePlan(plan);
  }

  /**
   * Traverse PostgreSQL query plan tree
   */
  traversePlan(node, depth = 0) {
    const indent = '  '.repeat(depth);

    console.log(`${indent}📌 ${node['Node Type']}`);

    // Check for Sequential Scan (bad for large tables)
    if (node['Node Type'] === 'Seq Scan') {
      const rows = node['Plan Rows'] || 0;
      if (rows > 1000) {
        this.addRecommendation(
          'critical',
          `Sequential Scan on ${node['Relation Name']} (${rows} rows)`,
          `CREATE INDEX idx_${node['Relation Name']}_<columns> ON ${node['Relation Name']} (<filter_columns>);`,
          -20
        );
      } else if (rows > 100) {
        this.addRecommendation(
          'warning',
          `Sequential Scan on ${node['Relation Name']} (${rows} rows)`,
          'Consider adding an index if this table grows',
          -5
        );
      }
    }

    // Check for Index Scan vs Index Only Scan
    if (node['Node Type'] === 'Index Scan') {
      console.log(`${indent}  ✅ Using index: ${node['Index Name']}`);

      // Suggest covering index if heap fetches are high
      if (node['Heap Fetches'] && node['Heap Fetches'] > 1000) {
        this.addRecommendation(
          'optimization',
          `High heap fetches (${node['Heap Fetches']}) on ${node['Index Name']}`,
          `CREATE INDEX ${node['Index Name']}_covering ON ${node['Relation Name']} (...) INCLUDE (...);`,
          -3
        );
      }
    }

    if (node['Node Type'] === 'Index Only Scan') {
      console.log(`${indent}  🚀 Index Only Scan (optimal)`);
    }

    // Check for Nested Loop with high cost
    if (node['Node Type'] === 'Nested Loop') {
      const actualTime = node['Actual Total Time'] || 0;
      if (actualTime > 100) {
        this.addRecommendation(
          'warning',
          `Slow Nested Loop (${actualTime.toFixed(2)}ms)`,
          'Consider Hash Join by adding WHERE conditions or changing join order',
          -10
        );
      }
    }

    // Check for Hash Join (usually good)
    if (node['Node Type'] === 'Hash Join') {
      console.log(`${indent}  ✅ Hash Join (efficient)`);
    }

    // Check for high cost operations
    const totalCost = node['Total Cost'] || 0;
    const startupCost = node['Startup Cost'] || 0;
    if (totalCost > 10000) {
      this.addRecommendation(
        'warning',
        `High query cost (${totalCost.toFixed(2)})`,
        'Review query structure and indexes',
        -5
      );
    }

    // Analyze buffer usage (if BUFFERS was used)
    if (node['Shared Hit Blocks']) {
      const hitRatio = node['Shared Hit Blocks'] /
        (node['Shared Hit Blocks'] + (node['Shared Read Blocks'] || 0) + 0.0001);

      if (hitRatio < 0.9) {
        console.log(`${indent}  ⚠️  Cache hit ratio: ${(hitRatio * 100).toFixed(1)}%`);
      } else {
        console.log(`${indent}  ✅ Cache hit ratio: ${(hitRatio * 100).toFixed(1)}%`);
      }
    }

    // Recursively traverse child plans
    if (node.Plans) {
      node.Plans.forEach(childPlan => this.traversePlan(childPlan, depth + 1));
    }
  }

  /**
   * Analyze MySQL EXPLAIN output
   */
  analyzeMysqlExplain(explain) {
    // MySQL EXPLAIN is typically array of rows
    const rows = Array.isArray(explain) ? explain : [explain];

    rows.forEach(row => {
      console.log(`📌 Table: ${row.table}`);

      // Check for full table scan
      if (row.type === 'ALL') {
        this.addRecommendation(
          'critical',
          `Full table scan on ${row.table} (${row.rows} rows)`,
          `CREATE INDEX idx_${row.table}_<columns> ON ${row.table} (<key_columns>);`,
          -20
        );
      }

      // Check for index usage
      if (row.type === 'index' || row.type === 'range') {
        console.log(`  ✅ Using index: ${row.possible_keys || row.key}`);
      }

      // Check for filesort
      if (row.Extra && row.Extra.includes('Using filesort')) {
        this.addRecommendation(
          'warning',
          `Filesort detected on ${row.table}`,
          'Add index on ORDER BY columns',
          -10
        );
      }

      // Check for temporary table
      if (row.Extra && row.Extra.includes('Using temporary')) {
        this.addRecommendation(
          'warning',
          `Temporary table created for ${row.table}`,
          'Optimize GROUP BY or DISTINCT clauses',
          -8
        );
      }

      // Check rows examined
      if (row.rows > 10000) {
        this.addRecommendation(
          'warning',
          `High row count examined: ${row.rows}`,
          'Add WHERE conditions or indexes',
          -5
        );
      }
    });
  }

  /**
   * Add optimization recommendation
   */
  addRecommendation(level, issue, suggestion, scoreDelta = 0) {
    this.recommendations.push({ level, issue, suggestion });
    this.score += scoreDelta;

    const icon = {
      critical: '🔴',
      warning: '🟡',
      optimization: '🔵',
      info: 'ℹ️'
    }[level];

    console.log(`  ${icon} ${issue.toUpperCase()}`);
    console.log(`     → ${suggestion}\n`);
  }

  /**
   * Print optimization summary
   */
  printRecommendations() {
    console.log('\n═══════════════════════════════════════════════════════════');
    console.log('📊 OPTIMIZATION SUMMARY');
    console.log('═══════════════════════════════════════════════════════════\n');

    const critical = this.recommendations.filter(r => r.level === 'critical').length;
    const warnings = this.recommendations.filter(r => r.level === 'warning').length;
    const optimizations = this.recommendations.filter(r => r.level === 'optimization').length;

    console.log(`🔴 Critical issues:      ${critical}`);
    console.log(`🟡 Warnings:             ${warnings}`);
    console.log(`🔵 Optimization tips:    ${optimizations}`);
    console.log(`\n📈 Performance score:    ${Math.max(0, this.score)}/100\n`);

    if (this.score >= 90) {
      console.log('✅ Query is well-optimized!\n');
    } else if (this.score >= 70) {
      console.log('⚠️  Query has some optimization opportunities.\n');
    } else {
      console.log('❌ Query needs significant optimization.\n');
    }

    // Print actionable recommendations
    if (this.recommendations.length > 0) {
      console.log('🔧 ACTION ITEMS:\n');
      this.recommendations.forEach((rec, idx) => {
        console.log(`${idx + 1}. ${rec.suggestion}`);
      });
      console.log('');
    }
  }

  /**
   * Detect N+1 query pattern from query log
   */
  detectN1Pattern(queries) {
    console.log('\n🔍 Detecting N+1 query patterns...\n');

    const queryCounts = {};
    queries.forEach(q => {
      const normalized = this.normalizeQuery(q);
      queryCounts[normalized] = (queryCounts[normalized] || 0) + 1;
    });

    Object.entries(queryCounts).forEach(([query, count]) => {
      if (count > 10) {
        this.addRecommendation(
          'critical',
          `Possible N+1 pattern detected (${count} identical queries)`,
          'Replace with single JOIN query or use eager loading',
          -15
        );
        console.log(`   Query: ${query.substring(0, 80)}...`);
      }
    });
  }

  /**
   * Normalize query for pattern matching
   */
  normalizeQuery(query) {
    return query
      .replace(/\d+/g, '?')  // Replace numbers with ?
      .replace(/'[^']*'/g, '?')  // Replace strings with ?
      .replace(/\s+/g, ' ')  // Normalize whitespace
      .trim()
      .toLowerCase();
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
    } else if (args[i] === '--explain' && args[i + 1]) {
      options.explainPath = args[i + 1];
      i++;
    }
  }

  if (!options.explainPath) {
    console.error('Usage: node query-optimizer.js --db [postgres|mysql] --explain explain.json');
    console.error('\nExample:');
    console.error('  psql -d mydb -c "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT ..." > explain.json');
    console.error('  node query-optimizer.js --db postgres --explain explain.json');
    process.exit(1);
  }

  const optimizer = new QueryOptimizer(options);
  const score = optimizer.analyzeExplain(options.explainPath);

  process.exit(score >= 70 ? 0 : 1);
}

module.exports = QueryOptimizer;
