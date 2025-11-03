/**
 * Test suite for ReasoningBank migration to AgentDB
 * Tests migration tool, data integrity, and backward compatibility.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const sqlite3 = require('sqlite3').verbose();
const assert = require('assert');

class MigrationTests {
  constructor() {
    this.testDir = path.join(__dirname, '../../../tests/temp/migration-tests');
    this.sourcePath = path.join(this.testDir, 'source.db');
    this.targetPath = path.join(this.testDir, 'target.db');
    this.results = {
      passed: 0,
      failed: 0,
      errors: []
    };
  }

  setup() {
    // Create test directory
    if (!fs.existsSync(this.testDir)) {
      fs.mkdirSync(this.testDir, { recursive: true });
    }

    // Create source database with test data
    this.createSourceDB();
  }

  cleanup() {
    // Clean up test files
    if (fs.existsSync(this.testDir)) {
      fs.rmSync(this.testDir, { recursive: true, force: true });
    }
  }

  createSourceDB() {
    const db = new sqlite3.Database(this.sourcePath);

    db.serialize(() => {
      // Create legacy schema
      db.run(`
        CREATE TABLE patterns (
          id TEXT PRIMARY KEY,
          domain TEXT,
          pattern_text TEXT,
          confidence REAL,
          usage_count INTEGER,
          success_count INTEGER
        )
      `);

      db.run(`
        CREATE TABLE trajectories (
          id TEXT PRIMARY KEY,
          task TEXT,
          steps TEXT,
          outcome TEXT,
          metrics TEXT
        )
      `);

      db.run(`CREATE TABLE memories (id TEXT PRIMARY KEY, data TEXT)`);

      // Insert test patterns
      const patterns = [
        ['pat_1', 'api-optimization', 'Use JWT for auth', 0.9, 10, 9],
        ['pat_2', 'debugging', 'Add logging', 0.7, 5, 3],
        ['pat_3', 'testing', 'Write unit tests first', 0.95, 20, 19]
      ];

      const insertPattern = db.prepare('INSERT INTO patterns VALUES (?, ?, ?, ?, ?, ?)');
      patterns.forEach(p => insertPattern.run(p));
      insertPattern.finalize();

      // Insert test trajectories
      const trajectories = [
        [
          'traj_1',
          'optimize-api',
          JSON.stringify([
            { action: 'profile', result: 'found slow endpoint' },
            { action: 'add-cache', result: 'improved latency' }
          ]),
          'success',
          JSON.stringify({ latency_reduction: 0.8 })
        ],
        [
          'traj_2',
          'fix-bug',
          JSON.stringify([
            { action: 'reproduce', result: 'confirmed bug' },
            { action: 'fix', result: 'bug fixed' }
          ]),
          'success',
          JSON.stringify({ time_spent: 30 })
        ]
      ];

      const insertTrajectory = db.prepare('INSERT INTO trajectories VALUES (?, ?, ?, ?, ?)');
      trajectories.forEach(t => insertTrajectory.run(t));
      insertTrajectory.finalize();
    });

    db.close();
  }

  async runTest(name, fn) {
    try {
      console.log(`\nRunning test: ${name}`);
      await fn();
      this.results.passed++;
      console.log(`✓ ${name} passed`);
    } catch (error) {
      this.results.failed++;
      this.results.errors.push({ test: name, error: error.message });
      console.error(`✗ ${name} failed: ${error.message}`);
    }
  }

  // Test 1: Source database validation
  async testSourceValidation() {
    await this.runTest('Source database validation', () => {
      assert(fs.existsSync(this.sourcePath), 'Source database should exist');

      const db = new sqlite3.Database(this.sourcePath);

      return new Promise((resolve, reject) => {
        db.all("SELECT name FROM sqlite_master WHERE type='table'", (err, tables) => {
          if (err) reject(err);

          const tableNames = tables.map(t => t.name);
          assert(tableNames.includes('patterns'), 'Should have patterns table');
          assert(tableNames.includes('trajectories'), 'Should have trajectories table');
          assert(tableNames.includes('memories'), 'Should have memories table');

          db.close();
          resolve();
        });
      });
    });
  }

  // Test 2: Migration execution
  async testMigrationExecution() {
    await this.runTest('Migration execution', () => {
      const migrationScript = path.join(__dirname, '../resources/migration-tool.py');

      // Run migration
      const result = execSync(
        `python "${migrationScript}" --source "${this.sourcePath}" --target "${this.targetPath}"`,
        { encoding: 'utf8' }
      );

      assert(fs.existsSync(this.targetPath), 'Target database should be created');
      assert(result.includes('Migration successful'), 'Migration should succeed');
    });
  }

  // Test 3: Data integrity
  async testDataIntegrity() {
    await this.runTest('Data integrity', () => {
      const sourceDB = new sqlite3.Database(this.sourcePath);
      const targetDB = new sqlite3.Database(this.targetPath);

      return new Promise((resolve, reject) => {
        // Count source patterns
        sourceDB.get('SELECT COUNT(*) as count FROM patterns', (err, sourceRow) => {
          if (err) reject(err);

          // Count target patterns
          targetDB.get('SELECT COUNT(*) as count FROM patterns WHERE type="pattern"', (err, targetRow) => {
            if (err) reject(err);

            assert.strictEqual(
              sourceRow.count,
              targetRow.count,
              'Pattern counts should match'
            );

            sourceDB.close();
            targetDB.close();
            resolve();
          });
        });
      });
    });
  }

  // Test 4: Embedding generation
  async testEmbeddingGeneration() {
    await this.runTest('Embedding generation', () => {
      const db = new sqlite3.Database(this.targetPath);

      return new Promise((resolve, reject) => {
        db.all('SELECT id, embedding FROM patterns LIMIT 5', (err, rows) => {
          if (err) reject(err);

          assert(rows.length > 0, 'Should have migrated patterns');

          rows.forEach(row => {
            assert(row.embedding !== null, `Pattern ${row.id} should have embedding`);
            assert(row.embedding.length > 0, `Pattern ${row.id} embedding should not be empty`);
          });

          db.close();
          resolve();
        });
      });
    });
  }

  // Test 5: Schema compatibility
  async testSchemaCompatibility() {
    await this.runTest('Schema compatibility', () => {
      const db = new sqlite3.Database(this.targetPath);

      return new Promise((resolve, reject) => {
        db.all("PRAGMA table_info(patterns)", (err, columns) => {
          if (err) reject(err);

          const columnNames = columns.map(c => c.name);
          const requiredColumns = [
            'id', 'type', 'domain', 'pattern_data', 'embedding',
            'confidence', 'usage_count', 'success_count',
            'created_at', 'last_used'
          ];

          requiredColumns.forEach(col => {
            assert(
              columnNames.includes(col),
              `Should have column: ${col}`
            );
          });

          db.close();
          resolve();
        });
      });
    });
  }

  // Test 6: Index creation
  async testIndexCreation() {
    await this.runTest('Index creation', () => {
      const db = new sqlite3.Database(this.targetPath);

      return new Promise((resolve, reject) => {
        db.all("SELECT name FROM sqlite_master WHERE type='index'", (err, indexes) => {
          if (err) reject(err);

          const indexNames = indexes.map(i => i.name);

          assert(
            indexNames.some(n => n.includes('domain')),
            'Should have domain index'
          );
          assert(
            indexNames.some(n => n.includes('confidence')),
            'Should have confidence index'
          );

          db.close();
          resolve();
        });
      });
    });
  }

  // Test 7: Metadata preservation
  async testMetadataPreservation() {
    await this.runTest('Metadata preservation', () => {
      const db = new sqlite3.Database(this.targetPath);

      return new Promise((resolve, reject) => {
        db.get(
          `SELECT * FROM patterns WHERE id LIKE 'pat_%' ORDER BY id LIMIT 1`,
          (err, row) => {
            if (err) reject(err);

            assert(row !== undefined, 'Should have migrated pattern');
            assert(row.confidence > 0, 'Confidence should be preserved');
            assert(row.usage_count > 0, 'Usage count should be preserved');
            assert(row.success_count >= 0, 'Success count should be preserved');

            db.close();
            resolve();
          }
        );
      });
    });
  }

  // Test 8: Trajectory migration
  async testTrajectoryMigration() {
    await this.runTest('Trajectory migration', () => {
      const sourceDB = new sqlite3.Database(this.sourcePath);
      const targetDB = new sqlite3.Database(this.targetPath);

      return new Promise((resolve, reject) => {
        sourceDB.get('SELECT COUNT(*) as count FROM trajectories', (err, sourceRow) => {
          if (err) reject(err);

          targetDB.get(
            'SELECT COUNT(*) as count FROM patterns WHERE type="trajectory"',
            (err, targetRow) => {
              if (err) reject(err);

              assert.strictEqual(
                sourceRow.count,
                targetRow.count,
                'Trajectory counts should match'
              );

              sourceDB.close();
              targetDB.close();
              resolve();
            }
          );
        });
      });
    });
  }

  async runAll() {
    console.log('Starting ReasoningBank migration tests...\n');

    this.setup();

    await this.testSourceValidation();
    await this.testMigrationExecution();
    await this.testDataIntegrity();
    await this.testEmbeddingGeneration();
    await this.testSchemaCompatibility();
    await this.testIndexCreation();
    await this.testMetadataPreservation();
    await this.testTrajectoryMigration();

    this.cleanup();

    console.log('\n' + '='.repeat(60));
    console.log('Test Summary:');
    console.log(`  Passed: ${this.results.passed}`);
    console.log(`  Failed: ${this.results.failed}`);

    if (this.results.failed > 0) {
      console.log('\nFailed tests:');
      this.results.errors.forEach(({ test, error }) => {
        console.log(`  - ${test}: ${error}`);
      });
    }
    console.log('='.repeat(60));

    return this.results.failed === 0;
  }
}

// Run tests
if (require.main === module) {
  const tests = new MigrationTests();
  tests.runAll().then(success => {
    process.exit(success ? 0 : 1);
  });
}

module.exports = { MigrationTests };
