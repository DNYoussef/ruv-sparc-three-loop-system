/**
 * Test Suite for Migration Generator
 * Run with: node migration-generator.test.js
 */

const fs = require('fs');
const path = require('path');
const MigrationGenerator = require('../resources/scripts/migration-generator');

class TestRunner {
  constructor() {
    this.passed = 0;
    this.failed = 0;
    this.tests = [];
  }

  test(description, fn) {
    this.tests.push({ description, fn });
  }

  async run() {
    console.log('\n🧪 Running Migration Generator Tests\n');
    console.log('═'.repeat(60));

    for (const { description, fn } of this.tests) {
      try {
        await fn();
        console.log(`✅ PASS: ${description}`);
        this.passed++;
      } catch (error) {
        console.log(`❌ FAIL: ${description}`);
        console.log(`   Error: ${error.message}`);
        this.failed++;
      }
    }

    console.log('═'.repeat(60));
    console.log(`\n📊 Results: ${this.passed} passed, ${this.failed} failed\n`);

    return this.failed === 0;
  }
}

// Assertion helpers
function assertTrue(condition, message) {
  if (!condition) {
    throw new Error(message || 'Assertion failed');
  }
}

function assertContains(haystack, needle, message) {
  if (!haystack.includes(needle)) {
    throw new Error(message || `Expected to find "${needle}" in content`);
  }
}

function assertNotContains(haystack, needle, message) {
  if (haystack.includes(needle)) {
    throw new Error(message || `Did not expect to find "${needle}" in content`);
  }
}

// Create temp directory
const TEST_DIR = path.join(__dirname, '.test-temp-migrations');
if (!fs.existsSync(TEST_DIR)) {
  fs.mkdirSync(TEST_DIR, { recursive: true });
}

function cleanup() {
  if (fs.existsSync(TEST_DIR)) {
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  }
}

function getMigrationFiles() {
  if (!fs.existsSync(TEST_DIR)) return [];
  return fs.readdirSync(TEST_DIR).filter(f => f.endsWith('.sql'));
}

function readMigrationFile(filename) {
  return fs.readFileSync(path.join(TEST_DIR, filename), 'utf-8');
}

// Tests
const runner = new TestRunner();

runner.test('PostgreSQL add column migration generated correctly', async () => {
  const generator = new MigrationGenerator({ db: 'postgres', dir: TEST_DIR });
  generator.generateAddColumn('users', 'email_verified', 'BOOLEAN', {
    notNull: true,
    defaultValue: false
  });

  const files = getMigrationFiles();
  assertTrue(files.length === 1, 'Should generate one migration file');

  const content = readMigrationFile(files[0]);
  assertContains(content, 'ALTER TABLE users', 'Should alter users table');
  assertContains(content, 'ADD COLUMN email_verified BOOLEAN', 'Should add column');
  assertContains(content, 'DEFAULT false', 'Should include default value');
  assertContains(content, 'SET NOT NULL', 'Should add NOT NULL constraint');
  assertContains(content, 'DO $$', 'Should include batch backfill');
});

runner.test('PostgreSQL add column without NOT NULL is simpler', async () => {
  cleanup();
  fs.mkdirSync(TEST_DIR, { recursive: true });

  const generator = new MigrationGenerator({ db: 'postgres', dir: TEST_DIR });
  generator.generateAddColumn('products', 'description', 'TEXT', {
    notNull: false
  });

  const files = getMigrationFiles();
  const content = readMigrationFile(files[0]);

  assertContains(content, 'ADD COLUMN description TEXT', 'Should add column');
  assertNotContains(content, 'SET NOT NULL', 'Should not add NOT NULL');
  assertNotContains(content, 'DO $$', 'Should not include backfill for nullable column');
});

runner.test('MySQL add column uses ALGORITHM=INPLACE', async () => {
  cleanup();
  fs.mkdirSync(TEST_DIR, { recursive: true });

  const generator = new MigrationGenerator({ db: 'mysql', dir: TEST_DIR });
  generator.generateAddColumn('users', 'phone', 'VARCHAR(20)', {
    notNull: false
  });

  const files = getMigrationFiles();
  const content = readMigrationFile(files[0]);

  assertContains(content, 'ALTER TABLE users', 'Should alter table');
  assertContains(content, 'ADD COLUMN phone VARCHAR(20)', 'Should add column');
  assertContains(content, 'ALGORITHM=INPLACE', 'Should use online DDL');
  assertContains(content, 'LOCK=NONE', 'Should not lock table');
});

runner.test('PostgreSQL add index uses CREATE INDEX CONCURRENTLY', async () => {
  cleanup();
  fs.mkdirSync(TEST_DIR, { recursive: true });

  const generator = new MigrationGenerator({ db: 'postgres', dir: TEST_DIR });
  generator.generateAddIndex('orders', 'user_id,created_at', {
    unique: false
  });

  const files = getMigrationFiles();
  const content = readMigrationFile(files[0]);

  assertContains(content, 'CREATE INDEX CONCURRENTLY', 'Should use concurrent index creation');
  assertContains(content, 'idx_orders_user_id_created_at', 'Should generate index name');
  assertContains(content, 'ON orders (user_id, created_at)', 'Should include columns');
  assertNotContains(content, 'BEGIN;', 'Should not use transaction');
});

runner.test('PostgreSQL unique index includes UNIQUE keyword', async () => {
  cleanup();
  fs.mkdirSync(TEST_DIR, { recursive: true });

  const generator = new MigrationGenerator({ db: 'postgres', dir: TEST_DIR });
  generator.generateAddIndex('users', 'email', {
    unique: true
  });

  const files = getMigrationFiles();
  const content = readMigrationFile(files[0]);

  assertContains(content, 'CREATE UNIQUE INDEX CONCURRENTLY', 'Should create unique index');
});

runner.test('PostgreSQL partial index includes WHERE clause', async () => {
  cleanup();
  fs.mkdirSync(TEST_DIR, { recursive: true });

  const generator = new MigrationGenerator({ db: 'postgres', dir: TEST_DIR });
  generator.generateAddIndex('orders', 'created_at', {
    where: "status = 'pending'"
  });

  const files = getMigrationFiles();
  const content = readMigrationFile(files[0]);

  assertContains(content, "WHERE status = 'pending'", 'Should include WHERE clause');
});

runner.test('PostgreSQL covering index includes INCLUDE', async () => {
  cleanup();
  fs.mkdirSync(TEST_DIR, { recursive: true });

  const generator = new MigrationGenerator({ db: 'postgres', dir: TEST_DIR });
  generator.generateAddIndex('orders', 'user_id', {
    include: ['total', 'status']
  });

  const files = getMigrationFiles();
  const content = readMigrationFile(files[0]);

  assertContains(content, 'INCLUDE (total, status)', 'Should include covering columns');
});

runner.test('MySQL add index uses ALGORITHM=INPLACE', async () => {
  cleanup();
  fs.mkdirSync(TEST_DIR, { recursive: true });

  const generator = new MigrationGenerator({ db: 'mysql', dir: TEST_DIR });
  generator.generateAddIndex('products', 'category_id', {});

  const files = getMigrationFiles();
  const content = readMigrationFile(files[0]);

  assertContains(content, 'CREATE INDEX', 'Should create index');
  assertContains(content, 'ALGORITHM=INPLACE', 'Should use online DDL');
  assertContains(content, 'LOCK=NONE', 'Should not lock table');
});

runner.test('PostgreSQL rename column is instant', async () => {
  cleanup();
  fs.mkdirSync(TEST_DIR, { recursive: true });

  const generator = new MigrationGenerator({ db: 'postgres', dir: TEST_DIR });
  generator.generateRenameColumn('users', 'email', 'email_address');

  const files = getMigrationFiles();
  const content = readMigrationFile(files[0]);

  assertContains(content, 'ALTER TABLE users', 'Should alter table');
  assertContains(content, 'RENAME COLUMN email TO email_address', 'Should rename column');
  assertContains(content, 'BEGIN;', 'Should use transaction');
  assertContains(content, 'COMMIT;', 'Should commit');
});

runner.test('PostgreSQL batch backfill uses DO $$ block', async () => {
  cleanup();
  fs.mkdirSync(TEST_DIR, { recursive: true });

  const generator = new MigrationGenerator({ db: 'postgres', dir: TEST_DIR });
  generator.generateBatchBackfill('users', 'status', 'active');

  const files = getMigrationFiles();
  const content = readMigrationFile(files[0]);

  assertContains(content, 'DO $$', 'Should use DO block');
  assertContains(content, 'batch_size INT := 1000', 'Should define batch size');
  assertContains(content, 'FOR UPDATE SKIP LOCKED', 'Should skip locked rows');
  assertContains(content, 'COMMIT;', 'Should commit each batch');
  assertContains(content, 'pg_sleep(0.1)', 'Should pause between batches');
});

runner.test('MySQL batch backfill uses stored procedure', async () => {
  cleanup();
  fs.mkdirSync(TEST_DIR, { recursive: true });

  const generator = new MigrationGenerator({ db: 'mysql', dir: TEST_DIR });
  generator.generateBatchBackfill('orders', 'currency', 'USD');

  const files = getMigrationFiles();
  const content = readMigrationFile(files[0]);

  assertContains(content, 'CREATE PROCEDURE', 'Should create stored procedure');
  assertContains(content, 'CALL backfill_', 'Should call procedure');
  assertContains(content, 'DROP PROCEDURE', 'Should cleanup procedure');
  assertContains(content, 'DO SLEEP(0.1)', 'Should pause between batches');
});

runner.test('Migrations include rollback instructions', async () => {
  cleanup();
  fs.mkdirSync(TEST_DIR, { recursive: true });

  const generator = new MigrationGenerator({ db: 'postgres', dir: TEST_DIR });
  generator.generateAddColumn('users', 'test_column', 'VARCHAR(100)');

  const files = getMigrationFiles();
  const content = readMigrationFile(files[0]);

  assertContains(content, 'Rollback plan:', 'Should include rollback plan');
  assertContains(content, 'DROP COLUMN', 'Should show how to rollback');
});

runner.test('Value formatting works correctly', async () => {
  const generator = new MigrationGenerator();

  assertTrue(generator.formatValue(null) === 'NULL', 'null should format as NULL');
  assertTrue(generator.formatValue(true) === 'TRUE', 'true should format as TRUE');
  assertTrue(generator.formatValue(false) === 'FALSE', 'false should format as FALSE');
  assertTrue(generator.formatValue(123) === 123, 'numbers should pass through');
  assertTrue(generator.formatValue('test') === "'test'", 'strings should be quoted');
  assertTrue(generator.formatValue("it's") === "'it''s'", 'apostrophes should be escaped');
});

// Run tests
runner.run().then(success => {
  cleanup();
  process.exit(success ? 0 : 1);
}).catch(error => {
  console.error('Test runner error:', error);
  cleanup();
  process.exit(1);
});
