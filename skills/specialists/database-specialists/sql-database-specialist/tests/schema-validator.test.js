/**
 * Test Suite for Schema Validator
 * Run with: node schema-validator.test.js
 */

const fs = require('fs');
const path = require('path');
const SchemaValidator = require('../resources/scripts/schema-validator');

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
    console.log('\n🧪 Running Schema Validator Tests\n');
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

// Assertion helper
function assertEquals(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(message || `Expected ${expected}, got ${actual}`);
  }
}

function assertTrue(condition, message) {
  if (!condition) {
    throw new Error(message || 'Assertion failed');
  }
}

// Test fixtures
const VALID_SCHEMA = `
CREATE TABLE users (
  id BIGSERIAL PRIMARY KEY,
  email VARCHAR(255) NOT NULL UNIQUE,
  name VARCHAR(100) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE orders (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL,
  total DECIMAL(10, 2) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE INDEX idx_orders_user_id ON orders (user_id);
`;

const INVALID_SCHEMA_NO_PK = `
CREATE TABLE users (
  email VARCHAR(255) NOT NULL,
  name VARCHAR(100)
);
`;

const INVALID_SCHEMA_FLOAT_MONEY = `
CREATE TABLE orders (
  id BIGSERIAL PRIMARY KEY,
  total FLOAT,
  price DOUBLE PRECISION
);
`;

const INVALID_SCHEMA_MISSING_FK_INDEX = `
CREATE TABLE orders (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users (id)
);
`;

const INVALID_SCHEMA_BAD_NAMING = `
CREATE TABLE UserAccounts (
  ID BIGSERIAL PRIMARY KEY,
  EmailAddress VARCHAR(255)
);
`;

// Create temp directory for test files
const TEST_DIR = path.join(__dirname, '.test-temp');
if (!fs.existsSync(TEST_DIR)) {
  fs.mkdirSync(TEST_DIR, { recursive: true });
}

function writeTestSchema(filename, content) {
  const filePath = path.join(TEST_DIR, filename);
  fs.writeFileSync(filePath, content);
  return filePath;
}

function cleanup() {
  if (fs.existsSync(TEST_DIR)) {
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  }
}

// Tests
const runner = new TestRunner();

runner.test('Valid schema passes validation', async () => {
  const schemaPath = writeTestSchema('valid.sql', VALID_SCHEMA);
  const validator = new SchemaValidator({ db: 'postgres' });
  const result = validator.validateSchema(schemaPath);
  assertTrue(result, 'Valid schema should pass');
  assertEquals(validator.violations.length, 0, 'Should have no violations');
});

runner.test('Detects missing primary key', async () => {
  const schemaPath = writeTestSchema('no_pk.sql', INVALID_SCHEMA_NO_PK);
  const validator = new SchemaValidator({ db: 'postgres' });
  validator.validateSchema(schemaPath);

  const hasPKViolation = validator.violations.some(v =>
    v.issue.includes('Missing primary key')
  );
  assertTrue(hasPKViolation, 'Should detect missing primary key');
});

runner.test('Detects FLOAT/DOUBLE for money columns', async () => {
  const schemaPath = writeTestSchema('float_money.sql', INVALID_SCHEMA_FLOAT_MONEY);
  const validator = new SchemaValidator({ db: 'postgres' });
  validator.validateSchema(schemaPath);

  const hasFloatWarning = validator.warnings.some(w =>
    w.issue.includes('FLOAT') || w.issue.includes('DOUBLE')
  );
  assertTrue(hasFloatWarning, 'Should warn about FLOAT/DOUBLE for money');
});

runner.test('Detects missing foreign key index', async () => {
  const schemaPath = writeTestSchema('missing_fk_index.sql', INVALID_SCHEMA_MISSING_FK_INDEX);
  const validator = new SchemaValidator({ db: 'postgres' });
  validator.validateSchema(schemaPath);

  const hasFKWarning = validator.warnings.some(w =>
    w.issue.includes('Foreign key') && w.issue.includes('index')
  );
  assertTrue(hasFKWarning, 'Should warn about missing FK index');
});

runner.test('Detects bad naming conventions', async () => {
  const schemaPath = writeTestSchema('bad_naming.sql', INVALID_SCHEMA_BAD_NAMING);
  const validator = new SchemaValidator({ db: 'postgres' });
  validator.validateSchema(schemaPath);

  const hasNamingWarning = validator.warnings.some(w =>
    w.issue.includes('not snake_case')
  );
  assertTrue(hasNamingWarning, 'Should warn about naming conventions');
});

runner.test('Suggests timestamps for audit trail', async () => {
  const schemaWithoutTimestamps = `
    CREATE TABLE products (
      id BIGSERIAL PRIMARY KEY,
      name VARCHAR(255) NOT NULL,
      price DECIMAL(10, 2)
    );
  `;
  const schemaPath = writeTestSchema('no_timestamps.sql', schemaWithoutTimestamps);
  const validator = new SchemaValidator({ db: 'postgres', strict: true });
  validator.validateSchema(schemaPath);

  const hasTimestampInfo = validator.info.some(i =>
    i.issue.includes('created_at') || i.issue.includes('timestamp')
  );
  assertTrue(hasTimestampInfo, 'Should suggest timestamps in strict mode');
});

runner.test('Column parsing works correctly', async () => {
  const validator = new SchemaValidator();
  const columnsText = `
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    balance DECIMAL(10, 2) DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT true
  `;

  const columns = validator.parseColumns(columnsText);

  assertEquals(columns.length, 4, 'Should parse 4 columns');
  assertTrue(columns[0].isPrimaryKey, 'First column should be primary key');
  assertTrue(columns[1].isNotNull, 'Email should be NOT NULL');
  assertTrue(columns[1].isUnique, 'Email should be UNIQUE');
  assertTrue(columns[2].hasDefault, 'Balance should have DEFAULT');
});

runner.test('Foreign key parsing works correctly', async () => {
  const validator = new SchemaValidator();
  const columnsText = `
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    product_id BIGINT,
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (product_id) REFERENCES products (id)
  `;

  const constraints = validator.parseConstraints(columnsText);

  assertEquals(constraints.foreignKeys.length, 2, 'Should parse 2 foreign keys');
  assertEquals(constraints.foreignKeys[0].column, 'user_id', 'First FK should be user_id');
  assertEquals(constraints.foreignKeys[0].referencedTable, 'users', 'Should reference users table');
});

runner.test('Strict mode shows info messages', async () => {
  const schemaPath = writeTestSchema('strict_test.sql', VALID_SCHEMA);
  const strictValidator = new SchemaValidator({ db: 'postgres', strict: true });
  const normalValidator = new SchemaValidator({ db: 'postgres', strict: false });

  strictValidator.validateSchema(schemaPath);
  normalValidator.validateSchema(schemaPath);

  // Strict mode should report more findings
  const strictTotal = strictValidator.violations.length +
                      strictValidator.warnings.length +
                      strictValidator.info.length;

  const normalTotal = normalValidator.violations.length +
                      normalValidator.warnings.length;

  assertTrue(strictTotal >= normalTotal, 'Strict mode should show more findings');
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
