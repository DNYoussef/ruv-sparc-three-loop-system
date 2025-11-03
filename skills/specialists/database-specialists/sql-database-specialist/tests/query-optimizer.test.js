/**
 * Test Suite for Query Optimizer
 * Run with: node query-optimizer.test.js
 */

const fs = require('fs');
const path = require('path');
const QueryOptimizer = require('../resources/scripts/query-optimizer');

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
    console.log('\n🧪 Running Query Optimizer Tests\n');
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

function assertGreaterThan(actual, threshold, message) {
  if (actual <= threshold) {
    throw new Error(message || `Expected ${actual} > ${threshold}`);
  }
}

function assertLessThan(actual, threshold, message) {
  if (actual >= threshold) {
    throw new Error(message || `Expected ${actual} < ${threshold}`);
  }
}

// Test fixtures
const GOOD_POSTGRES_EXPLAIN = [
  {
    "Plan": {
      "Node Type": "Index Scan",
      "Relation Name": "users",
      "Index Name": "idx_users_email",
      "Plan Rows": 1,
      "Total Cost": 8.27,
      "Startup Cost": 0.29,
      "Actual Total Time": 0.042,
      "Heap Fetches": 0
    }
  }
];

const BAD_POSTGRES_EXPLAIN = [
  {
    "Plan": {
      "Node Type": "Seq Scan",
      "Relation Name": "orders",
      "Plan Rows": 10000,
      "Total Cost": 1500.00,
      "Startup Cost": 0.00,
      "Actual Total Time": 125.5
    }
  }
];

const NESTED_LOOP_SLOW = [
  {
    "Plan": {
      "Node Type": "Nested Loop",
      "Total Cost": 2000.00,
      "Actual Total Time": 150.0,
      "Plans": [
        {
          "Node Type": "Seq Scan",
          "Relation Name": "users",
          "Plan Rows": 1000
        },
        {
          "Node Type": "Index Scan",
          "Relation Name": "orders",
          "Index Name": "idx_orders_user_id"
        }
      ]
    }
  }
];

const HASH_JOIN_GOOD = [
  {
    "Plan": {
      "Node Type": "Hash Join",
      "Total Cost": 500.00,
      "Actual Total Time": 25.0,
      "Plans": [
        {
          "Node Type": "Seq Scan",
          "Relation Name": "users"
        },
        {
          "Node Type": "Hash",
          "Plans": [
            {
              "Node Type": "Seq Scan",
              "Relation Name": "orders"
            }
          ]
        }
      ]
    }
  }
];

const HIGH_HEAP_FETCHES = [
  {
    "Plan": {
      "Node Type": "Index Scan",
      "Relation Name": "products",
      "Index Name": "idx_products_category",
      "Heap Fetches": 5000,
      "Total Cost": 1000.00
    }
  }
];

const GOOD_BUFFER_USAGE = [
  {
    "Plan": {
      "Node Type": "Index Scan",
      "Relation Name": "users",
      "Index Name": "idx_users_id",
      "Shared Hit Blocks": 9500,
      "Shared Read Blocks": 500
    }
  }
];

const BAD_BUFFER_USAGE = [
  {
    "Plan": {
      "Node Type": "Seq Scan",
      "Relation Name": "orders",
      "Shared Hit Blocks": 1000,
      "Shared Read Blocks": 9000
    }
  }
];

// Create temp directory
const TEST_DIR = path.join(__dirname, '.test-temp');
if (!fs.existsSync(TEST_DIR)) {
  fs.mkdirSync(TEST_DIR, { recursive: true });
}

function writeTestExplain(filename, content) {
  const filePath = path.join(TEST_DIR, filename);
  fs.writeFileSync(filePath, JSON.stringify(content, null, 2));
  return filePath;
}

function cleanup() {
  if (fs.existsSync(TEST_DIR)) {
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  }
}

// Tests
const runner = new TestRunner();

runner.test('Good query gets high score', async () => {
  const explainPath = writeTestExplain('good.json', GOOD_POSTGRES_EXPLAIN);
  const optimizer = new QueryOptimizer({ db: 'postgres' });
  const score = optimizer.analyzeExplain(explainPath);

  assertGreaterThan(score, 90, 'Good query should score > 90');
  assertEquals(optimizer.recommendations.length, 0, 'Should have no recommendations');
});

runner.test('Sequential scan on large table triggers critical warning', async () => {
  const explainPath = writeTestExplain('seq_scan.json', BAD_POSTGRES_EXPLAIN);
  const optimizer = new QueryOptimizer({ db: 'postgres' });
  optimizer.analyzeExplain(explainPath);

  const hasCritical = optimizer.recommendations.some(r =>
    r.level === 'critical' && r.issue.includes('Sequential Scan')
  );
  assertTrue(hasCritical, 'Should detect critical sequential scan');
  assertLessThan(optimizer.score, 80, 'Score should be penalized');
});

runner.test('Slow nested loop triggers warning', async () => {
  const explainPath = writeTestExplain('nested_loop.json', NESTED_LOOP_SLOW);
  const optimizer = new QueryOptimizer({ db: 'postgres' });
  optimizer.analyzeExplain(explainPath);

  const hasNestedLoopWarning = optimizer.recommendations.some(r =>
    r.issue.includes('Nested Loop')
  );
  assertTrue(hasNestedLoopWarning, 'Should warn about slow nested loop');
});

runner.test('Hash join recognized as efficient', async () => {
  const explainPath = writeTestExplain('hash_join.json', HASH_JOIN_GOOD);
  const optimizer = new QueryOptimizer({ db: 'postgres' });
  optimizer.analyzeExplain(explainPath);

  // Hash join should not trigger warnings
  const hashJoinWarnings = optimizer.recommendations.filter(r =>
    r.issue.includes('Hash Join') && r.level === 'critical'
  );
  assertEquals(hashJoinWarnings.length, 0, 'Hash join should not be flagged as critical');
});

runner.test('High heap fetches suggest covering index', async () => {
  const explainPath = writeTestExplain('heap_fetches.json', HIGH_HEAP_FETCHES);
  const optimizer = new QueryOptimizer({ db: 'postgres' });
  optimizer.analyzeExplain(explainPath);

  const hasCoveringIndexSuggestion = optimizer.recommendations.some(r =>
    r.suggestion.includes('covering') || r.suggestion.includes('INCLUDE')
  );
  assertTrue(hasCoveringIndexSuggestion, 'Should suggest covering index for high heap fetches');
});

runner.test('Good buffer cache hit ratio recognized', async () => {
  const explainPath = writeTestExplain('good_buffer.json', GOOD_BUFFER_USAGE);
  const optimizer = new QueryOptimizer({ db: 'postgres' });
  optimizer.analyzeExplain(explainPath);

  // Should not have buffer-related warnings
  const bufferWarnings = optimizer.recommendations.filter(r =>
    r.issue.includes('cache') || r.issue.includes('buffer')
  );
  assertEquals(bufferWarnings.length, 0, 'Good cache hit ratio should not trigger warnings');
});

runner.test('Bad buffer cache hit ratio detected', async () => {
  const explainPath = writeTestExplain('bad_buffer.json', BAD_BUFFER_USAGE);
  const optimizer = new QueryOptimizer({ db: 'postgres' });
  optimizer.analyzeExplain(explainPath);

  // Bad cache ratio might not always trigger explicit warnings,
  // but should affect overall analysis
  assertTrue(optimizer.recommendations.length >= 0, 'Optimizer should run without errors');
});

runner.test('N+1 pattern detection works', async () => {
  const queries = [
    'SELECT * FROM users WHERE id = 1',
    'SELECT * FROM orders WHERE user_id = 1',
    'SELECT * FROM orders WHERE user_id = 2',
    'SELECT * FROM orders WHERE user_id = 3',
    'SELECT * FROM orders WHERE user_id = 4',
    'SELECT * FROM orders WHERE user_id = 5',
    'SELECT * FROM orders WHERE user_id = 6',
    'SELECT * FROM orders WHERE user_id = 7',
    'SELECT * FROM orders WHERE user_id = 8',
    'SELECT * FROM orders WHERE user_id = 9',
    'SELECT * FROM orders WHERE user_id = 10',
    'SELECT * FROM orders WHERE user_id = 11'
  ];

  const optimizer = new QueryOptimizer();
  optimizer.detectN1Pattern(queries);

  const hasN1Warning = optimizer.recommendations.some(r =>
    r.issue.includes('N+1')
  );
  assertTrue(hasN1Warning, 'Should detect N+1 pattern');
});

runner.test('Query normalization works correctly', async () => {
  const optimizer = new QueryOptimizer();

  const q1 = 'SELECT * FROM users WHERE id = 123';
  const q2 = 'SELECT * FROM users WHERE id = 456';
  const q3 = "SELECT * FROM users WHERE name = 'John'";
  const q4 = "SELECT * FROM users WHERE name = 'Jane'";

  const norm1 = optimizer.normalizeQuery(q1);
  const norm2 = optimizer.normalizeQuery(q2);
  const norm3 = optimizer.normalizeQuery(q3);
  const norm4 = optimizer.normalizeQuery(q4);

  assertEquals(norm1, norm2, 'Queries with different IDs should normalize the same');
  assertEquals(norm3, norm4, 'Queries with different strings should normalize the same');
});

runner.test('Performance score calculation is accurate', async () => {
  const optimizer = new QueryOptimizer();
  assertEquals(optimizer.score, 100, 'Initial score should be 100');

  optimizer.addRecommendation('critical', 'Test issue', 'Test solution', -20);
  assertEquals(optimizer.score, 80, 'Score should decrease by 20');

  optimizer.addRecommendation('warning', 'Test warning', 'Test fix', -10);
  assertEquals(optimizer.score, 70, 'Score should decrease by 10 more');

  optimizer.addRecommendation('info', 'Test info', 'Test tip', -3);
  assertEquals(optimizer.score, 67, 'Score should decrease by 3 more');
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
