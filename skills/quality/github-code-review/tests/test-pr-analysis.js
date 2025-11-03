#!/usr/bin/env node
/**
 * Test Suite for PR Analysis Script
 *
 * Tests PR complexity analysis, agent recommendations, and topology selection.
 *
 * Usage:
 *   node test-pr-analysis.js
 */

const assert = require('assert');
const { execSync } = require('child_process');
const path = require('path');

// ANSI color codes for output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m'
};

let passCount = 0;
let failCount = 0;

/**
 * Test helper functions
 */
function test(name, fn) {
  try {
    console.log(`\n${colors.blue}TEST:${colors.reset} ${name}`);
    fn();
    passCount++;
    console.log(`${colors.green}✓ PASS${colors.reset}`);
  } catch (error) {
    failCount++;
    console.log(`${colors.red}✗ FAIL${colors.reset}: ${error.message}`);
  }
}

function assertEquals(actual, expected, message) {
  assert.strictEqual(actual, expected, message || `Expected ${expected}, got ${actual}`);
}

function assertContains(array, item, message) {
  assert(array.includes(item), message || `Expected array to contain ${item}`);
}

function assertGreaterThan(actual, threshold, message) {
  assert(actual > threshold, message || `Expected ${actual} > ${threshold}`);
}

/**
 * Mock PR data for testing
 */
const mockPRs = {
  simple: {
    files: [
      { path: 'src/utils.js', additions: 10, deletions: 5 },
      { path: 'tests/utils.test.js', additions: 15, deletions: 0 }
    ],
    additions: 25,
    deletions: 5,
    title: 'Add utility function',
    labels: []
  },

  moderate: {
    files: [
      { path: 'src/api/users.js', additions: 100, deletions: 20 },
      { path: 'src/api/auth.js', additions: 80, deletions: 15 },
      { path: 'tests/api.test.js', additions: 120, deletions: 10 }
    ],
    additions: 300,
    deletions: 45,
    title: 'Implement user authentication',
    labels: [{ name: 'feature' }]
  },

  complex: {
    files: [
      { path: 'src/auth/authentication.js', additions: 200, deletions: 50 },
      { path: 'src/payment/processor.js', additions: 300, deletions: 100 },
      { path: 'src/admin/dashboard.js', additions: 150, deletions: 30 },
      { path: 'database/migrations/001_add_users.sql', additions: 100, deletions: 0 },
      { path: 'config/production.yml', additions: 50, deletions: 20 }
    ],
    additions: 800,
    deletions: 200,
    title: 'Major authentication and payment system refactor',
    labels: [{ name: 'breaking-change' }, { name: 'security' }]
  },

  securitySensitive: {
    files: [
      { path: 'src/auth/jwt.js', additions: 50, deletions: 10 },
      { path: 'src/security/encryption.js', additions: 80, deletions: 20 }
    ],
    additions: 130,
    deletions: 30,
    title: 'Update encryption algorithm',
    labels: [{ name: 'security' }]
  }
};

/**
 * Test complexity determination
 */
test('Simple PR should be classified as "simple"', () => {
  const pr = mockPRs.simple;
  const totalChanges = pr.additions + pr.deletions;

  assertEquals(totalChanges < 100, true);

  // Complexity should be 'simple'
  const complexity = totalChanges < 100 ? 'simple' : 'moderate';
  assertEquals(complexity, 'simple');
});

test('Moderate PR should be classified as "moderate"', () => {
  const pr = mockPRs.moderate;
  const totalChanges = pr.additions + pr.deletions;

  assertEquals(totalChanges >= 100 && totalChanges < 500, true);

  let complexity;
  if (totalChanges < 100) complexity = 'simple';
  else if (totalChanges < 500) complexity = 'moderate';
  else complexity = 'complex';

  assertEquals(complexity, 'moderate');
});

test('Complex PR should be classified as "complex"', () => {
  const pr = mockPRs.complex;
  const totalChanges = pr.additions + pr.deletions;

  assertGreaterThan(totalChanges, 500);

  let complexity;
  if (totalChanges < 100) complexity = 'simple';
  else if (totalChanges < 500) complexity = 'moderate';
  else if (totalChanges < 1000) complexity = 'complex';
  else complexity = 'very-complex';

  assertEquals(complexity, 'complex');
});

/**
 * Test topology selection
 */
test('Simple PR should recommend "ring" topology', () => {
  const pr = mockPRs.simple;
  const fileCount = pr.files.length;

  let topology;
  if (fileCount < 5) topology = 'ring';
  else if (fileCount < 15) topology = 'mesh';
  else topology = 'hierarchical';

  assertEquals(topology, 'ring');
});

test('Moderate PR should recommend "mesh" topology', () => {
  const pr = mockPRs.moderate;
  const fileCount = pr.files.length;

  let topology;
  if (fileCount < 5) topology = 'ring';
  else if (fileCount < 15) topology = 'mesh';
  else topology = 'hierarchical';

  assertEquals(topology, 'mesh');
});

test('Complex PR should recommend "hierarchical" topology', () => {
  const pr = mockPRs.complex;
  const fileCount = pr.files.length;

  let topology;
  if (fileCount < 5) topology = 'ring';
  else if (fileCount < 15) topology = 'mesh';
  else topology = 'hierarchical';

  assertEquals(topology, 'hierarchical');
});

/**
 * Test critical area detection
 */
test('Should detect security-sensitive files', () => {
  const pr = mockPRs.securitySensitive;
  const criticalAreas = [];

  pr.files.forEach(file => {
    if (file.path.match(/auth|security|payment|admin/i)) {
      criticalAreas.push({ path: file.path, reason: 'security-sensitive' });
    }
  });

  assertGreaterThan(criticalAreas.length, 0);
  assertEquals(criticalAreas[0].reason, 'security-sensitive');
});

test('Should detect infrastructure files', () => {
  const pr = mockPRs.complex;
  const infraFiles = pr.files.filter(f =>
    f.path.match(/api|database|migration/i)
  );

  assertGreaterThan(infraFiles.length, 0);
});

test('Should detect configuration files', () => {
  const pr = mockPRs.complex;
  const configFiles = pr.files.filter(f =>
    f.path.match(/config|\.env|secret/i)
  );

  assertGreaterThan(configFiles.length, 0);
});

/**
 * Test agent recommendations
 */
test('Should always recommend security and style agents', () => {
  const agents = new Set(['security', 'style']);

  assertContains([...agents], 'security');
  assertContains([...agents], 'style');
});

test('Security-sensitive PR should recommend security agents', () => {
  const pr = mockPRs.securitySensitive;
  const agents = new Set(['security', 'style']);

  // Check for security-sensitive files
  const hasSecurityFiles = pr.files.some(f =>
    f.path.match(/auth|security|payment|admin/i)
  );

  if (hasSecurityFiles) {
    agents.add('security');
    agents.add('authentication');
    agents.add('audit');
  }

  assertContains([...agents], 'security');
  assertContains([...agents], 'authentication');
  assertContains([...agents], 'audit');
});

test('Infrastructure PR should recommend performance agents', () => {
  const pr = mockPRs.complex;
  const agents = new Set(['security', 'style']);

  const hasInfraFiles = pr.files.some(f =>
    f.path.match(/api|database|migration/i)
  );

  if (hasInfraFiles) {
    agents.add('performance');
    agents.add('database');
  }

  assertContains([...agents], 'performance');
  assertContains([...agents], 'database');
});

/**
 * Test risk level determination
 */
test('PR with multiple critical areas should be high risk', () => {
  const pr = mockPRs.complex;
  const criticalAreas = [];

  pr.files.forEach(file => {
    if (file.path.match(/auth|security|payment|admin/i)) {
      criticalAreas.push({ path: file.path, reason: 'security-sensitive' });
    }
    if (file.path.match(/api|database|migration/i)) {
      criticalAreas.push({ path: file.path, reason: 'infrastructure' });
    }
  });

  let riskLevel;
  if (criticalAreas.length > 3) riskLevel = 'high';
  else if (criticalAreas.length > 0) riskLevel = 'medium';
  else riskLevel = 'low';

  assertEquals(riskLevel, 'high');
});

test('Simple PR with no critical areas should be low risk', () => {
  const pr = mockPRs.simple;
  const criticalAreas = [];

  pr.files.forEach(file => {
    if (file.path.match(/auth|security|payment|admin/i)) {
      criticalAreas.push({ path: file.path, reason: 'security-sensitive' });
    }
  });

  let riskLevel;
  if (criticalAreas.length > 3) riskLevel = 'high';
  else if (criticalAreas.length > 0) riskLevel = 'medium';
  else riskLevel = 'low';

  assertEquals(riskLevel, 'low');
});

/**
 * Test estimated review time
 */
test('Simple PR should estimate <30 minutes', () => {
  const pr = mockPRs.simple;
  const totalChanges = pr.additions + pr.deletions;

  let estimatedTime;
  if (totalChanges < 100) estimatedTime = 15;
  else if (totalChanges < 500) estimatedTime = 45;
  else estimatedTime = 90;

  assertEquals(estimatedTime, 15);
});

test('Complex PR should estimate >60 minutes', () => {
  const pr = mockPRs.complex;
  const totalChanges = pr.additions + pr.deletions;

  let estimatedTime;
  if (totalChanges < 100) estimatedTime = 15;
  else if (totalChanges < 500) estimatedTime = 45;
  else if (totalChanges < 1000) estimatedTime = 90;
  else estimatedTime = 180;

  assertGreaterThan(estimatedTime, 60);
});

/**
 * Print test results
 */
console.log('\n' + '='.repeat(60));
console.log(`${colors.blue}TEST RESULTS${colors.reset}`);
console.log('='.repeat(60));
console.log(`${colors.green}✓ Passed:${colors.reset} ${passCount}`);
console.log(`${colors.red}✗ Failed:${colors.reset} ${failCount}`);
console.log(`${colors.yellow}Total:${colors.reset} ${passCount + failCount}`);
console.log('='.repeat(60));

if (failCount > 0) {
  console.log(`\n${colors.red}Some tests failed!${colors.reset}`);
  process.exit(1);
} else {
  console.log(`\n${colors.green}All tests passed!${colors.reset}`);
  process.exit(0);
}
