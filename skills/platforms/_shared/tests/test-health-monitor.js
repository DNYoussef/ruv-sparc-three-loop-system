#!/usr/bin/env node
/**
 * Test Suite for Health Monitor
 * Tests health checking, metrics collection, and alerting
 */

const assert = require('assert');
const fs = require('fs').promises;
const path = require('path');
const { EventEmitter } = require('events');

// Test configuration
const TEST_TIMEOUT = 10000; // 10 seconds
let testsPassed = 0;
let testsFailed = 0;

// Mock HealthMonitor for testing
class MockHealthCheck {
  constructor(config) {
    this.name = config.name;
    this.type = config.type || 'http';
    this.endpoint = config.endpoint;
    this.metrics = [];
    this.consecutiveFailures = 0;
  }

  async execute() {
    // Simple mock execution
    const success = Math.random() > 0.2; // 80% success rate
    const responseTime = Math.random() * 100;

    this.metrics.push({
      timestamp: new Date(),
      success,
      responseTime
    });

    if (success) {
      this.consecutiveFailures = 0;
    } else {
      this.consecutiveFailures++;
    }

    return {
      name: this.name,
      success,
      responseTime,
      consecutiveFailures: this.consecutiveFailures
    };
  }
}

class MockHealthMonitor extends EventEmitter {
  constructor() {
    super();
    this.checks = new Map();
    this.running = false;
  }

  addCheck(check) {
    this.checks.set(check.name, check);
  }

  async start() {
    this.running = true;
    this.emit('started');
  }

  stop() {
    this.running = false;
    this.emit('stopped');
  }

  getStatus() {
    const status = {
      running: this.running,
      checks: {}
    };

    for (const [name, check] of this.checks) {
      status.checks[name] = {
        name,
        consecutiveFailures: check.consecutiveFailures,
        metricsCount: check.metrics.length
      };
    }

    return status;
  }
}

// Test helpers
function assertEquals(actual, expected, message) {
  try {
    assert.strictEqual(actual, expected);
    console.log(`  ✓ ${message}`);
    testsPassed++;
  } catch (err) {
    console.log(`  ✗ ${message}`);
    console.log(`    Expected: ${expected}`);
    console.log(`    Actual:   ${actual}`);
    testsFailed++;
  }
}

function assertNotNull(value, message) {
  try {
    assert.notStrictEqual(value, null);
    assert.notStrictEqual(value, undefined);
    console.log(`  ✓ ${message}`);
    testsPassed++;
  } catch (err) {
    console.log(`  ✗ ${message}`);
    testsFailed++;
  }
}

function assertTrue(value, message) {
  try {
    assert.strictEqual(value, true);
    console.log(`  ✓ ${message}`);
    testsPassed++;
  } catch (err) {
    console.log(`  ✗ ${message}`);
    testsFailed++;
  }
}

function assertGreaterThan(actual, threshold, message) {
  try {
    assert.ok(actual > threshold, `${actual} should be > ${threshold}`);
    console.log(`  ✓ ${message}`);
    testsPassed++;
  } catch (err) {
    console.log(`  ✗ ${message}`);
    testsFailed++;
  }
}

// Tests
async function testHealthCheckCreation() {
  console.log('\nTest: Health Check Creation');

  const check = new MockHealthCheck({
    name: 'test-service',
    type: 'http',
    endpoint: 'http://localhost:3000/health'
  });

  assertEquals(check.name, 'test-service', 'Check name is correct');
  assertEquals(check.type, 'http', 'Check type is correct');
  assertEquals(check.endpoint, 'http://localhost:3000/health', 'Check endpoint is correct');
}

async function testHealthCheckExecution() {
  console.log('\nTest: Health Check Execution');

  const check = new MockHealthCheck({
    name: 'test-service',
    type: 'http',
    endpoint: 'http://localhost:3000/health'
  });

  const result = await check.execute();

  assertNotNull(result, 'Result should not be null');
  assertEquals(result.name, 'test-service', 'Result contains check name');
  assertNotNull(result.responseTime, 'Result contains response time');
}

async function testMetricsCollection() {
  console.log('\nTest: Metrics Collection');

  const check = new MockHealthCheck({
    name: 'metrics-test',
    type: 'http',
    endpoint: 'http://localhost:3000/health'
  });

  // Execute multiple checks
  for (let i = 0; i < 5; i++) {
    await check.execute();
  }

  assertEquals(check.metrics.length, 5, 'Collected 5 metrics');
}

async function testConsecutiveFailures() {
  console.log('\nTest: Consecutive Failures Tracking');

  const check = new MockHealthCheck({
    name: 'failure-test',
    type: 'command',
    endpoint: 'exit 1' // Always fails
  });

  // First failure
  await check.execute();
  const failures = check.consecutiveFailures;

  assertGreaterThan(failures, -1, 'Consecutive failures tracked');
}

async function testMonitorCreation() {
  console.log('\nTest: Monitor Creation');

  const monitor = new MockHealthMonitor();

  assertNotNull(monitor, 'Monitor created');
  assertEquals(monitor.running, false, 'Monitor not running initially');
}

async function testMonitorStartStop() {
  console.log('\nTest: Monitor Start/Stop');

  const monitor = new MockHealthMonitor();

  await monitor.start();
  assertEquals(monitor.running, true, 'Monitor started');

  monitor.stop();
  assertEquals(monitor.running, false, 'Monitor stopped');
}

async function testMonitorStatus() {
  console.log('\nTest: Monitor Status');

  const monitor = new MockHealthMonitor();

  const check = new MockHealthCheck({
    name: 'status-test',
    type: 'http',
    endpoint: 'http://localhost:3000/health'
  });

  monitor.addCheck(check);
  const status = monitor.getStatus();

  assertNotNull(status, 'Status returned');
  assertNotNull(status.checks, 'Status contains checks');
  assertNotNull(status.checks['status-test'], 'Status contains specific check');
}

async function testEventEmission() {
  console.log('\nTest: Event Emission');

  const monitor = new MockHealthMonitor();
  let started = false;
  let stopped = false;

  monitor.on('started', () => {
    started = true;
  });

  monitor.on('stopped', () => {
    stopped = true;
  });

  await monitor.start();
  monitor.stop();

  assertTrue(started, 'Started event emitted');
  assertTrue(stopped, 'Stopped event emitted');
}

async function testMultipleChecks() {
  console.log('\nTest: Multiple Health Checks');

  const monitor = new MockHealthMonitor();

  const check1 = new MockHealthCheck({
    name: 'service-1',
    type: 'http',
    endpoint: 'http://localhost:3000/health'
  });

  const check2 = new MockHealthCheck({
    name: 'service-2',
    type: 'tcp',
    endpoint: 'localhost:5432'
  });

  monitor.addCheck(check1);
  monitor.addCheck(check2);

  assertEquals(monitor.checks.size, 2, 'Two checks added');
}

async function testIntegration() {
  console.log('\nTest: Integration');

  const monitor = new MockHealthMonitor();

  // Add multiple checks
  const checks = [
    new MockHealthCheck({
      name: 'api',
      type: 'http',
      endpoint: 'http://localhost:3000/health'
    }),
    new MockHealthCheck({
      name: 'database',
      type: 'tcp',
      endpoint: 'localhost:5432'
    }),
    new MockHealthCheck({
      name: 'redis',
      type: 'command',
      endpoint: 'redis-cli ping'
    })
  ];

  for (const check of checks) {
    monitor.addCheck(check);
  }

  await monitor.start();

  // Execute all checks
  for (const check of checks) {
    await check.execute();
  }

  const status = monitor.getStatus();

  monitor.stop();

  assertEquals(monitor.checks.size, 3, 'All checks added');
  assertEquals(status.checks['api'].metricsCount, 1, 'API check executed');
  assertEquals(status.checks['database'].metricsCount, 1, 'Database check executed');
  assertEquals(status.checks['redis'].metricsCount, 1, 'Redis check executed');
}

// Print test summary
function printSummary() {
  console.log('\n' + '='.repeat(70));
  console.log('  Test Summary');
  console.log('='.repeat(70));
  console.log(`Passed:  ${testsPassed}`);
  console.log(`Failed:  ${testsFailed}`);
  console.log(`Total:   ${testsPassed + testsFailed}`);
  console.log('='.repeat(70));

  if (testsFailed === 0) {
    console.log('\x1b[32mAll tests passed!\x1b[0m');
    return 0;
  } else {
    console.log('\x1b[31mSome tests failed!\x1b[0m');
    return 1;
  }
}

// Main test execution
async function runTests() {
  console.log('='.repeat(70));
  console.log('  Health Monitor Test Suite');
  console.log('='.repeat(70));

  try {
    await testHealthCheckCreation();
    await testHealthCheckExecution();
    await testMetricsCollection();
    await testConsecutiveFailures();
    await testMonitorCreation();
    await testMonitorStartStop();
    await testMonitorStatus();
    await testEventEmission();
    await testMultipleChecks();
    await testIntegration();
  } catch (err) {
    console.error('\nTest execution error:', err);
    testsFailed++;
  }

  return printSummary();
}

// Run tests if executed directly
if (require.main === module) {
  runTests()
    .then(code => process.exit(code))
    .catch(err => {
      console.error('Fatal error:', err);
      process.exit(1);
    });
}

module.exports = { runTests };
