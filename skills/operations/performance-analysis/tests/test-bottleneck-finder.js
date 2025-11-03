#!/usr/bin/env node
/**
 * Test Suite for Bottleneck Finder
 * Comprehensive tests for bottleneck detection and analysis
 */

const assert = require('assert');
const BottleneckFinder = require('../resources/bottleneck-finder');

/**
 * Test suite runner
 */
class TestRunner {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
  }

  test(name, fn) {
    this.tests.push({ name, fn });
  }

  async run() {
    console.log('\n='.repeat(70));
    console.log('Running Bottleneck Finder Tests');
    console.log('='.repeat(70) + '\n');

    for (const { name, fn } of this.tests) {
      try {
        await fn();
        this.passed++;
        console.log(`✓ ${name}`);
      } catch (error) {
        this.failed++;
        console.error(`✗ ${name}`);
        console.error(`  ${error.message}`);
      }
    }

    console.log('\n' + '='.repeat(70));
    console.log(`Results: ${this.passed} passed, ${this.failed} failed`);
    console.log('='.repeat(70) + '\n');

    return this.failed === 0;
  }
}

// Create test runner
const runner = new TestRunner();

// Test: Initialization
runner.test('Initialization with default thresholds', () => {
  const finder = new BottleneckFinder();
  assert.strictEqual(finder.thresholds.communicationDelay, 2000);
  assert.strictEqual(finder.thresholds.taskTime, 60000);
  assert.strictEqual(finder.thresholds.memoryUsage, 0.8);
  assert.strictEqual(finder.thresholds.utilization, 0.5);
});

// Test: Custom configuration
runner.test('Initialization with custom thresholds', () => {
  const finder = new BottleneckFinder({
    communicationDelay: 1500,
    taskTime: 45000,
    memoryUsage: 0.7
  });
  assert.strictEqual(finder.thresholds.communicationDelay, 1500);
  assert.strictEqual(finder.thresholds.taskTime, 45000);
  assert.strictEqual(finder.thresholds.memoryUsage, 0.7);
});

// Test: Communication bottleneck detection
runner.test('Detect communication bottlenecks', () => {
  const finder = new BottleneckFinder();
  const metrics = {
    agents: [
      {
        id: 'agent-1',
        avgResponseTime: 2500, // Exceeds threshold
        avgTaskTime: 45000,
        utilization: 0.75,
        memoryUsage: 0.60
      }
    ],
    messageQueue: {
      depth: 5
    }
  };

  const bottlenecks = finder.analyze(metrics);

  assert.ok(bottlenecks.critical.length > 0);
  const commBottleneck = bottlenecks.critical.find(
    b => b.type === 'communication' && b.subtype === 'message_delay'
  );
  assert.ok(commBottleneck);
  assert.strictEqual(commBottleneck.agent, 'agent-1');
});

// Test: Processing bottleneck detection
runner.test('Detect processing bottlenecks', () => {
  const finder = new BottleneckFinder();
  const metrics = {
    agents: [
      {
        id: 'agent-slow',
        avgResponseTime: 1200,
        avgTaskTime: 75000, // Exceeds threshold
        utilization: 0.35,  // Below threshold
        memoryUsage: 0.50
      }
    ]
  };

  const bottlenecks = finder.analyze(metrics);

  // Should detect both slow tasks and low utilization
  const slowTask = bottlenecks.critical.find(
    b => b.type === 'processing' && b.subtype === 'slow_tasks'
  );
  const lowUtil = bottlenecks.warning.find(
    b => b.type === 'processing' && b.subtype === 'low_utilization'
  );

  assert.ok(slowTask);
  assert.ok(lowUtil);
  assert.strictEqual(slowTask.agent, 'agent-slow');
});

// Test: Memory bottleneck detection
runner.test('Detect memory bottlenecks', () => {
  const finder = new BottleneckFinder();
  const metrics = {
    agents: [
      {
        id: 'agent-memory',
        avgResponseTime: 1200,
        avgTaskTime: 45000,
        utilization: 0.75,
        memoryUsage: 0.90 // Exceeds threshold
      }
    ],
    cache: {
      hitRate: 0.60 // Below threshold
    }
  };

  const bottlenecks = finder.analyze(metrics);

  const highMemory = bottlenecks.critical.find(
    b => b.type === 'memory' && b.subtype === 'high_usage'
  );
  const lowCache = bottlenecks.warning.find(
    b => b.type === 'memory' && b.subtype === 'low_cache_hits'
  );

  assert.ok(highMemory);
  assert.ok(lowCache);
});

// Test: Network bottleneck detection
runner.test('Detect network bottlenecks', () => {
  const finder = new BottleneckFinder();
  const metrics = {
    agents: [],
    network: {
      apiLatency: 1500,     // Exceeds threshold
      failedRequests: 8     // Exceeds threshold
    }
  };

  const bottlenecks = finder.analyze(metrics);

  const latency = bottlenecks.warning.find(
    b => b.type === 'network' && b.subtype === 'api_latency'
  );
  const failures = bottlenecks.critical.find(
    b => b.type === 'network' && b.subtype === 'failed_requests'
  );

  assert.ok(latency);
  assert.ok(failures);
});

// Test: Coordination bottleneck detection
runner.test('Detect coordination bottlenecks', () => {
  const finder = new BottleneckFinder();
  const metrics = {
    agents: [
      { id: 'a1', type: 'coder', avgResponseTime: 1000, avgTaskTime: 40000, utilization: 0.7, memoryUsage: 0.5 },
      { id: 'a2', type: 'coder', avgResponseTime: 1000, avgTaskTime: 40000, utilization: 0.7, memoryUsage: 0.5 },
      { id: 'a3', type: 'coder', avgResponseTime: 1000, avgTaskTime: 40000, utilization: 0.7, memoryUsage: 0.5 },
      { id: 'a4', type: 'coder', avgResponseTime: 1000, avgTaskTime: 40000, utilization: 0.7, memoryUsage: 0.5 },
      { id: 'a5', type: 'coder', avgResponseTime: 1000, avgTaskTime: 40000, utilization: 0.7, memoryUsage: 0.5 },
      { id: 'a6', type: 'coder', avgResponseTime: 1000, avgTaskTime: 40000, utilization: 0.7, memoryUsage: 0.5 },
      { id: 'r1', type: 'reviewer', avgResponseTime: 1000, avgTaskTime: 40000, utilization: 0.7, memoryUsage: 0.5 }
    ],
    swarm: {
      topology: 'mesh'
    }
  };

  const bottlenecks = finder.analyze(metrics);

  const imbalance = bottlenecks.warning.find(
    b => b.type === 'coordination' && b.subtype === 'imbalanced_agents'
  );

  assert.ok(imbalance);
});

// Test: Impact calculation
runner.test('Calculate impact correctly', () => {
  const finder = new BottleneckFinder();

  // Test impact with 50% over threshold
  const impact1 = finder.calculateImpact(3000, 2000);
  assert.strictEqual(impact1, 0.5);

  // Test impact with 100% over threshold
  const impact2 = finder.calculateImpact(4000, 2000);
  assert.strictEqual(impact2, 1.0);

  // Test impact capped at 1.0
  const impact3 = finder.calculateImpact(8000, 2000);
  assert.strictEqual(impact3, 1.0);
});

// Test: Multiple bottleneck categories
runner.test('Handle multiple bottleneck categories', () => {
  const finder = new BottleneckFinder();
  const metrics = {
    agents: [
      {
        id: 'problematic-agent',
        avgResponseTime: 2500,
        avgTaskTime: 75000,
        utilization: 0.35,
        memoryUsage: 0.90
      }
    ],
    messageQueue: { depth: 12 },
    cache: { hitRate: 0.60 },
    network: { apiLatency: 1500, failedRequests: 6 }
  };

  const bottlenecks = finder.analyze(metrics);

  // Should have bottlenecks in multiple categories
  assert.ok(bottlenecks.critical.length > 0);
  assert.ok(bottlenecks.warning.length > 0);

  // Verify different bottleneck types exist
  const types = new Set([
    ...bottlenecks.critical.map(b => b.type),
    ...bottlenecks.warning.map(b => b.type)
  ]);
  assert.ok(types.has('communication'));
  assert.ok(types.has('processing'));
  assert.ok(types.has('memory'));
  assert.ok(types.has('network'));
});

// Test: Report generation - text format
runner.test('Generate text report', () => {
  const finder = new BottleneckFinder();
  const metrics = {
    agents: [
      {
        id: 'agent-1',
        avgResponseTime: 2500,
        avgTaskTime: 45000,
        utilization: 0.75,
        memoryUsage: 0.60
      }
    ]
  };

  finder.analyze(metrics);
  const report = finder.generateReport('text');

  assert.ok(typeof report === 'string');
  assert.ok(report.includes('BOTTLENECK ANALYSIS REPORT'));
  assert.ok(report.includes('SUMMARY'));
  assert.ok(report.includes('Critical Issues'));
});

// Test: Report generation - JSON format
runner.test('Generate JSON report', () => {
  const finder = new BottleneckFinder();
  const metrics = {
    agents: [
      {
        id: 'agent-1',
        avgResponseTime: 2500,
        avgTaskTime: 45000,
        utilization: 0.75,
        memoryUsage: 0.60
      }
    ]
  };

  finder.analyze(metrics);
  const report = finder.generateReport('json');

  assert.ok(typeof report === 'string');

  const parsed = JSON.parse(report);
  assert.ok(parsed.timestamp);
  assert.ok(parsed.summary);
  assert.ok(parsed.bottlenecks);
  assert.ok(parsed.bottlenecks.critical);
  assert.ok(parsed.bottlenecks.warning);
  assert.ok(parsed.bottlenecks.info);
});

// Test: No bottlenecks scenario
runner.test('Handle scenario with no bottlenecks', () => {
  const finder = new BottleneckFinder();
  const metrics = {
    agents: [
      {
        id: 'healthy-agent',
        avgResponseTime: 1000,
        avgTaskTime: 45000,
        utilization: 0.75,
        memoryUsage: 0.60
      }
    ],
    messageQueue: { depth: 5 },
    cache: { hitRate: 0.85 },
    network: { apiLatency: 800, failedRequests: 2 }
  };

  const bottlenecks = finder.analyze(metrics);

  assert.strictEqual(bottlenecks.critical.length, 0);
  // May have some warnings, but should be minimal
});

// Test: Task backlog detection
runner.test('Detect task backlog', () => {
  const finder = new BottleneckFinder();
  const metrics = {
    agents: [],
    tasks: {
      pending: new Array(15).fill(1) // 15 pending tasks
    }
  };

  const bottlenecks = finder.analyze(metrics);

  const backlog = bottlenecks.critical.find(
    b => b.type === 'processing' && b.subtype === 'task_backlog'
  );

  assert.ok(backlog);
  assert.strictEqual(backlog.value, 15);
});

// Test: Recommendations included
runner.test('Include recommendations for bottlenecks', () => {
  const finder = new BottleneckFinder();
  const metrics = {
    agents: [
      {
        id: 'agent-1',
        avgResponseTime: 2500,
        avgTaskTime: 45000,
        utilization: 0.75,
        memoryUsage: 0.60
      }
    ]
  };

  const bottlenecks = finder.analyze(metrics);

  const commBottleneck = bottlenecks.critical.find(
    b => b.type === 'communication'
  );

  assert.ok(commBottleneck);
  assert.ok(commBottleneck.recommendation);
  assert.ok(typeof commBottleneck.recommendation === 'string');
  assert.ok(commBottleneck.recommendation.length > 0);
});

// Run all tests
runner.run().then(success => {
  process.exit(success ? 0 : 1);
});
