#!/usr/bin/env node
/**
 * Tests for Bottleneck Detector
 */

const assert = require('assert');
const fs = require('fs').promises;
const path = require('path');
const BottleneckDetector = require('../resources/bottleneck-detector');

describe('BottleneckDetector', function() {
  this.timeout(10000); // Increase timeout for monitoring tests

  let detector;
  let tempDir;

  beforeEach(async function() {
    tempDir = path.join(__dirname, `temp-${Date.now()}`);
    await fs.mkdir(tempDir, { recursive: true });

    detector = new BottleneckDetector({
      eventLoopThreshold: 50,
      memoryLeakThreshold: 5,
      slowQueryThreshold: 50,
      slowRequestThreshold: 500,
      sampleInterval: 100,
      outputDir: tempDir
    });
  });

  afterEach(async function() {
    if (detector && detector.monitoring) {
      detector.stop();
    }
    try {
      await fs.rm(tempDir, { recursive: true, force: true });
    } catch (err) {
      // Ignore cleanup errors
    }
  });

  describe('Initialization', function() {
    it('should create detector with default options', function() {
      const det = new BottleneckDetector();
      assert.ok(det);
      assert.strictEqual(det.monitoring, false);
    });

    it('should create detector with custom options', function() {
      assert.strictEqual(detector.options.eventLoopThreshold, 50);
      assert.strictEqual(detector.options.memoryLeakThreshold, 5);
    });

    it('should initialize empty metrics', function() {
      assert.ok(Array.isArray(detector.metrics.eventLoop));
      assert.ok(Array.isArray(detector.metrics.memory));
      assert.ok(Array.isArray(detector.metrics.bottlenecks));
    });
  });

  describe('Event Loop Monitoring', function() {
    it('should detect event loop lag', function(done) {
      detector.on('bottleneck', (bottleneck) => {
        if (bottleneck.type === 'event_loop_lag') {
          assert.ok(bottleneck.lag > detector.options.eventLoopThreshold);
          assert.ok(['high', 'medium'].includes(bottleneck.severity));
          detector.stop();
          done();
        }
      });

      detector.start();

      // Simulate event loop blocking
      setTimeout(() => {
        const start = Date.now();
        while (Date.now() - start < 100) {
          // Block event loop
        }
      }, 200);
    });

    it('should track event loop metrics', function(done) {
      detector.start();

      setTimeout(() => {
        assert.ok(detector.metrics.eventLoop.length > 0);
        const metric = detector.metrics.eventLoop[0];
        assert.ok(metric.timestamp);
        assert.ok(typeof metric.lag === 'number');
        detector.stop();
        done();
      }, 500);
    });
  });

  describe('Memory Monitoring', function() {
    it('should track memory usage', function(done) {
      detector.start();

      setTimeout(() => {
        assert.ok(detector.metrics.memory.length > 0);
        const metric = detector.metrics.memory[0];
        assert.ok(metric.heapUsed);
        assert.ok(metric.heapTotal);
        assert.ok(typeof metric.growth === 'number');
        detector.stop();
        done();
      }, 600);
    });

    it('should detect memory growth', function(done) {
      let leak = [];

      detector.on('bottleneck', (bottleneck) => {
        if (bottleneck.type === 'memory_leak') {
          assert.ok(bottleneck.growth > detector.options.memoryLeakThreshold);
          assert.strictEqual(bottleneck.severity, 'high');
          detector.stop();
          leak = null; // Clean up
          done();
        }
      });

      detector.start();

      // Simulate memory growth
      const interval = setInterval(() => {
        for (let i = 0; i < 100000; i++) {
          leak.push(new Array(100).fill(Math.random()));
        }
      }, 100);

      setTimeout(() => {
        clearInterval(interval);
        if (detector.monitoring) {
          detector.stop();
          leak = null;
          done();
        }
      }, 2000);
    });
  });

  describe('Query Tracking', function() {
    it('should track slow queries', function() {
      const query = 'SELECT * FROM users WHERE id = 1';
      const duration = 150;

      detector.trackQuery(query, duration);

      assert.strictEqual(detector.metrics.queries.length, 1);
      const metric = detector.metrics.queries[0];
      assert.strictEqual(metric.query, query);
      assert.strictEqual(metric.duration, duration);
    });

    it('should report slow query bottleneck', function(done) {
      detector.on('bottleneck', (bottleneck) => {
        if (bottleneck.type === 'slow_query') {
          assert.strictEqual(bottleneck.severity, 'medium');
          assert.ok(bottleneck.duration > detector.options.slowQueryThreshold);
          done();
        }
      });

      detector.trackQuery('SELECT * FROM large_table', 120);
    });
  });

  describe('Request Tracking', function() {
    it('should track slow requests', function() {
      const endpoint = '/api/users';
      const duration = 800;
      const statusCode = 200;

      detector.trackRequest(endpoint, duration, statusCode);

      assert.strictEqual(detector.metrics.requests.length, 1);
      const metric = detector.metrics.requests[0];
      assert.strictEqual(metric.endpoint, endpoint);
      assert.strictEqual(metric.duration, duration);
      assert.strictEqual(metric.statusCode, statusCode);
    });

    it('should report slow request bottleneck', function(done) {
      detector.on('bottleneck', (bottleneck) => {
        if (bottleneck.type === 'slow_request') {
          assert.strictEqual(bottleneck.severity, 'medium');
          assert.strictEqual(bottleneck.endpoint, '/api/slow');
          done();
        }
      });

      detector.trackRequest('/api/slow', 1200, 200);
    });
  });

  describe('Analysis', function() {
    it('should analyze bottlenecks', function() {
      // Add some test bottlenecks
      detector.reportBottleneck({
        type: 'event_loop_lag',
        severity: 'high',
        lag: 200,
        message: 'Test lag'
      });

      detector.reportBottleneck({
        type: 'slow_query',
        severity: 'medium',
        duration: 150,
        message: 'Test query'
      });

      const analysis = detector.analyzeBottlenecks();

      assert.strictEqual(analysis.summary.total_bottlenecks, 2);
      assert.strictEqual(analysis.summary.high_severity, 1);
      assert.strictEqual(analysis.summary.medium_severity, 1);
      assert.strictEqual(analysis.categories.event_loop, 1);
      assert.strictEqual(analysis.categories.queries, 1);
    });

    it('should generate recommendations', function() {
      // Add multiple bottlenecks of same type
      for (let i = 0; i < 10; i++) {
        detector.reportBottleneck({
          type: 'slow_query',
          severity: 'medium',
          duration: 150,
          message: `Test query ${i}`
        });
      }

      const analysis = detector.analyzeBottlenecks();

      assert.ok(analysis.recommendations.length > 0);
      const dbRec = analysis.recommendations.find(r => r.category === 'database');
      assert.ok(dbRec);
      assert.strictEqual(dbRec.priority, 'medium');
    });
  });

  describe('Reporting', function() {
    it('should generate report file', async function() {
      detector.reportBottleneck({
        type: 'test',
        severity: 'low',
        message: 'Test bottleneck'
      });

      const reportPath = await detector.generateReport();

      assert.ok(reportPath);
      assert.ok(await fileExists(reportPath));

      const content = await fs.readFile(reportPath, 'utf8');
      const report = JSON.parse(content);

      assert.ok(report.metadata);
      assert.ok(report.analysis);
      assert.ok(report.bottlenecks);
      assert.ok(report.recommendations);
    });

    it('should include statistics in report', async function() {
      // Add some metrics
      detector.metrics.eventLoop.push({ lag: 50 }, { lag: 100 }, { lag: 150 });

      const reportPath = await detector.generateReport();
      const content = await fs.readFile(reportPath, 'utf8');
      const report = JSON.parse(content);

      assert.ok(report.metrics.eventLoop);
      assert.strictEqual(report.metrics.eventLoop.count, 3);
      assert.strictEqual(report.metrics.eventLoop.min, 50);
      assert.strictEqual(report.metrics.eventLoop.max, 150);
    });
  });

  describe('Statistics', function() {
    it('should calculate statistics correctly', function() {
      const values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
      const stats = detector.getStatistics(values);

      assert.strictEqual(stats.count, 10);
      assert.strictEqual(stats.min, 10);
      assert.strictEqual(stats.max, 100);
      assert.strictEqual(stats.avg, 55);
      assert.strictEqual(stats.median, 60);
    });

    it('should handle empty values', function() {
      const stats = detector.getStatistics([]);
      assert.strictEqual(stats, null);
    });
  });
});

// Helper function
async function fileExists(path) {
  try {
    await fs.access(path);
    return true;
  } catch {
    return false;
  }
}

// Run tests if executed directly
if (require.main === module) {
  const Mocha = require('mocha');
  const mocha = new Mocha();
  mocha.addFile(__filename);

  mocha.run(failures => {
    process.exitCode = failures ? 1 : 0;
  });
}

module.exports = { BottleneckDetector };
