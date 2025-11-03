#!/usr/bin/env node
/**
 * Bottleneck Detector - Identify performance bottlenecks in Node.js applications
 *
 * Features:
 * - Event loop lag detection
 * - Promise queue analysis
 * - Async operation tracking
 * - Database query performance
 * - HTTP request timing
 * - Memory leak detection
 */

const v8 = require('v8');
const perf_hooks = require('perf_hooks');
const { EventEmitter } = require('events');
const fs = require('fs').promises;

class BottleneckDetector extends EventEmitter {
  constructor(options = {}) {
    super();

    this.options = {
      eventLoopThreshold: options.eventLoopThreshold || 100, // ms
      memoryLeakThreshold: options.memoryLeakThreshold || 10, // MB growth
      slowQueryThreshold: options.slowQueryThreshold || 100, // ms
      slowRequestThreshold: options.slowRequestThreshold || 1000, // ms
      sampleInterval: options.sampleInterval || 1000, // ms
      outputDir: options.outputDir || './bottleneck-reports',
      ...options
    };

    this.metrics = {
      eventLoop: [],
      memory: [],
      queries: [],
      requests: [],
      promises: [],
      bottlenecks: []
    };

    this.startTime = Date.now();
    this.monitoring = false;
  }

  async start() {
    this.monitoring = true;
    console.log('Starting bottleneck detection...');

    // Monitor event loop lag
    this.monitorEventLoop();

    // Monitor memory usage
    this.monitorMemory();

    // Monitor async operations
    this.monitorAsyncOps();

    // Set up periodic analysis
    this.analysisInterval = setInterval(() => {
      this.analyzeBottlenecks();
    }, this.options.sampleInterval * 10);

    this.emit('started');
  }

  stop() {
    this.monitoring = false;

    if (this.eventLoopMonitor) {
      clearInterval(this.eventLoopMonitor);
    }
    if (this.memoryMonitor) {
      clearInterval(this.memoryMonitor);
    }
    if (this.analysisInterval) {
      clearInterval(this.analysisInterval);
    }

    this.emit('stopped');
    console.log('Stopped bottleneck detection');
  }

  monitorEventLoop() {
    let lastCheck = Date.now();

    this.eventLoopMonitor = setInterval(() => {
      const now = Date.now();
      const lag = now - lastCheck - this.options.sampleInterval;
      lastCheck = now;

      this.metrics.eventLoop.push({
        timestamp: now,
        lag: lag
      });

      if (lag > this.options.eventLoopThreshold) {
        this.reportBottleneck({
          type: 'event_loop_lag',
          severity: lag > this.options.eventLoopThreshold * 2 ? 'high' : 'medium',
          lag: lag,
          threshold: this.options.eventLoopThreshold,
          timestamp: now,
          message: `Event loop lag detected: ${lag}ms (threshold: ${this.options.eventLoopThreshold}ms)`
        });
      }
    }, this.options.sampleInterval);
  }

  monitorMemory() {
    let baselineMemory = null;

    this.memoryMonitor = setInterval(() => {
      const usage = process.memoryUsage();
      const heapUsed = usage.heapUsed / 1024 / 1024; // MB

      if (baselineMemory === null) {
        baselineMemory = heapUsed;
      }

      const growth = heapUsed - baselineMemory;

      this.metrics.memory.push({
        timestamp: Date.now(),
        heapUsed: heapUsed,
        heapTotal: usage.heapTotal / 1024 / 1024,
        external: usage.external / 1024 / 1024,
        rss: usage.rss / 1024 / 1024,
        growth: growth
      });

      if (growth > this.options.memoryLeakThreshold) {
        this.reportBottleneck({
          type: 'memory_leak',
          severity: 'high',
          growth: growth,
          current: heapUsed,
          baseline: baselineMemory,
          threshold: this.options.memoryLeakThreshold,
          timestamp: Date.now(),
          message: `Potential memory leak: ${growth.toFixed(2)}MB growth from baseline`
        });
      }
    }, this.options.sampleInterval * 5);
  }

  monitorAsyncOps() {
    const asyncHook = perf_hooks.monitorEventLoopDelay();
    asyncHook.enable();

    this.asyncMonitor = setInterval(() => {
      const delay = asyncHook.mean / 1000000; // Convert to ms

      this.metrics.promises.push({
        timestamp: Date.now(),
        delay: delay,
        min: asyncHook.min / 1000000,
        max: asyncHook.max / 1000000,
        stddev: asyncHook.stddev / 1000000
      });

      if (delay > this.options.eventLoopThreshold) {
        this.reportBottleneck({
          type: 'async_delay',
          severity: 'medium',
          delay: delay,
          max: asyncHook.max / 1000000,
          threshold: this.options.eventLoopThreshold,
          timestamp: Date.now(),
          message: `Async operation delay: ${delay.toFixed(2)}ms average`
        });
      }
    }, this.options.sampleInterval * 3);
  }

  trackQuery(query, duration) {
    const queryMetric = {
      timestamp: Date.now(),
      query: query,
      duration: duration
    };

    this.metrics.queries.push(queryMetric);

    if (duration > this.options.slowQueryThreshold) {
      this.reportBottleneck({
        type: 'slow_query',
        severity: duration > this.options.slowQueryThreshold * 2 ? 'high' : 'medium',
        query: query,
        duration: duration,
        threshold: this.options.slowQueryThreshold,
        timestamp: Date.now(),
        message: `Slow query detected: ${duration}ms`
      });
    }
  }

  trackRequest(endpoint, duration, statusCode) {
    const requestMetric = {
      timestamp: Date.now(),
      endpoint: endpoint,
      duration: duration,
      statusCode: statusCode
    };

    this.metrics.requests.push(requestMetric);

    if (duration > this.options.slowRequestThreshold) {
      this.reportBottleneck({
        type: 'slow_request',
        severity: duration > this.options.slowRequestThreshold * 2 ? 'high' : 'medium',
        endpoint: endpoint,
        duration: duration,
        statusCode: statusCode,
        threshold: this.options.slowRequestThreshold,
        timestamp: Date.now(),
        message: `Slow request: ${endpoint} took ${duration}ms`
      });
    }
  }

  reportBottleneck(bottleneck) {
    this.metrics.bottlenecks.push(bottleneck);
    this.emit('bottleneck', bottleneck);

    if (bottleneck.severity === 'high') {
      console.warn(`[HIGH] ${bottleneck.message}`);
    }
  }

  analyzeBottlenecks() {
    const analysis = {
      timestamp: Date.now(),
      duration: Date.now() - this.startTime,
      summary: {
        total_bottlenecks: this.metrics.bottlenecks.length,
        high_severity: this.metrics.bottlenecks.filter(b => b.severity === 'high').length,
        medium_severity: this.metrics.bottlenecks.filter(b => b.severity === 'medium').length,
        low_severity: this.metrics.bottlenecks.filter(b => b.severity === 'low').length
      },
      categories: {
        event_loop: this.metrics.bottlenecks.filter(b => b.type === 'event_loop_lag').length,
        memory: this.metrics.bottlenecks.filter(b => b.type === 'memory_leak').length,
        queries: this.metrics.bottlenecks.filter(b => b.type === 'slow_query').length,
        requests: this.metrics.bottlenecks.filter(b => b.type === 'slow_request').length,
        async: this.metrics.bottlenecks.filter(b => b.type === 'async_delay').length
      },
      recommendations: this.generateRecommendations()
    };

    this.emit('analysis', analysis);
    return analysis;
  }

  generateRecommendations() {
    const recommendations = [];
    const bottlenecks = this.metrics.bottlenecks;

    // Event loop recommendations
    const eventLoopIssues = bottlenecks.filter(b => b.type === 'event_loop_lag').length;
    if (eventLoopIssues > 5) {
      recommendations.push({
        category: 'event_loop',
        priority: 'high',
        issue: `${eventLoopIssues} event loop lag incidents detected`,
        suggestion: 'Move CPU-intensive operations to worker threads or use setImmediate() to break up long tasks'
      });
    }

    // Memory recommendations
    const memoryIssues = bottlenecks.filter(b => b.type === 'memory_leak').length;
    if (memoryIssues > 0) {
      recommendations.push({
        category: 'memory',
        priority: 'high',
        issue: 'Potential memory leak detected',
        suggestion: 'Use heap snapshots to identify memory retention. Check for event listener leaks and unclosed connections.'
      });
    }

    // Query recommendations
    const slowQueries = bottlenecks.filter(b => b.type === 'slow_query');
    if (slowQueries.length > 10) {
      const avgDuration = slowQueries.reduce((sum, q) => sum + q.duration, 0) / slowQueries.length;
      recommendations.push({
        category: 'database',
        priority: 'medium',
        issue: `${slowQueries.length} slow queries detected (avg: ${avgDuration.toFixed(0)}ms)`,
        suggestion: 'Add indexes, optimize query structure, or implement query result caching'
      });
    }

    // Request recommendations
    const slowRequests = bottlenecks.filter(b => b.type === 'slow_request');
    if (slowRequests.length > 10) {
      recommendations.push({
        category: 'http',
        priority: 'medium',
        issue: `${slowRequests.length} slow HTTP requests detected`,
        suggestion: 'Implement response caching, optimize middleware chain, or use CDN for static assets'
      });
    }

    return recommendations;
  }

  async generateReport() {
    const analysis = this.analyzeBottlenecks();
    const report = {
      metadata: {
        generated: new Date().toISOString(),
        duration: Date.now() - this.startTime,
        version: '1.0.0'
      },
      analysis: analysis,
      metrics: {
        eventLoop: this.getStatistics(this.metrics.eventLoop.map(e => e.lag)),
        memory: this.getStatistics(this.metrics.memory.map(m => m.heapUsed)),
        queries: this.getStatistics(this.metrics.queries.map(q => q.duration)),
        requests: this.getStatistics(this.metrics.requests.map(r => r.duration))
      },
      bottlenecks: this.metrics.bottlenecks,
      recommendations: analysis.recommendations
    };

    // Save report
    const outputDir = this.options.outputDir;
    await fs.mkdir(outputDir, { recursive: true });
    const reportPath = `${outputDir}/bottleneck-report-${Date.now()}.json`;
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));

    console.log(`\nBottleneck Report Generated:`);
    console.log(`  Total Bottlenecks: ${analysis.summary.total_bottlenecks}`);
    console.log(`  High Severity: ${analysis.summary.high_severity}`);
    console.log(`  Report saved to: ${reportPath}`);

    return reportPath;
  }

  getStatistics(values) {
    if (values.length === 0) return null;

    const sorted = values.sort((a, b) => a - b);
    const sum = values.reduce((a, b) => a + b, 0);

    return {
      count: values.length,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      avg: sum / values.length,
      median: sorted[Math.floor(sorted.length / 2)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)]
    };
  }
}

// CLI usage
if (require.main === module) {
  const detector = new BottleneckDetector({
    eventLoopThreshold: 100,
    memoryLeakThreshold: 10,
    slowQueryThreshold: 100,
    slowRequestThreshold: 1000,
    sampleInterval: 1000
  });

  detector.on('bottleneck', (bottleneck) => {
    console.log(`[${bottleneck.severity.toUpperCase()}] ${bottleneck.type}: ${bottleneck.message}`);
  });

  detector.start();

  // Run for 60 seconds then generate report
  setTimeout(async () => {
    await detector.generateReport();
    detector.stop();
    process.exit(0);
  }, 60000);

  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\nGenerating final report...');
    await detector.generateReport();
    detector.stop();
    process.exit(0);
  });
}

module.exports = BottleneckDetector;
