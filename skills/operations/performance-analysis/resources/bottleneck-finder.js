#!/usr/bin/env node
/**
 * Bottleneck Finder
 * Real-time bottleneck detection and analysis for Claude Flow swarms
 */

const fs = require('fs');
const path = require('path');

/**
 * Bottleneck categories and detection logic
 */
class BottleneckFinder {
  constructor(config = {}) {
    this.thresholds = {
      communicationDelay: config.communicationDelay || 2000, // ms
      taskTime: config.taskTime || 60000, // ms
      memoryUsage: config.memoryUsage || 0.8, // 80%
      utilization: config.utilization || 0.5, // 50%
      queueDepth: config.queueDepth || 10, // tasks
      cacheHitRate: config.cacheHitRate || 0.7, // 70%
      networkLatency: config.networkLatency || 1000 // ms
    };

    this.metrics = [];
    this.bottlenecks = {
      critical: [],
      warning: [],
      info: []
    };
  }

  /**
   * Analyze metrics and detect bottlenecks
   * @param {Object} metrics - Current system metrics
   * @returns {Object} Detected bottlenecks by severity
   */
  analyze(metrics) {
    this.metrics.push({
      timestamp: Date.now(),
      ...metrics
    });

    // Reset bottleneck lists
    this.bottlenecks = {
      critical: [],
      warning: [],
      info: []
    };

    // Run detection algorithms
    this.detectCommunicationBottlenecks(metrics);
    this.detectProcessingBottlenecks(metrics);
    this.detectMemoryBottlenecks(metrics);
    this.detectNetworkBottlenecks(metrics);
    this.detectCoordinationBottlenecks(metrics);

    return this.bottlenecks;
  }

  /**
   * Detect communication bottlenecks
   * @param {Object} metrics - System metrics
   */
  detectCommunicationBottlenecks(metrics) {
    const { agents = [], messageQueue = {} } = metrics;

    // Check message delays
    agents.forEach(agent => {
      if (agent.avgResponseTime > this.thresholds.communicationDelay) {
        this.addBottleneck('critical', {
          type: 'communication',
          subtype: 'message_delay',
          agent: agent.id,
          value: agent.avgResponseTime,
          threshold: this.thresholds.communicationDelay,
          impact: this.calculateImpact(
            agent.avgResponseTime,
            this.thresholds.communicationDelay
          ),
          description: `Agent ${agent.id} response time: ${agent.avgResponseTime}ms (threshold: ${this.thresholds.communicationDelay}ms)`,
          recommendation: 'Consider switching to hierarchical topology or reducing message payload size'
        });
      }
    });

    // Check queue depth
    if (messageQueue.depth > this.thresholds.queueDepth) {
      this.addBottleneck('warning', {
        type: 'communication',
        subtype: 'queue_depth',
        value: messageQueue.depth,
        threshold: this.thresholds.queueDepth,
        impact: this.calculateImpact(
          messageQueue.depth,
          this.thresholds.queueDepth
        ),
        description: `Message queue depth: ${messageQueue.depth} (threshold: ${this.thresholds.queueDepth})`,
        recommendation: 'Increase concurrent message processing or optimize message handling'
      });
    }
  }

  /**
   * Detect processing bottlenecks
   * @param {Object} metrics - System metrics
   */
  detectProcessingBottlenecks(metrics) {
    const { agents = [], tasks = {} } = metrics;

    // Check task completion times
    agents.forEach(agent => {
      if (agent.avgTaskTime > this.thresholds.taskTime) {
        this.addBottleneck('critical', {
          type: 'processing',
          subtype: 'slow_tasks',
          agent: agent.id,
          value: agent.avgTaskTime,
          threshold: this.thresholds.taskTime,
          impact: this.calculateImpact(
            agent.avgTaskTime,
            this.thresholds.taskTime
          ),
          description: `Agent ${agent.id} avg task time: ${agent.avgTaskTime}ms (threshold: ${this.thresholds.taskTime}ms)`,
          recommendation: 'Break down complex tasks or spawn additional specialized agents'
        });
      }

      // Check utilization
      if (agent.utilization < this.thresholds.utilization) {
        this.addBottleneck('warning', {
          type: 'processing',
          subtype: 'low_utilization',
          agent: agent.id,
          value: agent.utilization,
          threshold: this.thresholds.utilization,
          impact: 1 - agent.utilization,
          description: `Agent ${agent.id} utilization: ${(agent.utilization * 100).toFixed(1)}% (threshold: ${(this.thresholds.utilization * 100).toFixed(1)}%)`,
          recommendation: 'Rebalance workload or reduce agent count for this role'
        });
      }
    });

    // Check pending tasks
    if (tasks.pending && tasks.pending.length > this.thresholds.queueDepth) {
      this.addBottleneck('critical', {
        type: 'processing',
        subtype: 'task_backlog',
        value: tasks.pending.length,
        threshold: this.thresholds.queueDepth,
        impact: this.calculateImpact(
          tasks.pending.length,
          this.thresholds.queueDepth
        ),
        description: `Task backlog: ${tasks.pending.length} pending tasks (threshold: ${this.thresholds.queueDepth})`,
        recommendation: 'Spawn additional agents or optimize task distribution'
      });
    }
  }

  /**
   * Detect memory bottlenecks
   * @param {Object} metrics - System metrics
   */
  detectMemoryBottlenecks(metrics) {
    const { agents = [], cache = {} } = metrics;

    // Check agent memory usage
    agents.forEach(agent => {
      if (agent.memoryUsage > this.thresholds.memoryUsage) {
        this.addBottleneck('critical', {
          type: 'memory',
          subtype: 'high_usage',
          agent: agent.id,
          value: agent.memoryUsage,
          threshold: this.thresholds.memoryUsage,
          impact: agent.memoryUsage,
          description: `Agent ${agent.id} memory usage: ${(agent.memoryUsage * 100).toFixed(1)}% (threshold: ${(this.thresholds.memoryUsage * 100).toFixed(1)}%)`,
          recommendation: 'Enable memory optimization or implement aggressive garbage collection'
        });
      }
    });

    // Check cache performance
    if (cache.hitRate && cache.hitRate < this.thresholds.cacheHitRate) {
      this.addBottleneck('warning', {
        type: 'memory',
        subtype: 'low_cache_hits',
        value: cache.hitRate,
        threshold: this.thresholds.cacheHitRate,
        impact: 1 - cache.hitRate,
        description: `Cache hit rate: ${(cache.hitRate * 100).toFixed(1)}% (threshold: ${(this.thresholds.cacheHitRate * 100).toFixed(1)}%)`,
        recommendation: 'Implement cache warming or increase cache size'
      });
    }
  }

  /**
   * Detect network bottlenecks
   * @param {Object} metrics - System metrics
   */
  detectNetworkBottlenecks(metrics) {
    const { network = {} } = metrics;

    // Check API latency
    if (network.apiLatency > this.thresholds.networkLatency) {
      this.addBottleneck('warning', {
        type: 'network',
        subtype: 'api_latency',
        value: network.apiLatency,
        threshold: this.thresholds.networkLatency,
        impact: this.calculateImpact(
          network.apiLatency,
          this.thresholds.networkLatency
        ),
        description: `API latency: ${network.apiLatency}ms (threshold: ${this.thresholds.networkLatency}ms)`,
        recommendation: 'Implement request batching or use connection pooling'
      });
    }

    // Check failed requests
    if (network.failedRequests && network.failedRequests > 5) {
      this.addBottleneck('critical', {
        type: 'network',
        subtype: 'failed_requests',
        value: network.failedRequests,
        threshold: 5,
        impact: Math.min(network.failedRequests / 10, 1),
        description: `Failed network requests: ${network.failedRequests} (threshold: 5)`,
        recommendation: 'Implement retry logic with exponential backoff'
      });
    }
  }

  /**
   * Detect coordination bottlenecks
   * @param {Object} metrics - System metrics
   */
  detectCoordinationBottlenecks(metrics) {
    const { swarm = {}, agents = [] } = metrics;

    // Check agent distribution
    const agentsByType = agents.reduce((acc, agent) => {
      acc[agent.type] = (acc[agent.type] || 0) + 1;
      return acc;
    }, {});

    // Detect imbalanced distribution
    const counts = Object.values(agentsByType);
    if (counts.length > 1) {
      const max = Math.max(...counts);
      const min = Math.min(...counts);
      if (max / min > 3) {
        this.addBottleneck('warning', {
          type: 'coordination',
          subtype: 'imbalanced_agents',
          value: max / min,
          threshold: 3,
          impact: (max / min - 1) / 10,
          description: `Agent distribution imbalance: ${max}:${min} ratio (threshold: 3:1)`,
          recommendation: 'Rebalance agent types based on workload patterns'
        });
      }
    }

    // Check topology efficiency
    if (swarm.topology === 'mesh' && agents.length > 10) {
      this.addBottleneck('info', {
        type: 'coordination',
        subtype: 'inefficient_topology',
        value: agents.length,
        threshold: 10,
        impact: 0.2,
        description: `Mesh topology with ${agents.length} agents may be inefficient`,
        recommendation: 'Consider switching to hierarchical topology for better scalability'
      });
    }
  }

  /**
   * Add bottleneck to appropriate severity list
   * @param {string} severity - Severity level
   * @param {Object} bottleneck - Bottleneck details
   */
  addBottleneck(severity, bottleneck) {
    if (this.bottlenecks[severity]) {
      this.bottlenecks[severity].push({
        ...bottleneck,
        timestamp: Date.now()
      });
    }
  }

  /**
   * Calculate impact percentage
   * @param {number} actual - Actual value
   * @param {number} threshold - Threshold value
   * @returns {number} Impact percentage (0-1)
   */
  calculateImpact(actual, threshold) {
    return Math.min((actual - threshold) / threshold, 1);
  }

  /**
   * Generate bottleneck report
   * @param {string} format - Output format ('json' or 'text')
   * @returns {string} Formatted report
   */
  generateReport(format = 'text') {
    if (format === 'json') {
      return JSON.stringify({
        timestamp: Date.now(),
        summary: {
          critical: this.bottlenecks.critical.length,
          warning: this.bottlenecks.warning.length,
          info: this.bottlenecks.info.length
        },
        bottlenecks: this.bottlenecks
      }, null, 2);
    }

    // Text format
    const lines = [];
    lines.push('='.repeat(70));
    lines.push('BOTTLENECK ANALYSIS REPORT');
    lines.push('='.repeat(70));
    lines.push('');
    lines.push('SUMMARY');
    lines.push('-'.repeat(70));
    lines.push(`Critical Issues: ${this.bottlenecks.critical.length}`);
    lines.push(`Warnings: ${this.bottlenecks.warning.length}`);
    lines.push(`Info: ${this.bottlenecks.info.length}`);
    lines.push('');

    // Critical bottlenecks
    if (this.bottlenecks.critical.length > 0) {
      lines.push('CRITICAL BOTTLENECKS');
      lines.push('-'.repeat(70));
      this.bottlenecks.critical.forEach((b, i) => {
        lines.push(`${i + 1}. ${b.type.toUpperCase()} - ${b.subtype}`);
        lines.push(`   Description: ${b.description}`);
        lines.push(`   Impact: ${(b.impact * 100).toFixed(1)}%`);
        lines.push(`   Recommendation: ${b.recommendation}`);
        lines.push('');
      });
    }

    // Warning bottlenecks
    if (this.bottlenecks.warning.length > 0) {
      lines.push('WARNINGS');
      lines.push('-'.repeat(70));
      this.bottlenecks.warning.forEach((b, i) => {
        lines.push(`${i + 1}. ${b.type.toUpperCase()} - ${b.subtype}`);
        lines.push(`   Description: ${b.description}`);
        lines.push(`   Impact: ${(b.impact * 100).toFixed(1)}%`);
        lines.push(`   Recommendation: ${b.recommendation}`);
        lines.push('');
      });
    }

    lines.push('='.repeat(70));
    return lines.join('\n');
  }

  /**
   * Export bottlenecks to file
   * @param {string} filepath - Output file path
   * @param {string} format - Output format
   */
  exportToFile(filepath, format = 'json') {
    const content = this.generateReport(format);
    fs.writeFileSync(filepath, content);
    console.log(`Bottleneck report exported to: ${filepath}`);
  }
}

// Example usage
if (require.main === module) {
  const finder = new BottleneckFinder();

  // Simulate metrics
  const metrics = {
    agents: [
      {
        id: 'agent-1',
        type: 'coder',
        avgResponseTime: 2500,
        avgTaskTime: 65000,
        utilization: 0.45,
        memoryUsage: 0.85
      },
      {
        id: 'agent-2',
        type: 'reviewer',
        avgResponseTime: 1200,
        avgTaskTime: 45000,
        utilization: 0.75,
        memoryUsage: 0.60
      }
    ],
    messageQueue: {
      depth: 12
    },
    cache: {
      hitRate: 0.65
    },
    network: {
      apiLatency: 1200,
      failedRequests: 3
    },
    swarm: {
      topology: 'mesh'
    },
    tasks: {
      pending: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }
  };

  const bottlenecks = finder.analyze(metrics);
  console.log(finder.generateReport('text'));
  finder.exportToFile('bottlenecks.json', 'json');
}

module.exports = BottleneckFinder;
