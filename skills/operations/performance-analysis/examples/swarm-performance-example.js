#!/usr/bin/env node
/**
 * Swarm Performance Analysis Example
 * Comprehensive example of analyzing swarm performance with real-time monitoring
 *
 * This example demonstrates:
 * - Real-time metrics collection from multiple agents
 * - Performance analysis and bottleneck detection
 * - Automated optimization recommendations
 * - Report generation in multiple formats
 */

const fs = require('fs');
const path = require('path');

// Simulated swarm environment
class SwarmEnvironment {
  constructor(config = {}) {
    this.swarmId = config.swarmId || 'swarm-' + Date.now();
    this.topology = config.topology || 'mesh';
    this.agents = [];
    this.tasks = [];
    this.metrics = [];
    this.isRunning = false;
  }

  /**
   * Initialize swarm with agents
   */
  initialize(agentCount = 5) {
    console.log(`\n=== Initializing Swarm: ${this.swarmId} ===`);
    console.log(`Topology: ${this.topology}`);
    console.log(`Agent Count: ${agentCount}\n`);

    const agentTypes = ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'];

    for (let i = 0; i < agentCount; i++) {
      this.agents.push({
        id: `agent-${i + 1}`,
        type: agentTypes[i % agentTypes.length],
        status: 'idle',
        tasksCompleted: 0,
        tasksFailed: 0,
        totalTaskTime: 0,
        messageCount: 0,
        cpuUsage: 0,
        memoryUsage: 0
      });
    }

    console.log('Agents initialized:');
    this.agents.forEach(agent => {
      console.log(`  - ${agent.id} (${agent.type})`);
    });
  }

  /**
   * Simulate task execution
   */
  async executeTask(task) {
    // Select agent based on task type or round-robin
    const availableAgents = this.agents.filter(a => a.status === 'idle');
    if (availableAgents.length === 0) {
      console.log('⚠️  No available agents, queueing task...');
      return null;
    }

    const agent = availableAgents[Math.floor(Math.random() * availableAgents.length)];
    agent.status = 'busy';

    const startTime = Date.now();

    // Simulate task execution time (varies by complexity and agent type)
    const baseTime = task.complexity * 1000;
    const variance = Math.random() * 0.4 - 0.2; // ±20% variance
    const executionTime = baseTime * (1 + variance);

    // Simulate resource usage
    agent.cpuUsage = 0.5 + Math.random() * 0.4; // 50-90%
    agent.memoryUsage = 0.4 + Math.random() * 0.3; // 40-70%
    agent.messageCount += Math.floor(5 + Math.random() * 10);

    await new Promise(resolve => setTimeout(resolve, executionTime));

    const endTime = Date.now();
    const actualTime = endTime - startTime;

    // Determine success/failure (95% success rate)
    const success = Math.random() > 0.05;

    if (success) {
      agent.tasksCompleted++;
      agent.totalTaskTime += actualTime;
    } else {
      agent.tasksFailed++;
    }

    agent.status = 'idle';
    agent.cpuUsage *= 0.5; // Reduce CPU after task
    agent.memoryUsage *= 0.9; // Reduce memory slightly

    return {
      taskId: task.id,
      agentId: agent.id,
      success,
      executionTime: actualTime,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Collect metrics snapshot
   */
  collectMetrics() {
    const timestamp = new Date().toISOString();
    const agentMetrics = this.agents.map(agent => ({
      agent_id: agent.id,
      agent_type: agent.type,
      status: agent.status,
      tasks_completed: agent.tasksCompleted,
      tasks_failed: agent.tasksFailed,
      avg_task_time: agent.tasksCompleted > 0
        ? agent.totalTaskTime / agent.tasksCompleted
        : 0,
      cpu_usage: agent.cpuUsage,
      memory_usage: agent.memoryUsage,
      message_count: agent.messageCount,
      response_time: 1000 + Math.random() * 1500,
      utilization: agent.tasksCompleted / (agent.tasksCompleted + agent.tasksFailed + 1)
    }));

    const metrics = {
      timestamp,
      swarm_id: this.swarmId,
      topology: this.topology,
      agents: agentMetrics,
      system: {
        cpu_usage: agentMetrics.reduce((sum, a) => sum + a.cpu_usage, 0) / agentMetrics.length,
        memory_usage: agentMetrics.reduce((sum, a) => sum + a.memory_usage, 0) / agentMetrics.length
      }
    };

    this.metrics.push(metrics);
    return metrics;
  }

  /**
   * Run swarm simulation
   */
  async runSimulation(taskCount = 20, interval = 500) {
    console.log(`\n=== Running Simulation ===`);
    console.log(`Tasks: ${taskCount}`);
    console.log(`Interval: ${interval}ms\n`);

    this.isRunning = true;

    // Generate tasks
    const tasks = Array.from({ length: taskCount }, (_, i) => ({
      id: `task-${i + 1}`,
      complexity: 1 + Math.random() * 4, // 1-5 second tasks
      type: ['implementation', 'analysis', 'review', 'optimization'][Math.floor(Math.random() * 4)]
    }));

    const results = [];
    const metricsInterval = setInterval(() => {
      if (this.isRunning) {
        this.collectMetrics();
      }
    }, 2000); // Collect metrics every 2 seconds

    // Execute tasks
    for (const task of tasks) {
      console.log(`Executing: ${task.id} (complexity: ${task.complexity.toFixed(1)})`);
      const result = await this.executeTask(task);
      if (result) {
        results.push(result);
        const status = result.success ? '✓' : '✗';
        console.log(`  ${status} ${result.agentId} - ${result.executionTime.toFixed(0)}ms`);
      }
      await new Promise(resolve => setTimeout(resolve, interval));
    }

    clearInterval(metricsInterval);
    this.isRunning = false;

    console.log('\n=== Simulation Complete ===\n');

    return results;
  }

  /**
   * Generate performance summary
   */
  generateSummary() {
    const totalTasks = this.agents.reduce((sum, a) => sum + a.tasksCompleted + a.tasksFailed, 0);
    const completedTasks = this.agents.reduce((sum, a) => sum + a.tasksCompleted, 0);
    const failedTasks = this.agents.reduce((sum, a) => sum + a.tasksFailed, 0);
    const avgTaskTime = this.agents.reduce((sum, a) => sum + a.totalTaskTime, 0) / completedTasks;

    return {
      swarm_id: this.swarmId,
      topology: this.topology,
      agent_count: this.agents.length,
      total_tasks: totalTasks,
      completed_tasks: completedTasks,
      failed_tasks: failedTasks,
      success_rate: (completedTasks / totalTasks * 100).toFixed(1) + '%',
      avg_task_time: avgTaskTime.toFixed(0) + 'ms',
      avg_utilization: (this.agents.reduce((sum, a) => sum + a.tasksCompleted, 0) /
                        (totalTasks * this.agents.length) * 100).toFixed(1) + '%'
    };
  }
}

/**
 * Performance Analyzer
 */
class PerformanceAnalyzer {
  constructor() {
    this.bottlenecks = [];
    this.recommendations = [];
  }

  /**
   * Analyze collected metrics
   */
  analyze(metrics) {
    console.log('=== Analyzing Performance ===\n');

    const latestMetrics = metrics[metrics.length - 1];
    const agents = latestMetrics.agents;

    // Detect slow agents
    const avgTaskTime = agents.reduce((sum, a) => sum + a.avg_task_time, 0) / agents.length;
    const slowAgents = agents.filter(a => a.avg_task_time > avgTaskTime * 1.5);

    if (slowAgents.length > 0) {
      this.bottlenecks.push({
        type: 'processing',
        severity: 'warning',
        description: `${slowAgents.length} agent(s) performing 50% slower than average`,
        affected: slowAgents.map(a => a.agent_id),
        recommendation: 'Consider task redistribution or agent optimization'
      });
    }

    // Detect high memory usage
    const highMemoryAgents = agents.filter(a => a.memory_usage > 0.8);
    if (highMemoryAgents.length > 0) {
      this.bottlenecks.push({
        type: 'memory',
        severity: 'critical',
        description: `${highMemoryAgents.length} agent(s) using >80% memory`,
        affected: highMemoryAgents.map(a => a.agent_id),
        recommendation: 'Enable garbage collection or restart affected agents'
      });
    }

    // Detect low utilization
    const lowUtilAgents = agents.filter(a => a.utilization < 0.5);
    if (lowUtilAgents.length > 0) {
      this.bottlenecks.push({
        type: 'coordination',
        severity: 'info',
        description: `${lowUtilAgents.length} agent(s) under-utilized (<50%)`,
        affected: lowUtilAgents.map(a => a.agent_id),
        recommendation: 'Rebalance workload or reduce agent count'
      });
    }

    // Generate recommendations
    this.generateRecommendations(latestMetrics);

    return {
      bottlenecks: this.bottlenecks,
      recommendations: this.recommendations
    };
  }

  /**
   * Generate optimization recommendations
   */
  generateRecommendations(metrics) {
    const agents = metrics.agents;

    // Topology optimization
    if (metrics.topology === 'mesh' && agents.length > 8) {
      this.recommendations.push({
        priority: 'high',
        title: 'Switch to Hierarchical Topology',
        description: 'Mesh topology inefficient for >8 agents',
        impact: '30-40% reduction in coordination overhead',
        implementation: 'npx claude-flow@alpha swarm init --topology hierarchical'
      });
    }

    // Concurrency optimization
    const avgUtil = agents.reduce((sum, a) => sum + a.utilization, 0) / agents.length;
    if (avgUtil > 0.9) {
      this.recommendations.push({
        priority: 'medium',
        title: 'Increase Agent Concurrency',
        description: 'High utilization detected (>90%)',
        impact: '20-30% reduction in task completion time',
        implementation: 'Spawn 2-3 additional agents of most utilized types'
      });
    }

    // Memory optimization
    const avgMemory = agents.reduce((sum, a) => sum + a.memory_usage, 0) / agents.length;
    if (avgMemory > 0.7) {
      this.recommendations.push({
        priority: 'medium',
        title: 'Enable Memory Optimization',
        description: 'Average memory usage >70%',
        impact: '40-50% reduction in memory footprint',
        implementation: 'Enable aggressive GC and cache cleanup'
      });
    }
  }

  /**
   * Generate report
   */
  generateReport(summary, analysis) {
    const lines = [];
    lines.push('=' .repeat(70));
    lines.push('SWARM PERFORMANCE ANALYSIS REPORT');
    lines.push('='.repeat(70));
    lines.push('');
    lines.push('SWARM SUMMARY');
    lines.push('-'.repeat(70));
    Object.entries(summary).forEach(([key, value]) => {
      lines.push(`${key.padEnd(20)}: ${value}`);
    });
    lines.push('');

    if (analysis.bottlenecks.length > 0) {
      lines.push('BOTTLENECKS DETECTED');
      lines.push('-'.repeat(70));
      analysis.bottlenecks.forEach((b, i) => {
        lines.push(`${i + 1}. [${b.severity.toUpperCase()}] ${b.type}`);
        lines.push(`   ${b.description}`);
        lines.push(`   Affected: ${b.affected.join(', ')}`);
        lines.push(`   Recommendation: ${b.recommendation}`);
        lines.push('');
      });
    }

    if (analysis.recommendations.length > 0) {
      lines.push('OPTIMIZATION RECOMMENDATIONS');
      lines.push('-'.repeat(70));
      analysis.recommendations.forEach((r, i) => {
        lines.push(`${i + 1}. [${r.priority.toUpperCase()}] ${r.title}`);
        lines.push(`   ${r.description}`);
        lines.push(`   Impact: ${r.impact}`);
        lines.push(`   Implementation: ${r.implementation}`);
        lines.push('');
      });
    }

    lines.push('='.repeat(70));

    return lines.join('\n');
  }
}

/**
 * Main execution
 */
async function main() {
  console.log('\n' + '='.repeat(70));
  console.log('CLAUDE FLOW SWARM PERFORMANCE ANALYSIS EXAMPLE');
  console.log('='.repeat(70) + '\n');

  // Create swarm environment
  const swarm = new SwarmEnvironment({
    swarmId: 'example-swarm-001',
    topology: 'mesh'
  });

  // Initialize with 6 agents
  swarm.initialize(6);

  // Run simulation
  await swarm.runSimulation(30, 300); // 30 tasks, 300ms interval

  // Generate summary
  const summary = swarm.generateSummary();
  console.log('\n=== Performance Summary ===\n');
  Object.entries(summary).forEach(([key, value]) => {
    console.log(`${key.padEnd(20)}: ${value}`);
  });

  // Analyze performance
  const analyzer = new PerformanceAnalyzer();
  const analysis = analyzer.analyze(swarm.metrics);

  // Generate and display report
  const report = analyzer.generateReport(summary, analysis);
  console.log('\n' + report);

  // Save metrics to file
  const outputDir = path.join(__dirname, 'output');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const metricsFile = path.join(outputDir, `swarm-metrics-${Date.now()}.json`);
  fs.writeFileSync(metricsFile, JSON.stringify({
    summary,
    analysis,
    raw_metrics: swarm.metrics
  }, null, 2));

  console.log(`\nMetrics saved to: ${metricsFile}`);
  console.log('\n' + '='.repeat(70) + '\n');
}

// Run example
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { SwarmEnvironment, PerformanceAnalyzer };
