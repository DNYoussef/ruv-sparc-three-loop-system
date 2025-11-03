#!/usr/bin/env node
/**
 * Example: Queen-Led Swarm Coordination
 * Demonstrates complete queen-worker coordination for full-stack development
 *
 * Scenario: Building an e-commerce microservices platform
 * - Strategic queen coordinates high-level objectives
 * - Specialized workers handle implementation
 * - Byzantine consensus for critical architectural decisions
 * - Collective memory for shared knowledge
 */

const crypto = require('crypto');

// ============================================================================
// Worker Agent Implementation
// ============================================================================

class WorkerAgent {
  constructor(id, type, capabilities) {
    this.id = id;
    this.type = type;
    this.capabilities = capabilities;
    this.status = 'idle';
    this.currentTask = null;
    this.tasksCompleted = 0;
    this.performanceScore = 1.0;
    this.knowledgeBase = new Map();
  }

  async executeTask(task) {
    console.log(`[${this.id}] Starting task: ${task.description}`);
    this.status = 'busy';
    this.currentTask = task;

    // Simulate task execution
    const duration = Math.random() * 2000 + 1000;
    await new Promise(resolve => setTimeout(resolve, duration));

    // Generate result based on worker type
    const result = this._generateResult(task);

    this.tasksCompleted++;
    this.status = 'idle';
    this.currentTask = null;

    console.log(`[${this.id}] Completed task: ${task.description}`);
    return result;
  }

  _generateResult(task) {
    const results = {
      researcher: {
        findings: `Research findings for: ${task.description}`,
        recommendations: ['Approach A', 'Approach B', 'Approach C'],
        confidence: 0.85
      },
      architect: {
        design: `Architecture design for: ${task.description}`,
        components: ['Service Layer', 'Data Layer', 'API Gateway'],
        patterns: ['Microservices', 'Event-Driven', 'CQRS']
      },
      coder: {
        implementation: `Implementation for: ${task.description}`,
        files: ['service.js', 'controller.js', 'model.js'],
        linesOfCode: Math.floor(Math.random() * 500) + 100
      },
      tester: {
        testResults: {
          passed: Math.floor(Math.random() * 50) + 40,
          failed: Math.floor(Math.random() * 5),
          coverage: Math.random() * 15 + 85
        }
      }
    };

    return results[this.type] || { result: 'Task completed' };
  }

  vote(topic, options) {
    // Vote based on expertise and knowledge
    const preference = this._determinePreference(topic, options);
    return {
      agentId: this.id,
      agentType: this.type,
      vote: preference,
      reasoning: `Based on ${this.type} expertise`,
      timestamp: new Date().toISOString()
    };
  }

  _determinePreference(topic, options) {
    // Simple preference logic based on worker type
    const preferences = {
      researcher: options[0], // Prefer first (most researched)
      architect: options[Math.floor(options.length / 2)], // Middle ground
      coder: options[options.length - 1], // Prefer last (most practical)
      tester: options[0] // Prefer safest
    };

    return preferences[this.type] || options[0];
  }

  shareKnowledge(key, value) {
    this.knowledgeBase.set(key, {
      value,
      timestamp: new Date().toISOString(),
      confidence: 0.9
    });
  }
}

// ============================================================================
// Queen Coordinator Implementation
// ============================================================================

class QueenCoordinator {
  constructor(objective, queenType = 'strategic') {
    this.objective = objective;
    this.queenType = queenType;
    this.workers = new Map();
    this.tasks = [];
    this.decisions = [];
    this.collectiveMemory = new Map();
    this.byzantineThreshold = 2/3;
  }

  // Spawn specialized workers
  spawnWorkers(workerConfigs) {
    console.log(`\n[Queen] Spawning ${workerConfigs.length} workers...`);

    for (const config of workerConfigs) {
      const worker = new WorkerAgent(
        `${config.type}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        config.type,
        config.capabilities
      );

      this.workers.set(worker.id, worker);
      console.log(`[Queen] Spawned ${worker.type}: ${worker.id}`);
    }

    console.log(`[Queen] Total workers: ${this.workers.size}\n`);
  }

  // Delegate task to best worker
  async delegateTask(description, priority = 5, requiredCapabilities = []) {
    console.log(`[Queen] Delegating task (priority ${priority}): ${description}`);

    const task = {
      id: `task-${this.tasks.length + 1}`,
      description,
      priority,
      requiredCapabilities,
      status: 'pending',
      createdAt: new Date().toISOString()
    };

    // Find best worker
    const worker = this._findBestWorker(requiredCapabilities);

    if (!worker) {
      console.log(`[Queen] No available worker for task: ${description}`);
      task.status = 'queued';
      this.tasks.push(task);
      return task;
    }

    task.assignedTo = worker.id;
    task.status = 'in_progress';
    this.tasks.push(task);

    // Execute task
    const result = await worker.executeTask(task);

    task.status = 'completed';
    task.completedAt = new Date().toISOString();
    task.result = result;

    // Store result in collective memory
    this.storeInMemory(task.id, result, 'task_result');

    return task;
  }

  _findBestWorker(requiredCapabilities) {
    let bestWorker = null;
    let bestScore = -1;

    for (const worker of this.workers.values()) {
      if (worker.status !== 'idle') continue;

      // Calculate capability match score
      const matchScore = requiredCapabilities.filter(cap =>
        worker.capabilities.includes(cap)
      ).length;

      const totalScore = matchScore * worker.performanceScore;

      if (totalScore > bestScore) {
        bestScore = totalScore;
        bestWorker = worker;
      }
    }

    return bestWorker;
  }

  // Build Byzantine consensus
  async buildConsensus(topic, options) {
    console.log(`\n[Queen] Building consensus on: ${topic}`);
    console.log(`[Queen] Options: ${options.join(', ')}`);

    const votes = [];

    // Collect worker votes
    for (const worker of this.workers.values()) {
      const vote = worker.vote(topic, options);
      votes.push(vote);
      console.log(`  ${worker.id} (${worker.type}): ${vote.vote}`);
    }

    // Queen's weighted vote (counts 3x)
    const queenVote = this._makeStrategicDecision(topic, options);
    console.log(`  [Queen] Strategic vote: ${queenVote} (weight: 3x)`);

    for (let i = 0; i < 3; i++) {
      votes.push({
        agentId: `queen-${i}`,
        agentType: 'queen',
        vote: queenVote,
        reasoning: 'Strategic decision',
        timestamp: new Date().toISOString()
      });
    }

    // Calculate Byzantine consensus
    const decision = this._calculateByzantineConsensus(votes, options);

    // Store decision
    this.decisions.push({
      topic,
      decision: decision.winner,
      confidence: decision.confidence,
      votes,
      timestamp: new Date().toISOString(),
      supermajority: decision.supermajority
    });

    console.log(`[Queen] Consensus reached: ${decision.winner}`);
    console.log(`[Queen] Confidence: ${(decision.confidence * 100).toFixed(1)}%`);
    console.log(`[Queen] Supermajority: ${decision.supermajority ? 'YES' : 'NO'}\n`);

    return decision;
  }

  _makeStrategicDecision(topic, options) {
    // Strategic queen prefers balanced, well-researched options
    if (this.queenType === 'strategic') {
      // Check collective memory for similar decisions
      for (const [key, value] of this.collectiveMemory) {
        if (key.includes(topic.toLowerCase())) {
          return value.decision || options[0];
        }
      }
    }

    return options[0]; // Default to first option
  }

  _calculateByzantineConsensus(votes, options) {
    const counts = {};
    options.forEach(opt => counts[opt] = 0);

    votes.forEach(vote => {
      if (counts[vote.vote] !== undefined) {
        counts[vote.vote]++;
      }
    });

    const totalVotes = votes.length;
    const requiredVotes = Math.ceil(totalVotes * this.byzantineThreshold);

    // Check for supermajority
    for (const [option, count] of Object.entries(counts)) {
      if (count >= requiredVotes) {
        return {
          winner: option,
          confidence: count / totalVotes,
          supermajority: true,
          voteCounts: counts
        };
      }
    }

    // No supermajority - use majority
    let winner = options[0];
    let maxVotes = 0;

    for (const [option, count] of Object.entries(counts)) {
      if (count > maxVotes) {
        maxVotes = count;
        winner = option;
      }
    }

    console.log(`[Queen] Warning: No supermajority reached (need ${requiredVotes}/${totalVotes})`);

    return {
      winner,
      confidence: maxVotes / totalVotes,
      supermajority: false,
      voteCounts: counts
    };
  }

  // Store in collective memory
  storeInMemory(key, value, type = 'knowledge') {
    this.collectiveMemory.set(key, {
      value,
      type,
      timestamp: new Date().toISOString(),
      confidence: 0.9
    });
  }

  // Get hive status
  getStatus() {
    const idle = Array.from(this.workers.values()).filter(w => w.status === 'idle').length;
    const busy = this.workers.size - idle;

    const completed = this.tasks.filter(t => t.status === 'completed').length;
    const inProgress = this.tasks.filter(t => t.status === 'in_progress').length;
    const pending = this.tasks.filter(t => t.status === 'pending').length;

    return {
      objective: this.objective,
      queenType: this.queenType,
      workers: {
        total: this.workers.size,
        idle,
        busy,
        utilization: (busy / this.workers.size * 100).toFixed(1) + '%'
      },
      tasks: {
        total: this.tasks.length,
        completed,
        inProgress,
        pending,
        completionRate: (completed / this.tasks.length * 100).toFixed(1) + '%'
      },
      decisions: this.decisions.length,
      collectiveMemory: this.collectiveMemory.size
    };
  }
}

// ============================================================================
// Demo: E-Commerce Platform Development
// ============================================================================

async function main() {
  console.log('='.repeat(80));
  console.log('Queen-Led Swarm: E-Commerce Microservices Platform Development');
  console.log('='.repeat(80));

  // Initialize strategic queen
  const queen = new QueenCoordinator(
    'Build e-commerce microservices platform',
    'strategic'
  );

  // Spawn specialized workers
  queen.spawnWorkers([
    { type: 'researcher', capabilities: ['analysis', 'research', 'patterns'] },
    { type: 'researcher', capabilities: ['analysis', 'research', 'market'] },
    { type: 'architect', capabilities: ['design', 'architecture', 'systems'] },
    { type: 'architect', capabilities: ['design', 'microservices', 'apis'] },
    { type: 'coder', capabilities: ['implementation', 'backend', 'nodejs'] },
    { type: 'coder', capabilities: ['implementation', 'frontend', 'react'] },
    { type: 'tester', capabilities: ['testing', 'qa', 'automation'] },
    { type: 'tester', capabilities: ['testing', 'integration', 'e2e'] }
  ]);

  // Phase 1: Research and Planning
  console.log('\n' + '='.repeat(80));
  console.log('PHASE 1: Research and Planning');
  console.log('='.repeat(80));

  await queen.delegateTask(
    'Research e-commerce architecture patterns',
    9,
    ['research', 'analysis']
  );

  await queen.delegateTask(
    'Analyze market requirements for payment processing',
    8,
    ['research', 'market']
  );

  // Consensus on architecture
  const archDecision = await queen.buildConsensus(
    'Microservices architecture pattern',
    ['Event-Driven', 'Service Mesh', 'CQRS']
  );

  // Phase 2: Design
  console.log('\n' + '='.repeat(80));
  console.log('PHASE 2: System Design');
  console.log('='.repeat(80));

  await queen.delegateTask(
    'Design product catalog service',
    8,
    ['design', 'architecture']
  );

  await queen.delegateTask(
    'Design order processing microservice',
    9,
    ['design', 'microservices']
  );

  // Consensus on API design
  const apiDecision = await queen.buildConsensus(
    'API communication protocol',
    ['REST', 'GraphQL', 'gRPC']
  );

  // Phase 3: Implementation
  console.log('\n' + '='.repeat(80));
  console.log('PHASE 3: Implementation');
  console.log('='.repeat(80));

  await queen.delegateTask(
    'Implement user authentication service',
    9,
    ['implementation', 'backend']
  );

  await queen.delegateTask(
    'Implement React shopping cart UI',
    7,
    ['implementation', 'frontend']
  );

  // Phase 4: Testing
  console.log('\n' + '='.repeat(80));
  console.log('PHASE 4: Testing and Validation');
  console.log('='.repeat(80));

  await queen.delegateTask(
    'Create integration tests for payment flow',
    8,
    ['testing', 'integration']
  );

  await queen.delegateTask(
    'Run end-to-end testing suite',
    9,
    ['testing', 'e2e']
  );

  // Final Status
  console.log('\n' + '='.repeat(80));
  console.log('FINAL STATUS');
  console.log('='.repeat(80));

  const status = queen.getStatus();
  console.log(JSON.stringify(status, null, 2));

  console.log('\n' + '='.repeat(80));
  console.log('CONSENSUS DECISIONS');
  console.log('='.repeat(80));

  queen.decisions.forEach((decision, i) => {
    console.log(`\n${i + 1}. ${decision.topic}`);
    console.log(`   Decision: ${decision.decision}`);
    console.log(`   Confidence: ${(decision.confidence * 100).toFixed(1)}%`);
    console.log(`   Supermajority: ${decision.supermajority ? 'YES' : 'NO'}`);
  });

  console.log('\n' + '='.repeat(80));
  console.log('Demo completed successfully!');
  console.log('='.repeat(80));
}

// Run demo
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { QueenCoordinator, WorkerAgent };
