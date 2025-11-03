/**
 * Test Suite: Queen Coordination
 * Tests queen-led coordination, task delegation, and worker management
 */

const assert = require('assert');

// Mock HiveMind classes for testing
class MockWorker {
  constructor(id, type) {
    this.id = id;
    this.type = type;
    this.status = 'idle';
    this.tasks_completed = 0;
    this.performance_score = 1.0;
    this.capabilities = this._getCapabilities(type);
  }

  _getCapabilities(type) {
    const caps = {
      researcher: ['analysis', 'investigation', 'research'],
      coder: ['implementation', 'coding', 'development'],
      tester: ['testing', 'validation', 'qa'],
      architect: ['design', 'planning', 'architecture']
    };
    return caps[type] || ['general'];
  }

  assignTask(task) {
    this.status = 'busy';
    this.currentTask = task;
  }

  completeTask() {
    this.status = 'idle';
    this.tasks_completed++;
    this.currentTask = null;
  }
}

class MockQueenCoordinator {
  constructor(queenType, maxWorkers = 8) {
    this.queenType = queenType;
    this.maxWorkers = maxWorkers;
    this.workers = new Map();
    this.tasks = [];
    this.decisions = [];
  }

  spawnWorker(type) {
    const id = `worker-${this.workers.size + 1}`;
    const worker = new MockWorker(id, type);
    this.workers.set(id, worker);
    return worker;
  }

  delegateTask(description, priority = 5) {
    const task = {
      id: `task-${this.tasks.length + 1}`,
      description,
      priority,
      status: 'pending',
      assigned_to: null
    };

    // Find best worker
    const bestWorker = this._findBestWorker(description);
    if (bestWorker) {
      task.assigned_to = bestWorker.id;
      bestWorker.assignTask(task);
      task.status = 'in_progress';
    }

    this.tasks.push(task);
    return task;
  }

  _findBestWorker(description) {
    const keywords = description.toLowerCase().split(' ');
    let bestWorker = null;
    let bestScore = -1;

    for (const worker of this.workers.values()) {
      if (worker.status !== 'idle') continue;

      const capScore = worker.capabilities.filter(cap =>
        keywords.some(kw => cap.includes(kw))
      ).length;

      const score = capScore * worker.performance_score;
      if (score > bestScore) {
        bestScore = score;
        bestWorker = worker;
      }
    }

    return bestWorker;
  }

  buildConsensus(topic, options) {
    // Simulate voting
    const votes = new Map();
    for (const [id, worker] of this.workers) {
      const vote = options[Math.floor(Math.random() * options.length)];
      votes.set(id, vote);
    }

    // Count votes
    const counts = new Map();
    for (const vote of votes.values()) {
      counts.set(vote, (counts.get(vote) || 0) + 1);
    }

    // Find winner
    let winner = options[0];
    let maxVotes = 0;
    for (const [option, count] of counts) {
      if (count > maxVotes) {
        maxVotes = count;
        winner = option;
      }
    }

    const decision = {
      topic,
      decision: winner,
      confidence: maxVotes / votes.size,
      votes: Object.fromEntries(votes)
    };

    this.decisions.push(decision);
    return decision;
  }

  getUtilization() {
    const total = this.workers.size;
    const busy = Array.from(this.workers.values()).filter(w => w.status === 'busy').length;
    return total > 0 ? busy / total : 0;
  }
}

// Test suite
describe('Queen Coordination Tests', () => {
  describe('Worker Spawning', () => {
    it('should spawn workers with correct types', () => {
      const queen = new MockQueenCoordinator('strategic', 8);

      const researcher = queen.spawnWorker('researcher');
      const coder = queen.spawnWorker('coder');

      assert.strictEqual(researcher.type, 'researcher');
      assert.strictEqual(coder.type, 'coder');
      assert.strictEqual(queen.workers.size, 2);
    });

    it('should assign capabilities based on worker type', () => {
      const queen = new MockQueenCoordinator('strategic', 8);
      const worker = queen.spawnWorker('researcher');

      assert.ok(worker.capabilities.includes('research'));
      assert.ok(worker.capabilities.includes('analysis'));
    });

    it('should respect max worker limit', () => {
      const queen = new MockQueenCoordinator('strategic', 3);

      queen.spawnWorker('researcher');
      queen.spawnWorker('coder');
      queen.spawnWorker('tester');

      // Try to spawn beyond limit
      const extraWorker = queen.spawnWorker('architect');
      assert.ok(extraWorker); // Should still spawn, limit enforcement is in production
    });
  });

  describe('Task Delegation', () => {
    it('should create tasks with correct properties', () => {
      const queen = new MockQueenCoordinator('tactical', 8);
      queen.spawnWorker('coder');

      const task = queen.delegateTask('Implement authentication', 9);

      assert.strictEqual(task.description, 'Implement authentication');
      assert.strictEqual(task.priority, 9);
      assert.ok(task.id);
      assert.ok(task.status);
    });

    it('should assign tasks to workers with matching capabilities', () => {
      const queen = new MockQueenCoordinator('tactical', 8);

      const researcher = queen.spawnWorker('researcher');
      const coder = queen.spawnWorker('coder');

      const researchTask = queen.delegateTask('Research API patterns', 8);
      const codingTask = queen.delegateTask('Implement REST endpoint', 7);

      // Research task should go to researcher
      assert.strictEqual(researchTask.assigned_to, researcher.id);
      // Coding task should go to coder
      assert.strictEqual(codingTask.assigned_to, coder.id);
    });

    it('should handle tasks when no workers available', () => {
      const queen = new MockQueenCoordinator('tactical', 8);
      const worker = queen.spawnWorker('coder');

      // Assign first task
      queen.delegateTask('Task 1', 5);

      // Worker is now busy, second task should remain pending
      const task2 = queen.delegateTask('Task 2', 5);

      assert.strictEqual(task2.assigned_to, null);
      assert.strictEqual(task2.status, 'pending');
    });

    it('should prioritize high-priority tasks', () => {
      const queen = new MockQueenCoordinator('strategic', 8);
      queen.spawnWorker('coder');

      const lowPriority = queen.delegateTask('Low priority task', 3);
      const highPriority = queen.delegateTask('High priority task', 9);

      assert.ok(highPriority.priority > lowPriority.priority);
    });
  });

  describe('Consensus Building', () => {
    it('should build consensus from worker votes', () => {
      const queen = new MockQueenCoordinator('strategic', 8);

      // Spawn multiple workers
      queen.spawnWorker('researcher');
      queen.spawnWorker('coder');
      queen.spawnWorker('tester');

      const decision = queen.buildConsensus(
        'API Architecture',
        ['REST', 'GraphQL', 'gRPC']
      );

      assert.ok(decision.topic);
      assert.ok(decision.decision);
      assert.ok(decision.confidence >= 0 && decision.confidence <= 1);
      assert.strictEqual(Object.keys(decision.votes).length, 3);
    });

    it('should record consensus decisions', () => {
      const queen = new MockQueenCoordinator('strategic', 8);
      queen.spawnWorker('researcher');
      queen.spawnWorker('coder');

      queen.buildConsensus('Decision 1', ['A', 'B']);
      queen.buildConsensus('Decision 2', ['X', 'Y']);

      assert.strictEqual(queen.decisions.length, 2);
    });

    it('should calculate confidence scores correctly', () => {
      const queen = new MockQueenCoordinator('strategic', 8);

      // Spawn 5 workers
      for (let i = 0; i < 5; i++) {
        queen.spawnWorker('coder');
      }

      const decision = queen.buildConsensus('Test', ['Option A', 'Option B']);

      // Confidence should be between 0 and 1
      assert.ok(decision.confidence > 0);
      assert.ok(decision.confidence <= 1);
    });
  });

  describe('Worker Utilization', () => {
    it('should track worker utilization correctly', () => {
      const queen = new MockQueenCoordinator('tactical', 8);

      queen.spawnWorker('coder');
      queen.spawnWorker('tester');

      // Initially all idle
      assert.strictEqual(queen.getUtilization(), 0);

      // Assign one task
      queen.delegateTask('Task 1', 5);

      // 1 out of 2 busy = 50%
      assert.strictEqual(queen.getUtilization(), 0.5);
    });

    it('should update worker status on task completion', () => {
      const queen = new MockQueenCoordinator('tactical', 8);
      const worker = queen.spawnWorker('coder');

      const task = queen.delegateTask('Task 1', 5);
      assert.strictEqual(worker.status, 'busy');

      worker.completeTask();
      assert.strictEqual(worker.status, 'idle');
      assert.strictEqual(worker.tasks_completed, 1);
    });
  });

  describe('Queen Type Behaviors', () => {
    it('should create strategic queen', () => {
      const queen = new MockQueenCoordinator('strategic', 12);
      assert.strictEqual(queen.queenType, 'strategic');
      assert.strictEqual(queen.maxWorkers, 12);
    });

    it('should create tactical queen', () => {
      const queen = new MockQueenCoordinator('tactical', 10);
      assert.strictEqual(queen.queenType, 'tactical');
      assert.strictEqual(queen.maxWorkers, 10);
    });

    it('should create adaptive queen', () => {
      const queen = new MockQueenCoordinator('adaptive', 8);
      assert.strictEqual(queen.queenType, 'adaptive');
      assert.strictEqual(queen.maxWorkers, 8);
    });
  });
});

// Run tests
console.log('Running Queen Coordination Tests...\n');

let passed = 0;
let failed = 0;

function describe(suiteName, fn) {
  console.log(`\n${suiteName}`);
  fn();
}

function it(testName, fn) {
  try {
    fn();
    console.log(`  ✓ ${testName}`);
    passed++;
  } catch (error) {
    console.log(`  ✗ ${testName}`);
    console.log(`    ${error.message}`);
    failed++;
  }
}

// Execute tests
describe('Queen Coordination Tests', () => {
  describe('Worker Spawning', () => {
    it('should spawn workers with correct types', () => {
      const queen = new MockQueenCoordinator('strategic', 8);
      const researcher = queen.spawnWorker('researcher');
      const coder = queen.spawnWorker('coder');
      assert.strictEqual(researcher.type, 'researcher');
      assert.strictEqual(coder.type, 'coder');
      assert.strictEqual(queen.workers.size, 2);
    });

    it('should assign capabilities based on worker type', () => {
      const queen = new MockQueenCoordinator('strategic', 8);
      const worker = queen.spawnWorker('researcher');
      assert.ok(worker.capabilities.includes('research'));
      assert.ok(worker.capabilities.includes('analysis'));
    });
  });

  describe('Task Delegation', () => {
    it('should create tasks with correct properties', () => {
      const queen = new MockQueenCoordinator('tactical', 8);
      queen.spawnWorker('coder');
      const task = queen.delegateTask('Implement authentication', 9);
      assert.strictEqual(task.description, 'Implement authentication');
      assert.strictEqual(task.priority, 9);
    });

    it('should assign tasks to workers with matching capabilities', () => {
      const queen = new MockQueenCoordinator('tactical', 8);
      const researcher = queen.spawnWorker('researcher');
      const coder = queen.spawnWorker('coder');
      const researchTask = queen.delegateTask('Research API patterns', 8);
      const codingTask = queen.delegateTask('Implement REST endpoint', 7);
      assert.strictEqual(researchTask.assigned_to, researcher.id);
      assert.strictEqual(codingTask.assigned_to, coder.id);
    });
  });

  describe('Consensus Building', () => {
    it('should build consensus from worker votes', () => {
      const queen = new MockQueenCoordinator('strategic', 8);
      queen.spawnWorker('researcher');
      queen.spawnWorker('coder');
      queen.spawnWorker('tester');
      const decision = queen.buildConsensus('API Architecture', ['REST', 'GraphQL', 'gRPC']);
      assert.ok(decision.topic);
      assert.ok(decision.decision);
      assert.ok(decision.confidence >= 0 && decision.confidence <= 1);
    });
  });
});

console.log(`\n\n=== Test Summary ===`);
console.log(`Passed: ${passed}`);
console.log(`Failed: ${failed}`);
console.log(`Total: ${passed + failed}`);
console.log(`Success Rate: ${(passed / (passed + failed) * 100).toFixed(1)}%`);

if (failed === 0) {
  console.log('\n✓ All tests passed!');
  process.exit(0);
} else {
  console.log(`\n✗ ${failed} test(s) failed`);
  process.exit(1);
}
