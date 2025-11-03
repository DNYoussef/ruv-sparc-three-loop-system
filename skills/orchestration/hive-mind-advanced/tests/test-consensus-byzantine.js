/**
 * Test Suite: Byzantine Consensus
 * Tests Byzantine fault tolerant consensus algorithm
 */

const assert = require('assert');
const crypto = require('crypto');

// Byzantine Consensus implementation for testing
class Vote {
  constructor(agentId, agentType, option) {
    this.agentId = agentId;
    this.agentType = agentType;
    this.option = option;
    this.timestamp = new Date().toISOString();
    this.signature = this._sign();
  }

  _sign() {
    const data = `${this.agentId}:${this.option}:${this.timestamp}`;
    return crypto.createHash('sha256').update(data).digest('hex');
  }

  verify() {
    const expectedSig = this._sign();
    return this.signature === expectedSig;
  }
}

class ByzantineConsensus {
  constructor(threshold = 2/3) {
    this.threshold = threshold;
    this.decisions = [];
  }

  buildConsensus(topic, votes, options) {
    // Verify votes
    const validVotes = votes.filter(v => v.verify());

    if (validVotes.length === 0) {
      throw new Error('No valid votes');
    }

    // Count votes
    const counts = {};
    for (const vote of validVotes) {
      counts[vote.option] = (counts[vote.option] || 0) + 1;
    }

    const totalVotes = validVotes.length;
    const requiredVotes = Math.ceil(totalVotes * this.threshold);

    // Check for supermajority
    for (const [option, count] of Object.entries(counts)) {
      if (count >= requiredVotes) {
        const decision = {
          topic,
          decision: option,
          confidence: count / totalVotes,
          supermajority: true,
          totalVotes,
          threshold: this.threshold
        };
        this.decisions.push(decision);
        return decision;
      }
    }

    // No supermajority - fallback to majority
    let winner = null;
    let maxVotes = 0;
    for (const [option, count] of Object.entries(counts)) {
      if (count > maxVotes) {
        maxVotes = count;
        winner = option;
      }
    }

    const decision = {
      topic,
      decision: winner,
      confidence: maxVotes / totalVotes,
      supermajority: false,
      totalVotes,
      threshold: this.threshold,
      warning: 'Supermajority not reached, using simple majority'
    };

    this.decisions.push(decision);
    return decision;
  }

  // Simulate Byzantine fault (up to 1/3 can fail)
  tolerateFaults(totalVoters, faultyVoters) {
    return faultyVoters <= Math.floor(totalVoters / 3);
  }
}

// Test Suite
describe('Byzantine Consensus Tests', () => {
  describe('Vote Verification', () => {
    it('should create valid vote signatures', () => {
      const vote = new Vote('worker-1', 'worker', 'Option A');

      assert.ok(vote.signature);
      assert.strictEqual(vote.signature.length, 64); // SHA-256 hex
      assert.ok(vote.verify());
    });

    it('should detect tampered votes', () => {
      const vote = new Vote('worker-1', 'worker', 'Option A');
      const originalSig = vote.signature;

      // Tamper with vote
      vote.option = 'Option B';

      assert.strictEqual(vote.signature, originalSig);
      assert.strictEqual(vote.verify(), false);
    });

    it('should verify multiple votes', () => {
      const votes = [
        new Vote('worker-1', 'worker', 'A'),
        new Vote('worker-2', 'worker', 'B'),
        new Vote('worker-3', 'worker', 'A')
      ];

      assert.ok(votes.every(v => v.verify()));
    });
  });

  describe('Supermajority Requirements', () => {
    it('should achieve consensus with 2/3 supermajority', () => {
      const consensus = new ByzantineConsensus(2/3);

      const votes = [
        new Vote('worker-1', 'worker', 'Option A'),
        new Vote('worker-2', 'worker', 'Option A'),
        new Vote('worker-3', 'worker', 'Option A'),
        new Vote('worker-4', 'worker', 'Option A'),
        new Vote('worker-5', 'worker', 'Option B'),
        new Vote('worker-6', 'worker', 'Option B')
      ];

      const decision = consensus.buildConsensus('Test', votes, ['Option A', 'Option B']);

      assert.strictEqual(decision.decision, 'Option A');
      assert.strictEqual(decision.supermajority, true);
      assert.ok(decision.confidence >= 2/3);
    });

    it('should fallback to majority when no supermajority', () => {
      const consensus = new ByzantineConsensus(2/3);

      const votes = [
        new Vote('worker-1', 'worker', 'Option A'),
        new Vote('worker-2', 'worker', 'Option A'),
        new Vote('worker-3', 'worker', 'Option A'),
        new Vote('worker-4', 'worker', 'Option B'),
        new Vote('worker-5', 'worker', 'Option B'),
        new Vote('worker-6', 'worker', 'Option C')
      ];

      const decision = consensus.buildConsensus('Test', votes, ['Option A', 'Option B', 'Option C']);

      assert.strictEqual(decision.supermajority, false);
      assert.ok(decision.warning);
      assert.strictEqual(decision.decision, 'Option A'); // Most votes
    });

    it('should calculate confidence correctly', () => {
      const consensus = new ByzantineConsensus(2/3);

      const votes = [
        new Vote('w1', 'worker', 'A'),
        new Vote('w2', 'worker', 'A'),
        new Vote('w3', 'worker', 'A'),
        new Vote('w4', 'worker', 'A'),
        new Vote('w5', 'worker', 'B')
      ];

      const decision = consensus.buildConsensus('Test', votes, ['A', 'B']);

      assert.strictEqual(decision.confidence, 4/5); // 80%
    });
  });

  describe('Fault Tolerance', () => {
    it('should tolerate up to 1/3 faulty voters', () => {
      const consensus = new ByzantineConsensus(2/3);

      assert.ok(consensus.tolerateFaults(9, 3));  // 3/9 = 1/3
      assert.ok(consensus.tolerateFaults(12, 4)); // 4/12 = 1/3
      assert.ok(consensus.tolerateFaults(6, 2));  // 2/6 = 1/3
    });

    it('should not tolerate more than 1/3 faulty voters', () => {
      const consensus = new ByzantineConsensus(2/3);

      assert.strictEqual(consensus.tolerateFaults(9, 4), false);  // 4/9 > 1/3
      assert.strictEqual(consensus.tolerateFaults(6, 3), false);  // 3/6 > 1/3
    });

    it('should reach consensus with faulty voters (within limit)', () => {
      const consensus = new ByzantineConsensus(2/3);

      // 9 total voters, 3 faulty (voting randomly/incorrectly)
      const votes = [
        // 6 correct votes for Option A
        new Vote('w1', 'worker', 'Option A'),
        new Vote('w2', 'worker', 'Option A'),
        new Vote('w3', 'worker', 'Option A'),
        new Vote('w4', 'worker', 'Option A'),
        new Vote('w5', 'worker', 'Option A'),
        new Vote('w6', 'worker', 'Option A'),
        // 3 faulty votes
        new Vote('w7', 'worker', 'Option B'),
        new Vote('w8', 'worker', 'Option C'),
        new Vote('w9', 'worker', 'Option B')
      ];

      const decision = consensus.buildConsensus('Test', votes, ['Option A', 'Option B', 'Option C']);

      assert.strictEqual(decision.decision, 'Option A');
      assert.strictEqual(decision.supermajority, true);
      assert.ok(decision.confidence >= 2/3);
    });
  });

  describe('Edge Cases', () => {
    it('should handle single voter', () => {
      const consensus = new ByzantineConsensus(2/3);
      const votes = [new Vote('w1', 'worker', 'Only Option')];

      const decision = consensus.buildConsensus('Test', votes, ['Only Option']);

      assert.strictEqual(decision.decision, 'Only Option');
      assert.strictEqual(decision.confidence, 1.0);
    });

    it('should handle unanimous vote', () => {
      const consensus = new ByzantineConsensus(2/3);

      const votes = [
        new Vote('w1', 'worker', 'Unanimous'),
        new Vote('w2', 'worker', 'Unanimous'),
        new Vote('w3', 'worker', 'Unanimous'),
        new Vote('w4', 'worker', 'Unanimous'),
        new Vote('w5', 'worker', 'Unanimous')
      ];

      const decision = consensus.buildConsensus('Test', votes, ['Unanimous']);

      assert.strictEqual(decision.decision, 'Unanimous');
      assert.strictEqual(decision.confidence, 1.0);
      assert.strictEqual(decision.supermajority, true);
    });

    it('should handle tie votes', () => {
      const consensus = new ByzantineConsensus(2/3);

      const votes = [
        new Vote('w1', 'worker', 'A'),
        new Vote('w2', 'worker', 'A'),
        new Vote('w3', 'worker', 'B'),
        new Vote('w4', 'worker', 'B')
      ];

      const decision = consensus.buildConsensus('Test', votes, ['A', 'B']);

      assert.ok(['A', 'B'].includes(decision.decision));
      assert.strictEqual(decision.supermajority, false);
    });

    it('should throw error on no valid votes', () => {
      const consensus = new ByzantineConsensus(2/3);

      assert.throws(() => {
        consensus.buildConsensus('Test', [], ['A', 'B']);
      }, /No valid votes/);
    });
  });

  describe('Decision Tracking', () => {
    it('should record all consensus decisions', () => {
      const consensus = new ByzantineConsensus(2/3);

      consensus.buildConsensus('Decision 1', [
        new Vote('w1', 'worker', 'A'),
        new Vote('w2', 'worker', 'A'),
        new Vote('w3', 'worker', 'A')
      ], ['A', 'B']);

      consensus.buildConsensus('Decision 2', [
        new Vote('w1', 'worker', 'X'),
        new Vote('w2', 'worker', 'X'),
        new Vote('w3', 'worker', 'Y')
      ], ['X', 'Y']);

      assert.strictEqual(consensus.decisions.length, 2);
      assert.strictEqual(consensus.decisions[0].topic, 'Decision 1');
      assert.strictEqual(consensus.decisions[1].topic, 'Decision 2');
    });
  });
});

// Run tests
console.log('Running Byzantine Consensus Tests...\n');

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

// Execute all tests
describe('Byzantine Consensus Tests', () => {
  describe('Vote Verification', () => {
    it('should create valid vote signatures', () => {
      const vote = new Vote('worker-1', 'worker', 'Option A');
      assert.ok(vote.signature);
      assert.strictEqual(vote.signature.length, 64);
      assert.ok(vote.verify());
    });

    it('should detect tampered votes', () => {
      const vote = new Vote('worker-1', 'worker', 'Option A');
      const originalSig = vote.signature;
      vote.option = 'Option B';
      assert.strictEqual(vote.signature, originalSig);
      assert.strictEqual(vote.verify(), false);
    });
  });

  describe('Supermajority Requirements', () => {
    it('should achieve consensus with 2/3 supermajority', () => {
      const consensus = new ByzantineConsensus(2/3);
      const votes = [
        new Vote('w1', 'worker', 'Option A'),
        new Vote('w2', 'worker', 'Option A'),
        new Vote('w3', 'worker', 'Option A'),
        new Vote('w4', 'worker', 'Option A'),
        new Vote('w5', 'worker', 'Option B'),
        new Vote('w6', 'worker', 'Option B')
      ];
      const decision = consensus.buildConsensus('Test', votes, ['Option A', 'Option B']);
      assert.strictEqual(decision.decision, 'Option A');
      assert.strictEqual(decision.supermajority, true);
      assert.ok(decision.confidence >= 2/3);
    });
  });

  describe('Fault Tolerance', () => {
    it('should tolerate up to 1/3 faulty voters', () => {
      const consensus = new ByzantineConsensus(2/3);
      assert.ok(consensus.tolerateFaults(9, 3));
      assert.ok(consensus.tolerateFaults(12, 4));
      assert.ok(consensus.tolerateFaults(6, 2));
    });

    it('should not tolerate more than 1/3 faulty voters', () => {
      const consensus = new ByzantineConsensus(2/3);
      assert.strictEqual(consensus.tolerateFaults(9, 4), false);
      assert.strictEqual(consensus.tolerateFaults(6, 3), false);
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
