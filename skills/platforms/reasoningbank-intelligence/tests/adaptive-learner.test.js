#!/usr/bin/env node
/**
 * Unit tests for Adaptive Learner
 *
 * Tests learning algorithms, strategy selection, and edge cases.
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Import the module
const AdaptiveLearner = require('../resources/adaptive-learner.js');

describe('AdaptiveLearner', function() {
  let learner;

  beforeEach(function() {
    learner = new AdaptiveLearner({
      learningMode: 'thompson',
      explorationRate: 0.1,
      minAttempts: 3,
      contextAware: true
    });

    // Register test strategies
    ['strategy_a', 'strategy_b', 'strategy_c'].forEach(s => learner.registerStrategy(s));
  });

  describe('Strategy Registration', function() {
    it('should register new strategies', function() {
      assert.strictEqual(learner.strategies.size, 3);
      assert(learner.strategies.has('strategy_a'));
    });

    it('should not duplicate strategies', function() {
      learner.registerStrategy('strategy_a');
      assert.strictEqual(learner.strategies.size, 3);
    });
  });

  describe('Strategy Selection', function() {
    it('should explore under-explored strategies', function() {
      // Record outcomes for strategy_a and strategy_b
      for (let i = 0; i < 5; i++) {
        learner.recordOutcome('strategy_a', true);
        learner.recordOutcome('strategy_b', true);
      }

      // strategy_c has 0 attempts, should be selected for exploration
      const selected = learner.selectStrategy('test_task');
      assert.strictEqual(selected, 'strategy_c');
    });

    it('should select based on Thompson Sampling', function() {
      learner.learningMode = 'thompson';

      // Give strategy_a high success rate
      for (let i = 0; i < 10; i++) {
        learner.recordOutcome('strategy_a', true);
      }

      // Give strategy_b low success rate
      for (let i = 0; i < 10; i++) {
        learner.recordOutcome('strategy_b', false);
      }

      // Give strategy_c medium attempts
      for (let i = 0; i < 5; i++) {
        learner.recordOutcome('strategy_c', i < 3);
      }

      // Run multiple selections, strategy_a should be selected more often
      const selections = {};
      for (let i = 0; i < 100; i++) {
        const selected = learner.selectStrategy('test_task');
        selections[selected] = (selections[selected] || 0) + 1;
      }

      assert(selections['strategy_a'] > selections['strategy_b']);
    });

    it('should use UCB1 algorithm', function() {
      learner.learningMode = 'ucb';

      for (let i = 0; i < 10; i++) {
        learner.recordOutcome('strategy_a', true);
        learner.recordOutcome('strategy_b', false);
        learner.recordOutcome('strategy_c', i < 5);
      }

      const selected = learner.selectStrategy('test_task');
      assert(['strategy_a', 'strategy_c'].includes(selected));
    });

    it('should use epsilon-greedy algorithm', function() {
      learner.learningMode = 'epsilon-greedy';
      learner.explorationRate = 0.0; // No exploration for test

      for (let i = 0; i < 10; i++) {
        learner.recordOutcome('strategy_a', true);
        learner.recordOutcome('strategy_b', false);
        learner.recordOutcome('strategy_c', false);
      }

      const selected = learner.selectStrategy('test_task');
      assert.strictEqual(selected, 'strategy_a');
    });
  });

  describe('Outcome Recording', function() {
    it('should update strategy statistics', function() {
      learner.recordOutcome('strategy_a', true);
      learner.recordOutcome('strategy_a', false);

      const stats = learner.strategies.get('strategy_a');
      assert.strictEqual(stats.attempts, 2);
      assert.strictEqual(stats.successes, 1);
      assert.strictEqual(stats.failures, 1);
      assert.strictEqual(stats.successRate, 0.5);
    });

    it('should track recent outcomes', function() {
      for (let i = 0; i < 5; i++) {
        learner.recordOutcome('strategy_a', i < 3);
      }

      const stats = learner.strategies.get('strategy_a');
      assert.strictEqual(stats.recentOutcomes.length, 5);
      assert.strictEqual(stats.recentSuccessRate, 0.6);
    });

    it('should limit recent outcomes window', function() {
      for (let i = 0; i < 150; i++) {
        learner.recordOutcome('strategy_a', true);
      }

      const stats = learner.strategies.get('strategy_a');
      assert.strictEqual(stats.recentOutcomes.length, 100);
    });

    it('should track contextual performance', function() {
      const context1 = { language: 'python' };
      const context2 = { language: 'javascript' };

      learner.recordOutcome('strategy_a', true, null, context1);
      learner.recordOutcome('strategy_a', true, null, context1);
      learner.recordOutcome('strategy_a', false, null, context2);

      const stats = learner.strategies.get('strategy_a');
      assert.strictEqual(stats.getContextualSuccessRate(context1), 1.0);
      assert.strictEqual(stats.getContextualSuccessRate(context2), 0.0);
    });
  });

  describe('Context-Aware Selection', function() {
    it('should prefer strategies successful in similar contexts', function() {
      const context = { language: 'python', complexity: 'medium' };

      // Train strategy_a to be good for Python
      for (let i = 0; i < 10; i++) {
        learner.recordOutcome('strategy_a', true, null, context);
      }

      // Train strategy_b to be good overall but not for Python
      for (let i = 0; i < 10; i++) {
        learner.recordOutcome('strategy_b', true, null, { language: 'javascript' });
      }

      // Give both minimum exploration
      for (let i = 0; i < 5; i++) {
        learner.recordOutcome('strategy_c', false);
      }

      learner.contextAware = true;
      const selected = learner.selectStrategy('test_task', context);

      // Should prefer strategy_a due to contextual performance
      assert.strictEqual(selected, 'strategy_a');
    });
  });

  describe('Recommendations', function() {
    it('should return top K recommendations', function() {
      for (let i = 0; i < 10; i++) {
        learner.recordOutcome('strategy_a', true);
        learner.recordOutcome('strategy_b', i < 7);
        learner.recordOutcome('strategy_c', i < 5);
      }

      const recs = learner.getRecommendations('test_task', null, 2);
      assert.strictEqual(recs.length, 2);
      assert.strictEqual(recs[0].strategy, 'strategy_a');
      assert.strictEqual(recs[1].strategy, 'strategy_b');
    });

    it('should include confidence scores', function() {
      for (let i = 0; i < 50; i++) {
        learner.recordOutcome('strategy_a', true);
      }

      const recs = learner.getRecommendations('test_task', null, 1);
      assert(recs[0].confidence > 0);
      assert(recs[0].confidence <= 1);
    });
  });

  describe('Strategy Comparison', function() {
    it('should detect significant performance differences', function() {
      // strategy_a: 90% success (45/50)
      for (let i = 0; i < 50; i++) {
        learner.recordOutcome('strategy_a', i < 45);
      }

      // strategy_b: 50% success (25/50)
      for (let i = 0; i < 50; i++) {
        learner.recordOutcome('strategy_b', i < 25);
      }

      const comparison = learner.compareStrategies('strategy_a', 'strategy_b');
      assert.strictEqual(comparison.significant, true);
      assert(comparison.pValue < 0.05);
      assert.strictEqual(comparison.better, 'strategy_a');
    });

    it('should handle insufficient data', function() {
      learner.recordOutcome('strategy_a', true);
      learner.recordOutcome('strategy_b', false);

      const comparison = learner.compareStrategies('strategy_a', 'strategy_b');
      assert.strictEqual(comparison.significant, false);
      assert(comparison.reason.includes('Insufficient data'));
    });
  });

  describe('Export and Import', function() {
    it('should export learning state', function() {
      for (let i = 0; i < 10; i++) {
        learner.recordOutcome('strategy_a', true);
      }

      const tmpFile = path.join(os.tmpdir(), 'learner-test-export.json');
      learner.export(tmpFile);

      assert(fs.existsSync(tmpFile));
      const data = JSON.parse(fs.readFileSync(tmpFile, 'utf8'));
      assert.strictEqual(data.strategies.length, 3);

      fs.unlinkSync(tmpFile);
    });

    it('should import learning state', function() {
      for (let i = 0; i < 10; i++) {
        learner.recordOutcome('strategy_a', true);
      }

      const tmpFile = path.join(os.tmpdir(), 'learner-test-import.json');
      learner.export(tmpFile);

      const newLearner = new AdaptiveLearner();
      newLearner.import(tmpFile);

      assert.strictEqual(newLearner.strategies.size, 3);
      const stats = newLearner.strategies.get('strategy_a');
      assert.strictEqual(stats.attempts, 10);
      assert.strictEqual(stats.successes, 10);

      fs.unlinkSync(tmpFile);
    });
  });

  describe('Summary', function() {
    it('should provide learning summary', function() {
      for (let i = 0; i < 20; i++) {
        learner.recordOutcome('strategy_a', true);
        learner.recordOutcome('strategy_b', i < 10);
      }

      const summary = learner.getSummary();
      assert.strictEqual(summary.totalStrategies, 3);
      assert(summary.totalAttempts >= 40);
      assert.strictEqual(summary.learningMode, 'thompson');
      assert(Array.isArray(summary.strategies));
    });
  });

  describe('Edge Cases', function() {
    it('should handle zero strategies', function() {
      const emptyLearner = new AdaptiveLearner();
      assert.throws(() => emptyLearner.selectStrategy('test'), Error);
    });

    it('should handle all strategies with zero attempts', function() {
      // All strategies have 0 attempts, should select one for exploration
      const selected = learner.selectStrategy('test_task');
      assert(['strategy_a', 'strategy_b', 'strategy_c'].includes(selected));
    });

    it('should handle reward values', function() {
      learner.recordOutcome('strategy_a', true, 10);
      learner.recordOutcome('strategy_a', true, 5);

      const stats = learner.strategies.get('strategy_a');
      assert.strictEqual(stats.averageReward, 7.5);
    });
  });
});

// Run tests
if (require.main === module) {
  console.log('Running Adaptive Learner tests...\n');

  // Simple test runner
  const tests = [
    testStrategyRegistration,
    testStrategySelection,
    testOutcomeRecording,
    testRecommendations
  ];

  let passed = 0;
  let failed = 0;

  for (const test of tests) {
    try {
      test();
      console.log(`✓ ${test.name}`);
      passed++;
    } catch (error) {
      console.log(`✗ ${test.name}: ${error.message}`);
      failed++;
    }
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

function testStrategyRegistration() {
  const learner = new AdaptiveLearner();
  learner.registerStrategy('test_strategy');
  assert.strictEqual(learner.strategies.size, 1);
}

function testStrategySelection() {
  const learner = new AdaptiveLearner({ minAttempts: 2 });
  ['a', 'b', 'c'].forEach(s => learner.registerStrategy(s));

  for (let i = 0; i < 5; i++) {
    learner.recordOutcome('a', true);
    learner.recordOutcome('b', false);
  }

  const selected = learner.selectStrategy('test');
  assert(['a', 'b', 'c'].includes(selected));
}

function testOutcomeRecording() {
  const learner = new AdaptiveLearner();
  learner.registerStrategy('test');
  learner.recordOutcome('test', true);
  learner.recordOutcome('test', false);

  const stats = learner.strategies.get('test');
  assert.strictEqual(stats.attempts, 2);
  assert.strictEqual(stats.successRate, 0.5);
}

function testRecommendations() {
  const learner = new AdaptiveLearner();
  ['a', 'b'].forEach(s => learner.registerStrategy(s));

  for (let i = 0; i < 10; i++) {
    learner.recordOutcome('a', true);
    learner.recordOutcome('b', false);
  }

  const recs = learner.getRecommendations('test', null, 2);
  assert.strictEqual(recs.length, 2);
  assert.strictEqual(recs[0].strategy, 'a');
}

module.exports = AdaptiveLearner;
