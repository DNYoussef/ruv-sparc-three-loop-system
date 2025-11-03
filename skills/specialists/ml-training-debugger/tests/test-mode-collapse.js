#!/usr/bin/env node
/**
 * Test suite for mode collapse detection
 *
 * Tests overfitting-detector.js functionality for detecting mode collapse
 * and degenerate model outputs in ML training.
 */

const assert = require('assert');
const path = require('path');

// Mock OverfittingDetector if not available
let OverfittingDetector;
try {
  OverfittingDetector = require('../resources/scripts/overfitting-detector.js');
} catch (e) {
  console.warn('⚠️ Could not load OverfittingDetector - using mock');
  OverfittingDetector = class {
    constructor(options = {}) {
      this.gapThreshold = options.gapThreshold || 0.1;
    }

    calculateGap(train, val) {
      return train.map((t, i) => ({
        epoch: i,
        absolute: Math.abs(t - val[i]),
        relative: Math.abs(t - val[i]) / (Math.abs(val[i]) + 1e-8),
        train: t,
        val: val[i]
      }));
    }

    detectOverfitting(gaps) {
      return {
        isOverfitting: false,
        severity: 'none',
        onsetEpoch: null,
        maxGap: 0,
        avgGap: 0,
        trends: []
      };
    }

    analyze(train, val) {
      return {
        gaps: this.calculateGap(train, val),
        overfitting: this.detectOverfitting([]),
        earlyStopping: { shouldStop: false, bestEpoch: 0 },
        recommendations: [],
        summary: {
          totalEpochs: train.length,
          finalTrainMetric: train[train.length - 1],
          finalValMetric: val[val.length - 1]
        }
      };
    }
  };
}

// Test suite
class TestModeCollapse {
  constructor() {
    this.passed = 0;
    this.failed = 0;
    this.tests = [];
  }

  test(name, fn) {
    this.tests.push({ name, fn });
  }

  async run() {
    console.log('🧪 Running Mode Collapse Detection Tests\n');

    for (const { name, fn } of this.tests) {
      try {
        await fn();
        console.log(`✅ ${name}`);
        this.passed++;
      } catch (error) {
        console.log(`❌ ${name}`);
        console.log(`   ${error.message}`);
        this.failed++;
      }
    }

    console.log('\n' + '='.repeat(60));
    console.log(`Tests: ${this.passed} passed, ${this.failed} failed, ${this.tests.length} total`);
    console.log('='.repeat(60));

    return this.failed === 0;
  }
}

// Test cases
const suite = new TestModeCollapse();

suite.test('Detects mode collapse via perfect train, poor val', () => {
  const detector = new OverfittingDetector({ gapThreshold: 0.1 });

  // Perfect training performance (mode collapse to single output)
  const trainMetrics = Array(20).fill(0).map((_, i) =>
    i < 5 ? 2.5 - i * 0.4 : 0.01  // Rapid decrease to near-zero
  );

  // Validation remains poor (model not generalizing)
  const valMetrics = Array(20).fill(0).map((_, i) =>
    2.3 - i * 0.02  // Very slow decrease
  );

  const analysis = detector.analyze(trainMetrics, valMetrics);

  assert(analysis.overfitting.isOverfitting,
    'Should detect overfitting with perfect train and poor val');
  assert(analysis.overfitting.severity !== 'none',
    'Severity should not be none');
  assert.strictEqual(typeof analysis.overfitting.maxGap, 'number',
    'Should have numeric max gap');
});

suite.test('Detects sudden train/val divergence', () => {
  const detector = new OverfittingDetector({ gapThreshold: 0.15 });

  // Both decrease initially, then train continues but val plateaus
  const trainMetrics = Array(30).fill(0).map((_, i) =>
    3.0 * Math.exp(-0.15 * i)  // Exponential decrease
  );

  const valMetrics = Array(30).fill(0).map((_, i) => {
    if (i < 10) return 3.0 * Math.exp(-0.15 * i);  // Match train
    return 1.2 + Math.random() * 0.1;  // Plateau
  });

  const analysis = detector.analyze(trainMetrics, valMetrics);

  assert(analysis.overfitting.isOverfitting,
    'Should detect divergence when val plateaus');
  assert(analysis.overfitting.onsetEpoch !== null,
    'Should identify onset epoch');
  assert(analysis.overfitting.onsetEpoch >= 8 && analysis.overfitting.onsetEpoch <= 12,
    'Onset should be around epoch 10');
});

suite.test('Does not flag healthy convergence', () => {
  const detector = new OverfittingDetector({ gapThreshold: 0.1 });

  // Both train and val decrease similarly
  const trainMetrics = Array(20).fill(0).map((_, i) =>
    4.0 * Math.exp(-0.12 * i)
  );

  const valMetrics = Array(20).fill(0).map((_, i) =>
    4.2 * Math.exp(-0.11 * i)  // Slightly higher but similar trend
  );

  const analysis = detector.analyze(trainMetrics, valMetrics);

  // Small gap is acceptable
  assert(analysis.overfitting.maxGap < 0.15,
    'Gap should be small for healthy convergence');
  assert(analysis.overfitting.severity === 'none' ||
         analysis.overfitting.severity === 'mild',
    'Should not flag severe overfitting');
});

suite.test('Identifies early stopping opportunity', () => {
  const detector = new OverfittingDetector({
    gapThreshold: 0.1,
    earlyStoppingPatience: 5
  });

  // Val loss improves then plateaus
  const valMetrics = [
    4.5, 3.8, 3.2, 2.9, 2.7, 2.6,  // Improvement
    2.62, 2.63, 2.61, 2.64, 2.62, 2.63,  // Plateau (6 epochs)
    2.61, 2.65, 2.62
  ];

  const trainMetrics = valMetrics.map((v, i) =>
    v * 0.95  // Train slightly better
  );

  const analysis = detector.analyze(trainMetrics, valMetrics);

  assert(analysis.earlyStopping.shouldStop,
    'Should recommend early stopping after plateau');
  assert(analysis.earlyStopping.bestEpoch === 5,
    'Best epoch should be 5 (before plateau)');
  assert(analysis.earlyStopping.bestMetric === 2.6,
    'Best metric should be 2.6');
});

suite.test('Handles noisy validation metrics', () => {
  const detector = new OverfittingDetector({ gapThreshold: 0.15 });

  // Add noise to validation
  const trainMetrics = Array(25).fill(0).map((_, i) =>
    3.5 * Math.exp(-0.1 * i)
  );

  const valMetrics = trainMetrics.map((v, i) =>
    v * 1.05 + (Math.random() - 0.5) * 0.2  // Add noise
  );

  const analysis = detector.analyze(trainMetrics, valMetrics);

  // Should still complete analysis without errors
  assert(typeof analysis.summary.finalGap === 'number',
    'Should calculate final gap despite noise');
  assert(Array.isArray(analysis.recommendations),
    'Should provide recommendations');
  assert(analysis.recommendations.length > 0,
    'Should have at least one recommendation');
});

suite.test('Detects severe overfitting (>50% gap)', () => {
  const detector = new OverfittingDetector({ gapThreshold: 0.1 });

  // Extreme overfitting scenario
  const trainMetrics = [4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01];
  const valMetrics = [3.9, 3.7, 3.5, 3.4, 3.3, 3.2, 3.15, 3.1, 3.05, 3.0];

  const analysis = detector.analyze(trainMetrics, valMetrics);

  assert(analysis.overfitting.isOverfitting,
    'Should detect severe overfitting');
  assert(analysis.overfitting.severity === 'severe',
    'Severity should be severe');
  assert(analysis.overfitting.maxGap > 0.5,
    'Max gap should exceed 50%');

  // Should recommend strong regularization
  const recText = analysis.recommendations.map(r => r.message).join(' ');
  assert(recText.toLowerCase().includes('regularization'),
    'Should recommend regularization');
});

suite.test('Calculates trend direction correctly', () => {
  const detector = new OverfittingDetector({ gapThreshold: 0.1 });

  // Increasing gap over time
  const trainMetrics = Array(20).fill(0).map((_, i) =>
    3.0 * Math.exp(-0.2 * i)  // Fast decrease
  );

  const valMetrics = Array(20).fill(0).map((_, i) =>
    3.0 * Math.exp(-0.05 * i)  // Slow decrease
  );

  const analysis = detector.analyze(trainMetrics, valMetrics);

  // Gap should be increasing
  const gaps = analysis.gaps.map(g => g.relative);
  const recentGaps = gaps.slice(-5);
  const earlierGaps = gaps.slice(5, 10);

  const recentAvg = recentGaps.reduce((a, b) => a + b) / recentGaps.length;
  const earlierAvg = earlierGaps.reduce((a, b) => a + b) / earlierGaps.length;

  assert(recentAvg > earlierAvg,
    'Recent gap should be larger than earlier gap');
});

suite.test('Edge case: Single epoch', () => {
  const detector = new OverfittingDetector();

  const trainMetrics = [2.5];
  const valMetrics = [2.7];

  // Should handle without crashing
  const analysis = detector.analyze(trainMetrics, valMetrics);

  assert(analysis.summary.totalEpochs === 1,
    'Should report 1 epoch');
  assert(!analysis.earlyStopping.shouldStop,
    'Should not recommend stopping after 1 epoch');
});

suite.test('Edge case: Identical train and val', () => {
  const detector = new OverfittingDetector({ gapThreshold: 0.1 });

  // Perfect match (unlikely in practice)
  const metrics = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5];
  const trainMetrics = [...metrics];
  const valMetrics = [...metrics];

  const analysis = detector.analyze(trainMetrics, valMetrics);

  assert(!analysis.overfitting.isOverfitting,
    'Should not flag overfitting when metrics match');
  assert(analysis.overfitting.maxGap < 0.01,
    'Gap should be near zero');
});

suite.test('Provides actionable recommendations', () => {
  const detector = new OverfittingDetector({ gapThreshold: 0.1 });

  // Moderate overfitting
  const trainMetrics = [3.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.7];
  const valMetrics = [2.9, 2.2, 2.0, 1.95, 1.93, 1.92, 1.91];

  const analysis = detector.analyze(trainMetrics, valMetrics);

  assert(analysis.recommendations.length > 0,
    'Should provide recommendations');

  // Check for specific recommendation types
  const categories = analysis.recommendations.map(r => r.category);
  assert(categories.length > 0,
    'Recommendations should have categories');

  // Should suggest concrete actions
  const hasActions = analysis.recommendations.some(r =>
    r.actions && r.actions.length > 0
  );
  assert(hasActions,
    'At least one recommendation should have concrete actions');
});

// Run tests
if (require.main === module) {
  suite.run().then(success => {
    process.exit(success ? 0 : 1);
  }).catch(error => {
    console.error('Test suite error:', error);
    process.exit(1);
  });
}

module.exports = { TestModeCollapse, suite };
