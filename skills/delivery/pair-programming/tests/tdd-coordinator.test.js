#!/usr/bin/env node
/**
 * Unit tests for TDD Coordinator
 * Tests Red-Green-Refactor cycle management
 */

const assert = require('assert');
const { TDDCycle, TDDPhase, TestStatus } = require('../resources/tdd-coordinator.js');

describe('TDDCycle', () => {
  let tddCycle;

  beforeEach(() => {
    tddCycle = new TDDCycle('User Authentication', 'echo "test"');
  });

  describe('Initialization', () => {
    it('should initialize with correct feature name', () => {
      assert.strictEqual(tddCycle.feature, 'User Authentication');
    });

    it('should start in RED phase', () => {
      assert.strictEqual(tddCycle.currentPhase, TDDPhase.RED);
    });

    it('should initialize with empty cycles', () => {
      assert.strictEqual(tddCycle.cycles.length, 0);
    });

    it('should initialize with cycle number 1', () => {
      assert.strictEqual(tddCycle.currentCycle.cycleNumber, 1);
    });

    it('should use provided test command', () => {
      assert.strictEqual(tddCycle.testCommand, 'echo "test"');
    });
  });

  describe('Test Execution', () => {
    it('should run tests successfully', async () => {
      const result = await tddCycle.runTests();

      assert.ok(result);
      assert.ok('exitCode' in result);
      assert.ok('status' in result);
      assert.ok('timestamp' in result);
    });

    it('should capture test output', async () => {
      const result = await tddCycle.runTests();

      assert.ok('stdout' in result);
      assert.ok('stderr' in result);
    });

    it('should determine test status from exit code', async () => {
      const result = await tddCycle.runTests();

      if (result.exitCode === 0) {
        assert.strictEqual(result.status, TestStatus.PASSING);
      } else {
        assert.strictEqual(result.status, TestStatus.FAILING);
      }
    });
  });

  describe('Phase Management', () => {
    it('should track current phase', () => {
      assert.strictEqual(tddCycle.currentPhase, TDDPhase.RED);

      tddCycle.currentPhase = TDDPhase.GREEN;
      assert.strictEqual(tddCycle.currentPhase, TDDPhase.GREEN);
    });

    it('should record phase history', async () => {
      // Mock user input by providing a simple completion
      const originalWaitForUser = tddCycle.waitForUser;
      tddCycle.waitForUser = async () => Promise.resolve();

      // Start red phase
      const initialPhaseCount = tddCycle.currentCycle.phases.length;
      await tddCycle.startRedPhase('Test login functionality');

      assert.strictEqual(
        tddCycle.currentCycle.phases.length,
        initialPhaseCount + 1
      );

      const lastPhase = tddCycle.currentCycle.phases[tddCycle.currentCycle.phases.length - 1];
      assert.strictEqual(lastPhase.phase, TDDPhase.RED);
      assert.strictEqual(lastPhase.testDescription, 'Test login functionality');

      // Restore original method
      tddCycle.waitForUser = originalWaitForUser;
    });
  });

  describe('Cycle Completion', () => {
    it('should complete cycle and increment counter', async () => {
      const initialCycleNumber = tddCycle.currentCycle.cycleNumber;

      await tddCycle.completeCycle();

      assert.strictEqual(tddCycle.cycles.length, 1);
      assert.strictEqual(
        tddCycle.currentCycle.cycleNumber,
        initialCycleNumber + 1
      );
    });

    it('should record cycle duration', async () => {
      await tddCycle.completeCycle();

      const completedCycle = tddCycle.cycles[0];
      assert.ok('duration' in completedCycle);
      assert.ok(completedCycle.duration >= 0);
    });

    it('should track all completed cycles', async () => {
      await tddCycle.completeCycle();
      await tddCycle.completeCycle();
      await tddCycle.completeCycle();

      assert.strictEqual(tddCycle.cycles.length, 3);
    });
  });

  describe('Report Generation', () => {
    beforeEach(async () => {
      // Complete a few cycles for testing
      await tddCycle.completeCycle();
      await tddCycle.completeCycle();
    });

    it('should generate report with correct structure', () => {
      const report = tddCycle.generateReport();

      assert.ok('feature' in report);
      assert.ok('totalCycles' in report);
      assert.ok('cycles' in report);
      assert.ok('summary' in report);
    });

    it('should include feature name in report', () => {
      const report = tddCycle.generateReport();

      assert.strictEqual(report.feature, 'User Authentication');
    });

    it('should count total cycles correctly', () => {
      const report = tddCycle.generateReport();

      assert.strictEqual(report.totalCycles, 2);
    });

    it('should calculate summary statistics', () => {
      const report = tddCycle.generateReport();

      assert.ok('successfulCycles' in report.summary);
      assert.ok('failedCycles' in report.summary);
      assert.ok('averageCycleDuration' in report.summary);
    });

    it('should handle empty cycles gracefully', () => {
      const emptyCycle = new TDDCycle('Empty Feature', 'npm test');
      const report = emptyCycle.generateReport();

      assert.strictEqual(report.totalCycles, 0);
      assert.strictEqual(report.summary.averageCycleDuration, 0);
    });
  });

  describe('Session Export', () => {
    it('should export session to JSON file', () => {
      const fs = require('fs');
      const path = require('path');
      const tmpFile = path.join(__dirname, 'test-export.json');

      try {
        tddCycle.exportSession(tmpFile);
        assert.ok(fs.existsSync(tmpFile));

        const content = fs.readFileSync(tmpFile, 'utf8');
        const data = JSON.parse(content);

        assert.strictEqual(data.feature, 'User Authentication');
        assert.ok('cycles' in data);

        // Clean up
        fs.unlinkSync(tmpFile);
      } catch (error) {
        // Clean up on error
        if (fs.existsSync(tmpFile)) {
          fs.unlinkSync(tmpFile);
        }
        throw error;
      }
    });
  });

  describe('Test Status Detection', () => {
    it('should correctly identify passing tests', async () => {
      const passingCycle = new TDDCycle('Feature', 'exit 0');
      const result = await passingCycle.runTests();

      assert.strictEqual(result.status, TestStatus.PASSING);
    });

    it('should correctly identify failing tests', async () => {
      const failingCycle = new TDDCycle('Feature', 'exit 1');
      const result = await failingCycle.runTests();

      assert.strictEqual(result.status, TestStatus.FAILING);
    });
  });
});

// Run tests if this file is executed directly
if (require.main === module) {
  console.log('Running TDD Coordinator tests...\n');

  const tests = [
    { name: 'Initialization', fn: () => {
      const cycle = new TDDCycle('Test', 'echo test');
      assert.strictEqual(cycle.currentPhase, TDDPhase.RED);
      console.log('✓ Initialization tests passed');
    }},
    { name: 'Test Execution', fn: async () => {
      const cycle = new TDDCycle('Test', 'echo test');
      const result = await cycle.runTests();
      assert.ok(result);
      console.log('✓ Test execution tests passed');
    }},
    { name: 'Report Generation', fn: async () => {
      const cycle = new TDDCycle('Test', 'echo test');
      await cycle.completeCycle();
      const report = cycle.generateReport();
      assert.ok(report);
      console.log('✓ Report generation tests passed');
    }}
  ];

  (async () => {
    for (const test of tests) {
      try {
        await test.fn();
      } catch (error) {
        console.error(`✗ ${test.name} failed:`, error.message);
        process.exit(1);
      }
    }
    console.log('\n✅ All tests passed!');
  })();
}
