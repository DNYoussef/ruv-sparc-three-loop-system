#!/usr/bin/env node
/**
 * TDD Coordinator - Test-Driven Development Workflow Management
 * Coordinates Red-Green-Refactor cycles in pair programming
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// TDD Phases
const TDDPhase = {
  RED: 'red',       // Write failing test
  GREEN: 'green',   // Make test pass
  REFACTOR: 'refactor', // Improve code
  VERIFY: 'verify'  // Ensure tests still pass
};

// Test Status
const TestStatus = {
  PASSING: 'passing',
  FAILING: 'failing',
  ERROR: 'error',
  SKIPPED: 'skipped'
};

class TDDCycle {
  constructor(feature, testCommand = 'npm test') {
    this.feature = feature;
    this.testCommand = testCommand;
    this.currentPhase = TDDPhase.RED;
    this.cycles = [];
    this.currentCycle = {
      cycleNumber: 1,
      startTime: new Date(),
      phases: [],
      testResults: null,
      coverage: null
    };
  }

  async runTests() {
    return new Promise((resolve, reject) => {
      const [cmd, ...args] = this.testCommand.split(' ');
      const testProcess = spawn(cmd, args, {
        shell: true,
        stdio: 'pipe'
      });

      let stdout = '';
      let stderr = '';

      testProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      testProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      testProcess.on('close', (code) => {
        const result = {
          exitCode: code,
          stdout,
          stderr,
          status: code === 0 ? TestStatus.PASSING : TestStatus.FAILING,
          timestamp: new Date()
        };
        resolve(result);
      });

      testProcess.on('error', (error) => {
        reject(error);
      });
    });
  }

  async startRedPhase(testDescription) {
    console.log('\n🔴 RED PHASE: Write Failing Test');
    console.log(`Feature: ${this.feature}`);
    console.log(`Test: ${testDescription}`);

    this.currentPhase = TDDPhase.RED;
    this.currentCycle.phases.push({
      phase: TDDPhase.RED,
      startTime: new Date(),
      testDescription,
      expectedOutcome: TestStatus.FAILING
    });

    // Wait for user to write test
    console.log('\n📝 Write your failing test now...');
    console.log('Press ENTER when test is written');

    await this.waitForUser();

    // Run tests - should fail
    console.log('\n🧪 Running tests (expecting failure)...');
    const result = await this.runTests();

    this.currentCycle.testResults = result;
    this.currentCycle.phases[this.currentCycle.phases.length - 1].result = result;

    if (result.status === TestStatus.PASSING) {
      console.log('⚠️  WARNING: Tests are passing! You need a failing test first.');
      return false;
    }

    console.log('✅ Test is failing as expected. Ready for GREEN phase.');
    return true;
  }

  async startGreenPhase() {
    console.log('\n🟢 GREEN PHASE: Make Test Pass');
    console.log('Goal: Write minimal code to make the test pass');

    this.currentPhase = TDDPhase.GREEN;
    this.currentCycle.phases.push({
      phase: TDDPhase.GREEN,
      startTime: new Date(),
      expectedOutcome: TestStatus.PASSING
    });

    console.log('\n💻 Implement the feature now...');
    console.log('Press ENTER when implementation is complete');

    await this.waitForUser();

    // Run tests - should pass
    console.log('\n🧪 Running tests (expecting success)...');
    const result = await this.runTests();

    this.currentCycle.testResults = result;
    this.currentCycle.phases[this.currentCycle.phases.length - 1].result = result;

    if (result.status === TestStatus.FAILING) {
      console.log('❌ Tests still failing. Debug and try again.');
      console.log('Common issues:');
      console.log('  • Logic errors in implementation');
      console.log('  • Missing edge cases');
      console.log('  • Incorrect test expectations');
      return false;
    }

    console.log('✅ Tests passing! Ready for REFACTOR phase.');
    return true;
  }

  async startRefactorPhase() {
    console.log('\n🔧 REFACTOR PHASE: Improve Code Quality');
    console.log('Improve code while keeping tests green');
    console.log('\nRefactoring checklist:');
    console.log('  ✓ Remove duplication');
    console.log('  ✓ Improve naming');
    console.log('  ✓ Simplify logic');
    console.log('  ✓ Extract functions');
    console.log('  ✓ Add documentation');

    this.currentPhase = TDDPhase.REFACTOR;
    this.currentCycle.phases.push({
      phase: TDDPhase.REFACTOR,
      startTime: new Date(),
      expectedOutcome: TestStatus.PASSING
    });

    console.log('\n🎨 Refactor the code now...');
    console.log('Press ENTER when refactoring is complete');

    await this.waitForUser();

    // Run tests - should still pass
    console.log('\n🧪 Running tests (verifying refactor)...');
    const result = await this.runTests();

    this.currentCycle.testResults = result;
    this.currentCycle.phases[this.currentCycle.phases.length - 1].result = result;

    if (result.status === TestStatus.FAILING) {
      console.log('❌ Refactoring broke tests! Revert changes.');
      return false;
    }

    console.log('✅ Refactoring successful! Tests still green.');
    return true;
  }

  async completeCycle() {
    this.currentCycle.endTime = new Date();
    this.currentCycle.duration = this.currentCycle.endTime - this.currentCycle.startTime;

    this.cycles.push({ ...this.currentCycle });

    console.log('\n' + '='.repeat(60));
    console.log(`🎉 TDD Cycle ${this.currentCycle.cycleNumber} Complete!`);
    console.log('='.repeat(60));
    console.log(`Duration: ${Math.round(this.currentCycle.duration / 1000)}s`);
    console.log(`Phases completed: ${this.currentCycle.phases.length}`);

    // Prepare for next cycle
    this.currentCycle = {
      cycleNumber: this.currentCycle.cycleNumber + 1,
      startTime: new Date(),
      phases: [],
      testResults: null,
      coverage: null
    };
  }

  async runFullCycle(testDescription) {
    console.log('\n╔════════════════════════════════════════════════╗');
    console.log(`║  TDD CYCLE ${this.currentCycle.cycleNumber}: ${this.feature}${' '.repeat(Math.max(0, 30 - this.feature.length))}║`);
    console.log('╚════════════════════════════════════════════════╝');

    // RED: Write failing test
    const redSuccess = await this.startRedPhase(testDescription);
    if (!redSuccess) {
      console.log('\n❌ Red phase failed. Cycle aborted.');
      return false;
    }

    // GREEN: Make test pass
    const greenSuccess = await this.startGreenPhase();
    if (!greenSuccess) {
      console.log('\n❌ Green phase failed. Cycle aborted.');
      return false;
    }

    // REFACTOR: Improve code
    const refactorSuccess = await this.startRefactorPhase();
    if (!refactorSuccess) {
      console.log('\n⚠️  Refactor failed. Tests still green from previous phase.');
    }

    await this.completeCycle();
    return true;
  }

  generateReport() {
    const report = {
      feature: this.feature,
      totalCycles: this.cycles.length,
      totalDuration: this.cycles.reduce((sum, c) => sum + (c.duration || 0), 0),
      cycles: this.cycles.map(cycle => ({
        cycleNumber: cycle.cycleNumber,
        duration: cycle.duration,
        phasesCompleted: cycle.phases.length,
        success: cycle.testResults?.status === TestStatus.PASSING
      })),
      summary: {
        successfulCycles: this.cycles.filter(c => c.testResults?.status === TestStatus.PASSING).length,
        failedCycles: this.cycles.filter(c => c.testResults?.status === TestStatus.FAILING).length,
        averageCycleDuration: this.cycles.length > 0
          ? Math.round(this.cycles.reduce((sum, c) => sum + (c.duration || 0), 0) / this.cycles.length / 1000)
          : 0
      }
    };

    return report;
  }

  exportSession(filepath) {
    const report = this.generateReport();
    fs.writeFileSync(filepath, JSON.stringify(report, null, 2));
    console.log(`\n✅ TDD session exported to ${filepath}`);
  }

  waitForUser() {
    return new Promise((resolve) => {
      process.stdin.once('data', () => resolve());
    });
  }
}

// Interactive TDD Session
async function runInteractiveSession() {
  const readline = require('readline');
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const question = (prompt) => new Promise((resolve) => {
    rl.question(prompt, resolve);
  });

  console.log('\n╔════════════════════════════════════════════════╗');
  console.log('║     TDD Pair Programming Coordinator          ║');
  console.log('╚════════════════════════════════════════════════╝\n');

  const feature = await question('Feature name: ');
  const testCommand = await question('Test command (default: npm test): ') || 'npm test';

  const tdd = new TDDCycle(feature, testCommand);

  let continueSession = true;

  while (continueSession) {
    const testDescription = await question('\nTest description (or "quit" to exit): ');

    if (testDescription.toLowerCase() === 'quit') {
      continueSession = false;
      break;
    }

    await tdd.runFullCycle(testDescription);

    const another = await question('\nRun another cycle? (y/n): ');
    if (another.toLowerCase() !== 'y') {
      continueSession = false;
    }
  }

  // Generate and display report
  console.log('\n' + '='.repeat(60));
  console.log('📊 TDD SESSION REPORT');
  console.log('='.repeat(60));

  const report = tdd.generateReport();
  console.log(JSON.stringify(report, null, 2));

  const exportPath = await question('\nExport session? (filepath or "skip"): ');
  if (exportPath && exportPath.toLowerCase() !== 'skip') {
    tdd.exportSession(exportPath);
  }

  rl.close();
}

// CLI entry point
if (require.main === module) {
  runInteractiveSession().catch(console.error);
}

module.exports = { TDDCycle, TDDPhase, TestStatus };
