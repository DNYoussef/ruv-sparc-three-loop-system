#!/usr/bin/env node
/**
 * Test Runner - Fast parallel test execution
 * Part of quick-quality-check Enhanced tier resources
 */

const { spawn, spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Configuration
const CONFIG = {
  timeout: 30000, // 30 seconds max
  parallel: 4,
  quickMode: true,
  frameworks: ['jest', 'mocha', 'vitest', 'ava'],
};

class TestRunner {
  constructor(targetPath, options = {}) {
    this.targetPath = path.resolve(targetPath);
    this.options = { ...CONFIG, ...options };
    this.results = {
      framework: null,
      tests_run: 0,
      tests_passed: 0,
      tests_failed: 0,
      tests_skipped: 0,
      execution_time: 0,
      coverage: null,
      failures: [],
    };
  }

  /**
   * Detect which test framework is being used
   */
  detectFramework() {
    console.error('[INFO] Detecting test framework...');

    const packageJsonPath = this.findPackageJson();
    if (!packageJsonPath) {
      console.error('[WARN] No package.json found');
      return null;
    }

    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    const dependencies = {
      ...packageJson.dependencies,
      ...packageJson.devDependencies,
    };

    for (const framework of this.options.frameworks) {
      if (dependencies[framework]) {
        console.error(`[INFO] Detected framework: ${framework}`);
        return framework;
      }
    }

    console.error('[WARN] No recognized test framework found');
    return null;
  }

  /**
   * Find package.json in current directory or parents
   */
  findPackageJson() {
    let currentPath = this.targetPath;

    while (currentPath !== path.parse(currentPath).root) {
      const packageJsonPath = path.join(currentPath, 'package.json');
      if (fs.existsSync(packageJsonPath)) {
        return packageJsonPath;
      }
      currentPath = path.dirname(currentPath);
    }

    return null;
  }

  /**
   * Run tests using detected framework
   */
  async run() {
    const startTime = Date.now();
    const framework = this.detectFramework();

    if (!framework) {
      console.error('[ERROR] Cannot run tests without a test framework');
      return this.results;
    }

    this.results.framework = framework;

    try {
      switch (framework) {
        case 'jest':
          await this.runJest();
          break;
        case 'mocha':
          await this.runMocha();
          break;
        case 'vitest':
          await this.runVitest();
          break;
        case 'ava':
          await this.runAva();
          break;
        default:
          throw new Error(`Unsupported framework: ${framework}`);
      }
    } catch (error) {
      console.error(`[ERROR] Test execution failed: ${error.message}`);
      this.results.error = error.message;
    }

    this.results.execution_time = (Date.now() - startTime) / 1000;
    return this.results;
  }

  /**
   * Run Jest tests
   */
  async runJest() {
    console.error('[INFO] Running Jest tests...');

    const args = [
      '--json',
      '--no-coverage', // Quick mode skips coverage
      '--maxWorkers=4',
      '--silent',
    ];

    if (this.options.quickMode) {
      args.push('--testPathPattern=.*\\.test\\.(js|ts)$');
    }

    const result = await this.executeCommand('jest', args);

    if (result.stdout) {
      try {
        const jestResults = JSON.parse(result.stdout);

        this.results.tests_run = jestResults.numTotalTests;
        this.results.tests_passed = jestResults.numPassedTests;
        this.results.tests_failed = jestResults.numFailedTests;
        this.results.tests_skipped = jestResults.numPendingTests;

        if (jestResults.testResults) {
          this.results.failures = jestResults.testResults
            .filter(test => test.status === 'failed')
            .map(test => ({
              file: test.name,
              message: test.message,
            }));
        }
      } catch (e) {
        console.error('[WARN] Could not parse Jest output');
      }
    }
  }

  /**
   * Run Mocha tests
   */
  async runMocha() {
    console.error('[INFO] Running Mocha tests...');

    const args = [
      '--reporter', 'json',
      '--timeout', this.options.timeout.toString(),
      '--parallel',
    ];

    if (this.options.quickMode) {
      args.push('--grep', '.*');
    }

    const result = await this.executeCommand('mocha', args);

    if (result.stdout) {
      try {
        const mochaResults = JSON.parse(result.stdout);

        this.results.tests_run = mochaResults.stats.tests;
        this.results.tests_passed = mochaResults.stats.passes;
        this.results.tests_failed = mochaResults.stats.failures;
        this.results.tests_skipped = mochaResults.stats.pending;

        if (mochaResults.failures) {
          this.results.failures = mochaResults.failures.map(f => ({
            file: f.file,
            title: f.fullTitle,
            message: f.err.message,
          }));
        }
      } catch (e) {
        console.error('[WARN] Could not parse Mocha output');
      }
    }
  }

  /**
   * Run Vitest tests
   */
  async runVitest() {
    console.error('[INFO] Running Vitest tests...');

    const args = [
      '--reporter=json',
      '--run',
      '--threads=true',
    ];

    if (this.options.quickMode) {
      args.push('--changed');
    }

    const result = await this.executeCommand('vitest', args);

    if (result.stdout) {
      try {
        const vitestResults = JSON.parse(result.stdout);

        this.results.tests_run = vitestResults.numTotalTests;
        this.results.tests_passed = vitestResults.numPassedTests;
        this.results.tests_failed = vitestResults.numFailedTests;
        this.results.tests_skipped = vitestResults.numPendingTests;
      } catch (e) {
        console.error('[WARN] Could not parse Vitest output');
      }
    }
  }

  /**
   * Run AVA tests
   */
  async runAva() {
    console.error('[INFO] Running AVA tests...');

    const args = [
      '--tap',
      '--concurrency=4',
    ];

    const result = await this.executeCommand('ava', args);

    // Parse TAP output
    if (result.stdout) {
      const lines = result.stdout.split('\n');
      let passed = 0, failed = 0;

      for (const line of lines) {
        if (line.startsWith('ok ')) passed++;
        if (line.startsWith('not ok ')) failed++;
      }

      this.results.tests_run = passed + failed;
      this.results.tests_passed = passed;
      this.results.tests_failed = failed;
    }
  }

  /**
   * Execute command and capture output
   */
  executeCommand(command, args) {
    return new Promise((resolve) => {
      const cmd = spawnSync(command, args, {
        cwd: path.dirname(this.findPackageJson()),
        encoding: 'utf8',
        timeout: this.options.timeout,
      });

      resolve({
        stdout: cmd.stdout,
        stderr: cmd.stderr,
        exitCode: cmd.status,
      });
    });
  }
}

// Main execution
async function main() {
  const targetPath = process.argv[2] || '.';
  const outputFile = process.argv[3];

  console.error(`[INFO] Starting test run for: ${targetPath}`);

  const runner = new TestRunner(targetPath, {
    quickMode: true,
    parallel: 4,
  });

  const results = await runner.run();

  // Output results
  const output = JSON.stringify(results, null, 2);

  if (outputFile) {
    fs.writeFileSync(outputFile, output);
    console.error(`[INFO] Results written to ${outputFile}`);
  } else {
    console.log(output);
  }

  // Exit with error code if tests failed
  if (results.tests_failed > 0) {
    console.error(`[ERROR] ${results.tests_failed} tests failed`);
    process.exit(1);
  } else {
    console.error('[INFO] All tests passed');
    process.exit(0);
  }
}

if (require.main === module) {
  main().catch(error => {
    console.error(`[ERROR] ${error.message}`);
    process.exit(1);
  });
}

module.exports = TestRunner;
