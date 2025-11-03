/**
 * Test Suite for Binary Search Debugger
 *
 * Tests the binary search debugging script functionality including:
 * - Commit bisection
 * - Code line bisection
 * - Error handling
 * - Report generation
 */

const assert = require('assert');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const BinarySearchDebugger = require('../resources/scripts/binary-search-debug');

describe('BinarySearchDebugger', function() {
  this.timeout(30000); // Increase timeout for git operations

  const testDir = path.join(__dirname, '.test-temp');
  const testRepo = path.join(testDir, 'test-repo');

  before(function() {
    // Create test directory
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }
    fs.mkdirSync(testDir, { recursive: true });
  });

  after(function() {
    // Cleanup
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }
  });

  describe('Constructor', function() {
    it('should initialize with default options', function() {
      const debugger = new BinarySearchDebugger({ test: 'npm test' });

      assert.strictEqual(debugger.mode, 'commits');
      assert.strictEqual(debugger.testCommand, 'npm test');
      assert(Array.isArray(debugger.results));
      assert.strictEqual(debugger.results.length, 0);
    });

    it('should accept custom mode', function() {
      const debugger = new BinarySearchDebugger({
        mode: 'code',
        test: 'npm test',
        file: 'test.js'
      });

      assert.strictEqual(debugger.mode, 'code');
      assert.strictEqual(debugger.filePath, 'test.js');
    });
  });

  describe('runTest', function() {
    it('should return success for passing test', function() {
      const debugger = new BinarySearchDebugger({
        test: 'node -e "process.exit(0)"'
      });

      const result = debugger.runTest();

      assert.strictEqual(result.passed, true);
      assert.strictEqual(result.error, null);
    });

    it('should return failure for failing test', function() {
      const debugger = new BinarySearchDebugger({
        test: 'node -e "process.exit(1)"'
      });

      const result = debugger.runTest();

      assert.strictEqual(result.passed, false);
      assert(result.error !== null);
    });

    it('should handle timeout', function() {
      const debugger = new BinarySearchDebugger({
        test: 'node -e "setTimeout(() => {}, 60000)"'
      });

      const result = debugger.runTest();

      assert.strictEqual(result.passed, false);
      assert(result.error.includes('timeout') || result.error.includes('ETIMEDOUT'));
    });
  });

  describe('searchCode', function() {
    it('should find buggy line in simple case', async function() {
      const testFile = path.join(testDir, 'test-code.js');

      // Create test file with intentional bug on line 5
      const code = [
        '// Line 1',
        'function add(a, b) {',
        '  // Line 3',
        '  // Line 4',
        '  return a - b; // BUG: should be a + b',
        '}',
        '// Line 7',
        'module.exports = add;'
      ].join('\n');

      fs.writeFileSync(testFile, code, 'utf-8');

      // Create test command that fails if bug is present
      const testCmd = `node -e "const add = require('${testFile.replace(/\\/g, '\\\\')}'); if (add(2, 2) !== 4) process.exit(1);"`;

      const debugger = new BinarySearchDebugger({
        mode: 'code',
        file: testFile,
        test: testCmd
      });

      const result = await debugger.searchCode();

      // Should identify line 5 (zero-indexed = 4)
      assert(result.line >= 4 && result.line <= 6, `Expected line ~5, got ${result.line}`);

      // Restore original file
      fs.writeFileSync(testFile, code, 'utf-8');
    });

    it('should handle file not found', async function() {
      const debugger = new BinarySearchDebugger({
        mode: 'code',
        file: path.join(testDir, 'nonexistent.js'),
        test: 'echo test'
      });

      try {
        await debugger.searchCode();
        assert.fail('Should have thrown error');
      } catch (error) {
        assert(error.message.includes('File not found'));
      }
    });
  });

  describe('generateReport', function() {
    it('should generate valid JSON report', function() {
      const debugger = new BinarySearchDebugger({ test: 'npm test' });

      debugger.results.push({ commit: 'abc123', passed: true, error: null });
      debugger.results.push({ commit: 'def456', passed: false, error: 'Test failed' });

      debugger.generateReport();

      const reportPath = path.join(process.cwd(), 'binary-search-debug-report.json');

      assert(fs.existsSync(reportPath), 'Report file should exist');

      const report = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));

      assert.strictEqual(report.mode, 'commits');
      assert.strictEqual(report.results.length, 2);
      assert.strictEqual(report.summary.totalTests, 2);
      assert.strictEqual(report.summary.passed, 1);
      assert.strictEqual(report.summary.failed, 1);

      // Cleanup
      fs.unlinkSync(reportPath);
    });

    it('should include timestamp in report', function() {
      const debugger = new BinarySearchDebugger({ test: 'npm test' });

      debugger.generateReport();

      const reportPath = path.join(process.cwd(), 'binary-search-debug-report.json');
      const report = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));

      assert(report.timestamp);
      assert(!isNaN(Date.parse(report.timestamp)));

      // Cleanup
      fs.unlinkSync(reportPath);
    });
  });

  describe('Integration Test', function() {
    it('should successfully bisect through code', async function() {
      const testFile = path.join(testDir, 'integration-test.js');

      // Create file with multiple bugs
      const code = [
        'function calculate(x) {',
        '  let result = x;',
        '  result = result * 2;',
        '  result = result + 10;',
        '  result = result - 5; // BUG: Should be result + 5',
        '  result = result / 2;',
        '  return result;',
        '}',
        'module.exports = calculate;'
      ].join('\n');

      fs.writeFileSync(testFile, code, 'utf-8');

      const testCmd = `node -e "const calc = require('${testFile.replace(/\\/g, '\\\\')}'); if (calc(10) !== 20) process.exit(1);"`;

      const debugger = new BinarySearchDebugger({
        mode: 'code',
        file: testFile,
        test: testCmd
      });

      const result = await debugger.searchCode();

      // Should find line 5 (the buggy line)
      assert(result.line >= 4 && result.line <= 6);

      // Verify report was generated
      debugger.generateReport();
      const reportPath = path.join(process.cwd(), 'binary-search-debug-report.json');
      assert(fs.existsSync(reportPath));

      // Cleanup
      fs.unlinkSync(reportPath);
      fs.writeFileSync(testFile, code, 'utf-8');
    });
  });

  describe('Error Handling', function() {
    it('should handle invalid test command gracefully', function() {
      const debugger = new BinarySearchDebugger({
        test: 'this-command-does-not-exist'
      });

      const result = debugger.runTest();

      assert.strictEqual(result.passed, false);
      assert(result.error !== null);
    });

    it('should handle empty results array', function() {
      const debugger = new BinarySearchDebugger({ test: 'npm test' });

      assert.doesNotThrow(() => {
        debugger.generateReport();
      });

      const reportPath = path.join(process.cwd(), 'binary-search-debug-report.json');
      const report = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));

      assert.strictEqual(report.summary.totalTests, 0);

      // Cleanup
      fs.unlinkSync(reportPath);
    });
  });
});

// Run tests if executed directly
if (require.main === module) {
  console.log('Running BinarySearchDebugger tests...\n');

  const { spawn } = require('child_process');
  const mocha = spawn('npx', ['mocha', __filename, '--reporter', 'spec'], {
    stdio: 'inherit',
    shell: true
  });

  mocha.on('exit', (code) => {
    process.exit(code);
  });
}
