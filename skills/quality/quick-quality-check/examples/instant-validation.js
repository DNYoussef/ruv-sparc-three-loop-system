#!/usr/bin/env node
/**
 * Instant Validation Example
 *
 * Demonstrates quick-quality-check for instant validation of code changes
 * before committing. Perfect for pre-commit hooks and rapid feedback loops.
 *
 * Part of quick-quality-check Enhanced tier examples (150-300 lines)
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Configuration
const CONFIG = {
  targetPath: process.argv[2] || '.',
  timeout: 30000, // 30 seconds
  parallel: true,
  quickMode: true,
  outputDir: '.quality-reports',
};

/**
 * Execute command and capture output
 */
function executeCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, {
      ...options,
      timeout: CONFIG.timeout,
    });

    let stdout = '';
    let stderr = '';

    proc.stdout?.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr?.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      resolve({ code, stdout, stderr });
    });

    proc.on('error', (error) => {
      reject(error);
    });
  });
}

/**
 * Run fast linter
 */
async function runLinter(targetPath) {
  console.log('[1/4] Running fast linter...');

  const linterScript = path.join(__dirname, '../resources/fast-linter.sh');
  const outputFile = path.join(CONFIG.outputDir, 'lint-results.json');

  try {
    const result = await executeCommand('bash', [linterScript, targetPath, outputFile]);

    console.log(`✓ Linting completed (${result.code === 0 ? 'PASSED' : 'FAILED'})`);

    if (fs.existsSync(outputFile)) {
      const lintData = JSON.parse(fs.readFileSync(outputFile, 'utf8'));
      const summary = lintData.summary || {};

      console.log(`  Total issues: ${summary.total_issues || 0}`);
      console.log(`  ESLint errors: ${summary.eslint_errors || 0}`);
      console.log(`  ESLint warnings: ${summary.eslint_warnings || 0}`);
    }

    return result.code === 0;
  } catch (error) {
    console.error(`✗ Linting failed: ${error.message}`);
    return false;
  }
}

/**
 * Run security scanner
 */
async function runSecurityScan(targetPath) {
  console.log('[2/4] Running security scanner...');

  const scannerScript = path.join(__dirname, '../resources/security-scanner.py');
  const outputFile = path.join(CONFIG.outputDir, 'security-results.json');

  try {
    const result = await executeCommand('python3', [scannerScript, targetPath, '-o', outputFile]);

    console.log(`✓ Security scan completed (${result.code === 0 ? 'PASSED' : 'FAILED'})`);

    if (fs.existsSync(outputFile)) {
      const securityData = JSON.parse(fs.readFileSync(outputFile, 'utf8'));
      const summary = securityData.scan_summary || {};

      console.log(`  Files scanned: ${summary.files_scanned || 0}`);
      console.log(`  Critical issues: ${summary.critical || 0}`);
      console.log(`  High issues: ${summary.high || 0}`);

      if (summary.critical > 0) {
        console.error(`  ⚠️  CRITICAL: ${summary.critical} critical security issues found!`);
      }
    }

    return result.code === 0;
  } catch (error) {
    console.error(`✗ Security scan failed: ${error.message}`);
    return false;
  }
}

/**
 * Run test suite
 */
async function runTests(targetPath) {
  console.log('[3/4] Running test suite...');

  const testRunner = path.join(__dirname, '../resources/test-runner.js');
  const outputFile = path.join(CONFIG.outputDir, 'test-results.json');

  try {
    const result = await executeCommand('node', [testRunner, targetPath, outputFile]);

    console.log(`✓ Tests completed (${result.code === 0 ? 'PASSED' : 'FAILED'})`);

    if (fs.existsSync(outputFile)) {
      const testData = JSON.parse(fs.readFileSync(outputFile, 'utf8'));

      console.log(`  Framework: ${testData.framework || 'unknown'}`);
      console.log(`  Tests run: ${testData.tests_run || 0}`);
      console.log(`  Passed: ${testData.tests_passed || 0}`);
      console.log(`  Failed: ${testData.tests_failed || 0}`);
      console.log(`  Execution time: ${testData.execution_time || 0}s`);

      if (testData.tests_failed > 0) {
        console.error(`  ⚠️  ${testData.tests_failed} tests failing!`);
      }
    }

    return result.code === 0;
  } catch (error) {
    console.error(`✗ Tests failed: ${error.message}`);
    return false;
  }
}

/**
 * Generate quality report
 */
async function generateReport() {
  console.log('[4/4] Generating quality report...');

  const reporterScript = path.join(__dirname, '../resources/quality-reporter.py');
  const lintFile = path.join(CONFIG.outputDir, 'lint-results.json');
  const securityFile = path.join(CONFIG.outputDir, 'security-results.json');
  const testFile = path.join(CONFIG.outputDir, 'test-results.json');
  const outputFile = path.join(CONFIG.outputDir, 'quality-report.json');

  try {
    const args = [
      reporterScript,
      '--format', 'console',
    ];

    if (fs.existsSync(lintFile)) args.push('--lint', lintFile);
    if (fs.existsSync(securityFile)) args.push('--security', securityFile);
    if (fs.existsSync(testFile)) args.push('--tests', testFile);

    args.push('-o', outputFile);

    const result = await executeCommand('python3', args);

    console.log('\n' + result.stdout);

    if (fs.existsSync(outputFile)) {
      const reportData = JSON.parse(fs.readFileSync(outputFile, 'utf8'));
      return reportData.passed;
    }

    return result.code === 0;
  } catch (error) {
    console.error(`✗ Report generation failed: ${error.message}`);
    return false;
  }
}

/**
 * Main execution function
 */
async function main() {
  const startTime = Date.now();

  console.log('='.repeat(80));
  console.log('INSTANT VALIDATION - Quick Quality Check');
  console.log('='.repeat(80));
  console.log(`Target: ${CONFIG.targetPath}`);
  console.log(`Mode: ${CONFIG.quickMode ? 'Quick' : 'Comprehensive'}`);
  console.log(`Timeout: ${CONFIG.timeout}ms`);
  console.log('='.repeat(80));
  console.log('');

  // Ensure output directory exists
  if (!fs.existsSync(CONFIG.outputDir)) {
    fs.mkdirSync(CONFIG.outputDir, { recursive: true });
  }

  let allPassed = true;

  try {
    // Run all checks in sequence (parallel execution would go here)
    const lintPassed = await runLinter(CONFIG.targetPath);
    const securityPassed = await runSecurityScan(CONFIG.targetPath);
    const testsPassed = await runTests(CONFIG.targetPath);

    allPassed = lintPassed && securityPassed && testsPassed;

    console.log('');

    // Generate final report
    const reportPassed = await generateReport();

    const executionTime = ((Date.now() - startTime) / 1000).toFixed(2);

    console.log('');
    console.log('='.repeat(80));
    console.log(`Execution time: ${executionTime}s`);

    if (allPassed && reportPassed) {
      console.log('✅ VALIDATION PASSED - Code is ready to commit!');
      console.log('='.repeat(80));
      process.exit(0);
    } else {
      console.log('❌ VALIDATION FAILED - Please fix issues before committing');
      console.log('='.repeat(80));
      process.exit(1);
    }
  } catch (error) {
    console.error('');
    console.error('='.repeat(80));
    console.error(`❌ VALIDATION ERROR: ${error.message}`);
    console.error('='.repeat(80));
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main().catch((error) => {
    console.error(`Fatal error: ${error.message}`);
    process.exit(1);
  });
}

module.exports = {
  runLinter,
  runSecurityScan,
  runTests,
  generateReport,
};
