#!/usr/bin/env node
/**
 * Quality Validation Script for Feature Development
 *
 * Validates feature implementation against quality thresholds:
 * - Test coverage ≥80%
 * - Code quality score ≥85/100
 * - Zero critical security issues
 * - All tests passing
 *
 * @usage node quality-validator.js <output-dir> [--strict]
 */

const fs = require('fs');
const path = require('path');

const outputDir = process.argv[2];
const strictMode = process.argv.includes('--strict');

if (!outputDir) {
  console.error('❌ Error: Output directory required');
  console.error('Usage: node quality-validator.js <output-dir> [--strict]');
  process.exit(1);
}

// Default thresholds
const thresholds = {
  testCoverage: strictMode ? 90 : 80,
  qualityScore: strictMode ? 90 : 85,
  securityCritical: 0,
  securityHigh: strictMode ? 0 : 3,
  testsPassing: true
};

// Read result files
function readJSON(filename) {
  const filepath = path.join(outputDir, filename);
  try {
    return JSON.parse(fs.readFileSync(filepath, 'utf-8'));
  } catch (error) {
    console.warn(`⚠️  Could not read ${filename}: ${error.message}`);
    return null;
  }
}

const testResults = readJSON('test-results.json');
const styleReport = readJSON('style-report.json');
const securityReport = readJSON('security-report.json');

// Validation checks
const checks = {
  testsPassing: {
    name: 'All Tests Passing',
    actual: testResults?.all_passed || false,
    expected: thresholds.testsPassing,
    pass: (testResults?.all_passed || false) === thresholds.testsPassing,
    severity: 'critical'
  },
  testCoverage: {
    name: 'Test Coverage',
    actual: testResults?.coverage_percent || 0,
    expected: `≥${thresholds.testCoverage}%`,
    pass: (testResults?.coverage_percent || 0) >= thresholds.testCoverage,
    severity: 'high'
  },
  qualityScore: {
    name: 'Code Quality Score',
    actual: styleReport?.quality_score || 0,
    expected: `≥${thresholds.qualityScore}/100`,
    pass: (styleReport?.quality_score || 0) >= thresholds.qualityScore,
    severity: 'high'
  },
  securityCritical: {
    name: 'Critical Security Issues',
    actual: securityReport?.critical_issues || 0,
    expected: thresholds.securityCritical,
    pass: (securityReport?.critical_issues || 0) === thresholds.securityCritical,
    severity: 'critical'
  },
  securityHigh: {
    name: 'High Security Issues',
    actual: securityReport?.high_issues || 0,
    expected: `≤${thresholds.securityHigh}`,
    pass: (securityReport?.high_issues || 0) <= thresholds.securityHigh,
    severity: 'medium'
  }
};

// Print validation report
console.log('\n' + '='.repeat(70));
console.log('QUALITY VALIDATION REPORT');
console.log('='.repeat(70));
console.log(`Mode: ${strictMode ? 'STRICT' : 'STANDARD'}`);
console.log(`Directory: ${outputDir}`);
console.log('='.repeat(70));

let hasFailures = false;
let hasCriticalFailures = false;

Object.entries(checks).forEach(([key, check]) => {
  const icon = check.pass ? '✅' : '❌';
  const status = check.pass ? 'PASS' : 'FAIL';

  console.log(`\n${icon} ${check.name}: ${status}`);
  console.log(`   Expected: ${check.expected}`);
  console.log(`   Actual: ${check.actual}`);
  console.log(`   Severity: ${check.severity.toUpperCase()}`);

  if (!check.pass) {
    hasFailures = true;
    if (check.severity === 'critical') {
      hasCriticalFailures = true;
    }
  }
});

// Additional metrics
console.log('\n' + '='.repeat(70));
console.log('ADDITIONAL METRICS');
console.log('='.repeat(70));

if (testResults) {
  console.log(`\n📊 Test Statistics:`);
  console.log(`   Total Tests: ${testResults.total_tests || 'N/A'}`);
  console.log(`   Passed: ${testResults.passed_tests || 'N/A'}`);
  console.log(`   Failed: ${testResults.failed_tests || 'N/A'}`);
  console.log(`   Skipped: ${testResults.skipped_tests || 'N/A'}`);
}

if (styleReport) {
  console.log(`\n🎨 Style Metrics:`);
  console.log(`   Style Violations: ${styleReport.violations || 'N/A'}`);
  console.log(`   Warnings: ${styleReport.warnings || 'N/A'}`);
  console.log(`   Lines of Code: ${styleReport.lines_of_code || 'N/A'}`);
  console.log(`   Complexity Score: ${styleReport.complexity_score || 'N/A'}`);
}

if (securityReport) {
  console.log(`\n🔒 Security Metrics:`);
  console.log(`   Critical: ${securityReport.critical_issues || 0}`);
  console.log(`   High: ${securityReport.high_issues || 0}`);
  console.log(`   Medium: ${securityReport.medium_issues || 0}`);
  console.log(`   Low: ${securityReport.low_issues || 0}`);
  console.log(`   Info: ${securityReport.info_issues || 0}`);
}

// Final verdict
console.log('\n' + '='.repeat(70));
console.log('FINAL VERDICT');
console.log('='.repeat(70));

if (hasCriticalFailures) {
  console.log('\n🚨 FAILED - Critical issues must be resolved\n');
  process.exit(1);
} else if (hasFailures && strictMode) {
  console.log('\n⚠️  FAILED - Issues found in strict mode\n');
  process.exit(1);
} else if (hasFailures) {
  console.log('\n⚠️  PASSED WITH WARNINGS - Non-critical issues detected\n');
  process.exit(0);
} else {
  console.log('\n✅ PASSED - All quality checks successful\n');
  process.exit(0);
}
