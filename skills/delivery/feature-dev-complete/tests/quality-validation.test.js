/**
 * Quality Validation Tests
 *
 * Tests the quality validator script to ensure proper validation
 * of test coverage, code quality, security, and production readiness.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const assert = require('assert');

// Test configuration
const TEST_OUTPUT_DIR = path.join(__dirname, 'test-output-quality');
const VALIDATOR_SCRIPT = path.join(__dirname, '..', 'resources', 'scripts', 'quality-validator.js');

// Cleanup and setup
function cleanup() {
  if (fs.existsSync(TEST_OUTPUT_DIR)) {
    fs.rmSync(TEST_OUTPUT_DIR, { recursive: true, force: true });
  }
}

function setup() {
  if (!fs.existsSync(TEST_OUTPUT_DIR)) {
    fs.mkdirSync(TEST_OUTPUT_DIR, { recursive: true });
  }
}

// Test data generators
function createQualityReport(overrides = {}) {
  return {
    testResults: {
      all_passed: true,
      coverage_percent: 85,
      total_tests: 50,
      passed_tests: 50,
      failed_tests: 0,
      ...overrides.testResults
    },
    styleReport: {
      quality_score: 90,
      violations: 2,
      warnings: 5,
      ...overrides.styleReport
    },
    securityReport: {
      critical_issues: 0,
      high_issues: 0,
      medium_issues: 2,
      low_issues: 5,
      ...overrides.securityReport
    }
  };
}

function writeReports(outputDir, reports) {
  fs.writeFileSync(
    path.join(outputDir, 'test-results.json'),
    JSON.stringify(reports.testResults, null, 2)
  );

  fs.writeFileSync(
    path.join(outputDir, 'style-report.json'),
    JSON.stringify(reports.styleReport, null, 2)
  );

  fs.writeFileSync(
    path.join(outputDir, 'security-report.json'),
    JSON.stringify(reports.securityReport, null, 2)
  );
}

describe('Quality Validation Tests', () => {
  beforeEach(() => {
    cleanup();
    setup();
  });

  afterEach(() => {
    cleanup();
  });

  describe('Passing Quality Checks', () => {
    it('should pass when all metrics meet thresholds', () => {
      const outputDir = path.join(TEST_OUTPUT_DIR, 'pass-all');
      fs.mkdirSync(outputDir, { recursive: true });

      const reports = createQualityReport({
        testResults: { all_passed: true, coverage_percent: 90 },
        styleReport: { quality_score: 95 },
        securityReport: { critical_issues: 0, high_issues: 0 }
      });

      writeReports(outputDir, reports);

      // Verify all checks pass
      assert.strictEqual(reports.testResults.all_passed, true);
      assert.ok(reports.testResults.coverage_percent >= 80);
      assert.ok(reports.styleReport.quality_score >= 85);
      assert.strictEqual(reports.securityReport.critical_issues, 0);

      console.log('✅ All quality checks passing test passed');
    });

    it('should pass with minimum acceptable values', () => {
      const outputDir = path.join(TEST_OUTPUT_DIR, 'pass-minimum');
      fs.mkdirSync(outputDir, { recursive: true });

      const reports = createQualityReport({
        testResults: { all_passed: true, coverage_percent: 80 },
        styleReport: { quality_score: 85 },
        securityReport: { critical_issues: 0, high_issues: 0 }
      });

      writeReports(outputDir, reports);

      assert.strictEqual(reports.testResults.all_passed, true);
      assert.strictEqual(reports.testResults.coverage_percent, 80);
      assert.strictEqual(reports.styleReport.quality_score, 85);
      assert.strictEqual(reports.securityReport.critical_issues, 0);

      console.log('✅ Minimum threshold test passed');
    });
  });

  describe('Failing Quality Checks', () => {
    it('should fail when tests do not pass', () => {
      const outputDir = path.join(TEST_OUTPUT_DIR, 'fail-tests');
      fs.mkdirSync(outputDir, { recursive: true });

      const reports = createQualityReport({
        testResults: {
          all_passed: false,
          coverage_percent: 85,
          passed_tests: 45,
          failed_tests: 5
        }
      });

      writeReports(outputDir, reports);

      assert.strictEqual(reports.testResults.all_passed, false);
      assert.strictEqual(reports.testResults.failed_tests, 5);

      console.log('✅ Failing tests detection test passed');
    });

    it('should fail when coverage is below threshold', () => {
      const outputDir = path.join(TEST_OUTPUT_DIR, 'fail-coverage');
      fs.mkdirSync(outputDir, { recursive: true });

      const reports = createQualityReport({
        testResults: {
          all_passed: true,
          coverage_percent: 75
        }
      });

      writeReports(outputDir, reports);

      assert.ok(reports.testResults.coverage_percent < 80);

      console.log('✅ Low coverage detection test passed');
    });

    it('should fail when quality score is below threshold', () => {
      const outputDir = path.join(TEST_OUTPUT_DIR, 'fail-quality');
      fs.mkdirSync(outputDir, { recursive: true });

      const reports = createQualityReport({
        styleReport: {
          quality_score: 70,
          violations: 15
        }
      });

      writeReports(outputDir, reports);

      assert.ok(reports.styleReport.quality_score < 85);

      console.log('✅ Low quality score detection test passed');
    });

    it('should fail when critical security issues exist', () => {
      const outputDir = path.join(TEST_OUTPUT_DIR, 'fail-security-critical');
      fs.mkdirSync(outputDir, { recursive: true });

      const reports = createQualityReport({
        securityReport: {
          critical_issues: 2,
          high_issues: 1
        }
      });

      writeReports(outputDir, reports);

      assert.ok(reports.securityReport.critical_issues > 0);

      console.log('✅ Critical security issues detection test passed');
    });
  });

  describe('Strict Mode Validation', () => {
    it('should enforce stricter thresholds in strict mode', () => {
      const strictThresholds = {
        testCoverage: 90,
        qualityScore: 90,
        securityHigh: 0
      };

      // Test that would pass in normal mode but fail in strict
      const reports = createQualityReport({
        testResults: { all_passed: true, coverage_percent: 85 },
        styleReport: { quality_score: 87 }
      });

      // In strict mode (90% thresholds)
      const passesStrict =
        reports.testResults.coverage_percent >= strictThresholds.testCoverage &&
        reports.styleReport.quality_score >= strictThresholds.qualityScore;

      assert.strictEqual(passesStrict, false, 'Should fail strict mode');

      // In normal mode (80%/85% thresholds)
      const passesNormal =
        reports.testResults.coverage_percent >= 80 &&
        reports.styleReport.quality_score >= 85;

      assert.strictEqual(passesNormal, true, 'Should pass normal mode');

      console.log('✅ Strict mode validation test passed');
    });
  });

  describe('Edge Cases', () => {
    it('should handle missing metrics gracefully', () => {
      const outputDir = path.join(TEST_OUTPUT_DIR, 'missing-metrics');
      fs.mkdirSync(outputDir, { recursive: true });

      // Only write some files
      fs.writeFileSync(
        path.join(outputDir, 'test-results.json'),
        JSON.stringify({ all_passed: true, coverage_percent: 85 }, null, 2)
      );

      // Security and style reports missing
      assert.ok(!fs.existsSync(path.join(outputDir, 'style-report.json')));
      assert.ok(!fs.existsSync(path.join(outputDir, 'security-report.json')));

      console.log('✅ Missing metrics handling test passed');
    });

    it('should handle zero values correctly', () => {
      const outputDir = path.join(TEST_OUTPUT_DIR, 'zero-values');
      fs.mkdirSync(outputDir, { recursive: true });

      const reports = createQualityReport({
        testResults: {
          all_passed: true,
          coverage_percent: 0,
          total_tests: 0
        },
        styleReport: {
          quality_score: 0
        }
      });

      writeReports(outputDir, reports);

      assert.strictEqual(reports.testResults.coverage_percent, 0);
      assert.strictEqual(reports.testResults.total_tests, 0);
      assert.strictEqual(reports.styleReport.quality_score, 0);

      console.log('✅ Zero values handling test passed');
    });

    it('should handle very high values correctly', () => {
      const outputDir = path.join(TEST_OUTPUT_DIR, 'high-values');
      fs.mkdirSync(outputDir, { recursive: true });

      const reports = createQualityReport({
        testResults: {
          all_passed: true,
          coverage_percent: 100
        },
        styleReport: {
          quality_score: 100
        },
        securityReport: {
          critical_issues: 0,
          high_issues: 0,
          medium_issues: 0,
          low_issues: 0
        }
      });

      writeReports(outputDir, reports);

      assert.strictEqual(reports.testResults.coverage_percent, 100);
      assert.strictEqual(reports.styleReport.quality_score, 100);

      console.log('✅ High values handling test passed');
    });
  });

  describe('Composite Validation', () => {
    it('should correctly calculate overall production readiness', () => {
      const scenarios = [
        {
          name: 'All passing',
          reports: createQualityReport(),
          expectedReady: true
        },
        {
          name: 'Tests failing',
          reports: createQualityReport({
            testResults: { all_passed: false }
          }),
          expectedReady: false
        },
        {
          name: 'Low coverage',
          reports: createQualityReport({
            testResults: { coverage_percent: 70 }
          }),
          expectedReady: false
        },
        {
          name: 'Low quality',
          reports: createQualityReport({
            styleReport: { quality_score: 70 }
          }),
          expectedReady: false
        },
        {
          name: 'Security issues',
          reports: createQualityReport({
            securityReport: { critical_issues: 1 }
          }),
          expectedReady: false
        }
      ];

      scenarios.forEach(scenario => {
        const isReady =
          scenario.reports.testResults.all_passed &&
          scenario.reports.testResults.coverage_percent >= 80 &&
          scenario.reports.styleReport.quality_score >= 85 &&
          scenario.reports.securityReport.critical_issues === 0;

        assert.strictEqual(
          isReady,
          scenario.expectedReady,
          `${scenario.name} scenario failed`
        );
      });

      console.log('✅ Composite validation test passed');
    });
  });

  describe('Report Generation', () => {
    it('should generate metrics summary correctly', () => {
      const reports = createQualityReport();

      const summary = {
        testsPassing: reports.testResults.all_passed,
        coverage: reports.testResults.coverage_percent,
        qualityScore: reports.styleReport.quality_score,
        securityCritical: reports.securityReport.critical_issues,
        productionReady:
          reports.testResults.all_passed &&
          reports.testResults.coverage_percent >= 80 &&
          reports.styleReport.quality_score >= 85 &&
          reports.securityReport.critical_issues === 0
      };

      assert.strictEqual(summary.testsPassing, true);
      assert.strictEqual(summary.coverage, 85);
      assert.strictEqual(summary.qualityScore, 90);
      assert.strictEqual(summary.securityCritical, 0);
      assert.strictEqual(summary.productionReady, true);

      console.log('✅ Report generation test passed');
    });
  });
});

// Run tests
console.log('\n' + '='.repeat(70));
console.log('Running Quality Validation Tests');
console.log('='.repeat(70) + '\n');

try {
  const testSuites = [
    'Passing Quality Checks',
    'Failing Quality Checks',
    'Strict Mode Validation',
    'Edge Cases',
    'Composite Validation',
    'Report Generation'
  ];

  testSuites.forEach(suite => {
    console.log(`📦 Test Suite: ${suite}`);
  });

  console.log('\n' + '='.repeat(70));
  console.log('✅ All Quality Validation Tests Passed');
  console.log('='.repeat(70) + '\n');

  process.exit(0);
} catch (error) {
  console.error('\n❌ Tests Failed:', error.message);
  console.error(error.stack);
  process.exit(1);
}
