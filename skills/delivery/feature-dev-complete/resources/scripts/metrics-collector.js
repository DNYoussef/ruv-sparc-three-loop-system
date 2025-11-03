#!/usr/bin/env node
/**
 * Metrics Collection and Aggregation Script
 *
 * Aggregates all quality metrics from feature development workflow:
 * - Test results (coverage, pass/fail)
 * - Code quality scores
 * - Security scan results
 * - Performance metrics
 * - Build statistics
 *
 * Outputs comprehensive metrics report in JSON and Markdown formats.
 *
 * @usage node metrics-collector.js <output-dir> [--format json|markdown|both]
 */

const fs = require('fs');
const path = require('path');

const outputDir = process.argv[2];
const format = (process.argv[3] || '--format').replace('--format=', '') || 'both';

if (!outputDir) {
  console.error('❌ Error: Output directory required');
  console.error('Usage: node metrics-collector.js <output-dir> [--format json|markdown|both]');
  process.exit(1);
}

// Read JSON files
function readJSON(filename) {
  const filepath = path.join(outputDir, filename);
  try {
    return JSON.parse(fs.readFileSync(filepath, 'utf-8'));
  } catch (error) {
    return null;
  }
}

// Collect metrics from all sources
const testResults = readJSON('test-results.json');
const styleReport = readJSON('style-report.json');
const securityReport = readJSON('security-report.json');
const theaterReport = readJSON('theater-report.json');

// Aggregate metrics
const metrics = {
  timestamp: new Date().toISOString(),
  outputDirectory: outputDir,

  testing: {
    allTestsPassing: testResults?.all_passed || false,
    totalTests: testResults?.total_tests || 0,
    passedTests: testResults?.passed_tests || 0,
    failedTests: testResults?.failed_tests || 0,
    skippedTests: testResults?.skipped_tests || 0,
    coveragePercent: testResults?.coverage_percent || 0,
    executionTimeMs: testResults?.execution_time_ms || 0,
    testFramework: testResults?.framework || 'unknown'
  },

  codeQuality: {
    qualityScore: styleReport?.quality_score || 0,
    violations: styleReport?.violations || 0,
    warnings: styleReport?.warnings || 0,
    linesOfCode: styleReport?.lines_of_code || 0,
    filesAnalyzed: styleReport?.files_analyzed || 0,
    complexityScore: styleReport?.complexity_score || 0,
    maintainabilityIndex: styleReport?.maintainability_index || 0,
    technicalDebt: styleReport?.technical_debt_minutes || 0
  },

  security: {
    criticalIssues: securityReport?.critical_issues || 0,
    highIssues: securityReport?.high_issues || 0,
    mediumIssues: securityReport?.medium_issues || 0,
    lowIssues: securityReport?.low_issues || 0,
    infoIssues: securityReport?.info_issues || 0,
    totalIssues: (securityReport?.critical_issues || 0) +
                 (securityReport?.high_issues || 0) +
                 (securityReport?.medium_issues || 0) +
                 (securityReport?.low_issues || 0) +
                 (securityReport?.info_issues || 0),
    vulnerabilities: securityReport?.vulnerabilities || []
  },

  implementation: {
    theaterIssues: theaterReport?.issues?.length || 0,
    placeholderCount: theaterReport?.placeholder_count || 0,
    todoCount: theaterReport?.todo_count || 0,
    implementationComplete: (theaterReport?.issues?.length || 0) === 0
  },

  overall: {
    productionReady: false,
    qualityGrade: 'F',
    score: 0
  }
};

// Calculate overall metrics
const weights = {
  testing: 0.35,
  codeQuality: 0.30,
  security: 0.25,
  implementation: 0.10
};

// Calculate weighted score
let overallScore = 0;

// Testing score (0-100)
const testingScore = metrics.testing.allTestsPassing
  ? Math.min(100, metrics.testing.coveragePercent + 20)
  : metrics.testing.coveragePercent * 0.5;
overallScore += testingScore * weights.testing;

// Code quality score (0-100)
const qualityScore = metrics.codeQuality.qualityScore;
overallScore += qualityScore * weights.codeQuality;

// Security score (0-100)
const securityPenalty = (
  metrics.security.criticalIssues * 50 +
  metrics.security.highIssues * 20 +
  metrics.security.mediumIssues * 5
);
const securityScore = Math.max(0, 100 - securityPenalty);
overallScore += securityScore * weights.security;

// Implementation score (0-100)
const implementationScore = metrics.implementation.implementationComplete
  ? 100
  : Math.max(0, 100 - (metrics.implementation.theaterIssues * 10));
overallScore += implementationScore * weights.implementation;

// Update overall metrics
metrics.overall.score = Math.round(overallScore);
metrics.overall.productionReady =
  metrics.testing.allTestsPassing &&
  metrics.codeQuality.qualityScore >= 85 &&
  metrics.security.criticalIssues === 0 &&
  metrics.implementation.implementationComplete;

// Assign grade
if (overallScore >= 90) metrics.overall.qualityGrade = 'A';
else if (overallScore >= 80) metrics.overall.qualityGrade = 'B';
else if (overallScore >= 70) metrics.overall.qualityGrade = 'C';
else if (overallScore >= 60) metrics.overall.qualityGrade = 'D';
else metrics.overall.qualityGrade = 'F';

// Output in requested format(s)
if (format === 'json' || format === 'both') {
  const jsonFile = path.join(outputDir, 'metrics-summary.json');
  fs.writeFileSync(jsonFile, JSON.stringify(metrics, null, 2));
  console.log(`✅ JSON metrics saved to: ${jsonFile}`);
}

if (format === 'markdown' || format === 'both') {
  const markdown = `# Feature Development Metrics

**Generated**: ${metrics.timestamp}
**Directory**: ${metrics.outputDirectory}

---

## Overall Assessment

| Metric | Value |
|--------|-------|
| **Overall Score** | ${metrics.overall.score}/100 |
| **Quality Grade** | ${metrics.overall.qualityGrade} |
| **Production Ready** | ${metrics.overall.productionReady ? '✅ Yes' : '❌ No'} |

---

## Testing Metrics

| Metric | Value |
|--------|-------|
| All Tests Passing | ${metrics.testing.allTestsPassing ? '✅ Yes' : '❌ No'} |
| Total Tests | ${metrics.testing.totalTests} |
| Passed | ${metrics.testing.passedTests} |
| Failed | ${metrics.testing.failedTests} |
| Skipped | ${metrics.testing.skippedTests} |
| **Coverage** | **${metrics.testing.coveragePercent}%** |
| Execution Time | ${metrics.testing.executionTimeMs}ms |
| Test Framework | ${metrics.testing.testFramework} |

---

## Code Quality Metrics

| Metric | Value |
|--------|-------|
| **Quality Score** | **${metrics.codeQuality.qualityScore}/100** |
| Violations | ${metrics.codeQuality.violations} |
| Warnings | ${metrics.codeQuality.warnings} |
| Lines of Code | ${metrics.codeQuality.linesOfCode.toLocaleString()} |
| Files Analyzed | ${metrics.codeQuality.filesAnalyzed} |
| Complexity Score | ${metrics.codeQuality.complexityScore} |
| Maintainability Index | ${metrics.codeQuality.maintainabilityIndex} |
| Technical Debt | ${metrics.codeQuality.technicalDebt} minutes |

---

## Security Metrics

| Severity | Count |
|----------|-------|
| 🚨 Critical | ${metrics.security.criticalIssues} |
| ⚠️  High | ${metrics.security.highIssues} |
| 🟡 Medium | ${metrics.security.mediumIssues} |
| 🟢 Low | ${metrics.security.lowIssues} |
| ℹ️  Info | ${metrics.security.infoIssues} |
| **Total** | **${metrics.security.totalIssues}** |

${metrics.security.vulnerabilities.length > 0 ? `
### Vulnerabilities Detected

${metrics.security.vulnerabilities.map(v => `- **${v.severity}**: ${v.title} (${v.location})`).join('\n')}
` : ''}

---

## Implementation Completeness

| Metric | Value |
|--------|-------|
| Theater Issues | ${metrics.implementation.theaterIssues} |
| Placeholders | ${metrics.implementation.placeholderCount} |
| TODOs | ${metrics.implementation.todoCount} |
| **Complete** | **${metrics.implementation.implementationComplete ? '✅ Yes' : '❌ No'}** |

---

## Scoring Breakdown

| Category | Weight | Score | Weighted |
|----------|--------|-------|----------|
| Testing | ${(weights.testing * 100).toFixed(0)}% | ${testingScore.toFixed(1)}/100 | ${(testingScore * weights.testing).toFixed(1)} |
| Code Quality | ${(weights.codeQuality * 100).toFixed(0)}% | ${qualityScore.toFixed(1)}/100 | ${(qualityScore * weights.codeQuality).toFixed(1)} |
| Security | ${(weights.security * 100).toFixed(0)}% | ${securityScore.toFixed(1)}/100 | ${(securityScore * weights.security).toFixed(1)} |
| Implementation | ${(weights.implementation * 100).toFixed(0)}% | ${implementationScore.toFixed(1)}/100 | ${(implementationScore * weights.implementation).toFixed(1)} |
| **Total** | **100%** | - | **${metrics.overall.score}/100** |

---

## Production Readiness Checklist

- [${metrics.testing.allTestsPassing ? 'x' : ' '}] All tests passing
- [${metrics.testing.coveragePercent >= 80 ? 'x' : ' '}] Test coverage ≥80%
- [${metrics.codeQuality.qualityScore >= 85 ? 'x' : ' '}] Code quality ≥85/100
- [${metrics.security.criticalIssues === 0 ? 'x' : ' '}] No critical security issues
- [${metrics.security.highIssues === 0 ? 'x' : ' '}] No high security issues
- [${metrics.implementation.implementationComplete ? 'x' : ' '}] Implementation complete (no placeholders)

${metrics.overall.productionReady
  ? '✅ **All checks passed - READY FOR PRODUCTION**'
  : '❌ **Not ready for production - address issues above**'}

---

*Generated by Claude Code Feature Development Metrics Collector*
`;

  const mdFile = path.join(outputDir, 'METRICS-REPORT.md');
  fs.writeFileSync(mdFile, markdown);
  console.log(`✅ Markdown report saved to: ${mdFile}`);
}

// Print summary to console
console.log('\n' + '='.repeat(70));
console.log('METRICS SUMMARY');
console.log('='.repeat(70));
console.log(`Overall Score: ${metrics.overall.score}/100 (Grade: ${metrics.overall.qualityGrade})`);
console.log(`Production Ready: ${metrics.overall.productionReady ? 'YES ✅' : 'NO ❌'}`);
console.log('');
console.log(`Testing: ${testingScore.toFixed(1)}/100 (${metrics.testing.coveragePercent}% coverage)`);
console.log(`Code Quality: ${qualityScore.toFixed(1)}/100`);
console.log(`Security: ${securityScore.toFixed(1)}/100 (${metrics.security.criticalIssues} critical)`);
console.log(`Implementation: ${implementationScore.toFixed(1)}/100`);
console.log('='.repeat(70));
console.log('');

process.exit(metrics.overall.productionReady ? 0 : 1);
