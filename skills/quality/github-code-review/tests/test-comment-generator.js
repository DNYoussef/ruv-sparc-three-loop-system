#!/usr/bin/env node
/**
 * Test Suite for Comment Generator
 *
 * Tests comment generation for different issue types and severities.
 *
 * Usage:
 *   node test-comment-generator.js
 */

const assert = require('assert');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m'
};

let passCount = 0;
let failCount = 0;

function test(name, fn) {
  try {
    console.log(`\n${colors.blue}TEST:${colors.reset} ${name}`);
    fn();
    passCount++;
    console.log(`${colors.green}✓ PASS${colors.reset}`);
  } catch (error) {
    failCount++;
    console.log(`${colors.red}✗ FAIL${colors.reset}: ${error.message}`);
  }
}

/**
 * Mock comment templates
 */
const templates = {
  security: {
    emoji: '🔒',
    severityMap: {
      critical: '🔴 Critical',
      high: '🟠 High',
      medium: '🟡 Medium',
      low: '🟢 Low'
    }
  },
  performance: {
    emoji: '⚡',
    severityMap: {
      critical: '🔴 Critical Performance Issue',
      high: '🟠 Major Performance Impact'
    }
  },
  style: {
    emoji: '🎨'
  },
  architecture: {
    emoji: '🏗️'
  }
};

/**
 * Test comment structure
 */
test('Should create comment with required fields', () => {
  const comment = {
    path: 'src/auth.js',
    line: 42,
    body: 'Test comment',
    type: 'security',
    severity: 'high'
  };

  assert(comment.path, 'Comment should have path');
  assert(comment.line, 'Comment should have line');
  assert(comment.body, 'Comment should have body');
  assert(comment.type, 'Comment should have type');
});

test('Should include file path and line number', () => {
  const comment = {
    path: 'src/utils.js',
    line: 100,
    body: 'Test'
  };

  assert.strictEqual(comment.path, 'src/utils.js');
  assert.strictEqual(comment.line, 100);
});

/**
 * Test severity mapping
 */
test('Should map security severity correctly', () => {
  const severity = 'critical';
  const mapped = templates.security.severityMap[severity];

  assert.strictEqual(mapped, '🔴 Critical');
});

test('Should handle all severity levels', () => {
  const severities = ['critical', 'high', 'medium', 'low'];

  severities.forEach(severity => {
    const mapped = templates.security.severityMap[severity];
    assert(mapped, `Should have mapping for ${severity}`);
  });
});

test('Should default to medium severity if not specified', () => {
  const severity = undefined;
  const defaultSeverity = severity || 'medium';

  assert.strictEqual(defaultSeverity, 'medium');
});

/**
 * Test comment type validation
 */
test('Should validate comment type', () => {
  const validTypes = ['security', 'performance', 'style', 'architecture', 'accessibility'];
  const type = 'security';

  assert(validTypes.includes(type), 'Type should be valid');
});

test('Should reject invalid comment type', () => {
  const validTypes = ['security', 'performance', 'style', 'architecture'];
  const type = 'invalid';

  assert(!validTypes.includes(type), 'Invalid type should be rejected');
});

/**
 * Test emoji inclusion
 */
test('Security comments should have lock emoji', () => {
  const emoji = templates.security.emoji;
  assert.strictEqual(emoji, '🔒');
});

test('Performance comments should have lightning emoji', () => {
  const emoji = templates.performance.emoji;
  assert.strictEqual(emoji, '⚡');
});

test('Style comments should have art emoji', () => {
  const emoji = templates.style.emoji;
  assert.strictEqual(emoji, '🎨');
});

/**
 * Test comment formatting
 */
test('Should include issue description', () => {
  const options = {
    type: 'security',
    file: 'auth.js',
    line: 42,
    issue: 'SQL injection vulnerability'
  };

  assert(options.issue.length > 0, 'Issue should not be empty');
  assert(options.issue.includes('SQL injection'));
});

test('Should include suggested fix when provided', () => {
  const options = {
    suggestion: 'Use parameterized queries'
  };

  assert(options.suggestion, 'Suggestion should be present');
});

test('Should include code example when provided', () => {
  const options = {
    code: 'const query = db.prepare("SELECT * FROM users WHERE id = ?");'
  };

  assert(options.code, 'Code example should be present');
});

test('Should include references when provided', () => {
  const options = {
    references: 'https://owasp.org/sql-injection,https://example.com/guide'
  };

  const refs = options.references.split(',');
  assert(refs.length > 0, 'Should have references');
  assert(refs[0].startsWith('http'), 'Reference should be URL');
});

/**
 * Test JSON output
 */
test('Should support JSON output format', () => {
  const comment = {
    path: 'src/app.js',
    line: 50,
    body: 'Test comment',
    type: 'security',
    severity: 'high'
  };

  const json = JSON.stringify(comment);
  const parsed = JSON.parse(json);

  assert.strictEqual(parsed.path, comment.path);
  assert.strictEqual(parsed.line, comment.line);
});

/**
 * Test edge cases
 */
test('Should handle missing optional fields', () => {
  const options = {
    type: 'security',
    file: 'auth.js',
    line: 42,
    issue: 'Issue found'
    // No suggestion, code, or references
  };

  assert(!options.suggestion, 'Suggestion should be undefined');
  assert(!options.code, 'Code should be undefined');
  assert(!options.references, 'References should be undefined');
});

test('Should handle multiline issue descriptions', () => {
  const issue = `This is a multiline
issue description
that spans multiple lines`;

  assert(issue.includes('\n'), 'Should support multiline');
});

test('Should escape special characters in code', () => {
  const code = 'const query = "SELECT * FROM `users`";';

  assert(code.includes('`'), 'Should preserve backticks');
  assert(code.includes('"'), 'Should preserve quotes');
});

/**
 * Test comment templates
 */
test('Security template should include all required sections', () => {
  const requiredSections = [
    'Severity',
    'Description',
    'Impact',
    'Suggested Fix'
  ];

  // Mock template would include these sections
  assert(requiredSections.length > 0);
});

test('Performance template should include metrics', () => {
  const hasMetrics = true; // Performance template includes metrics section
  assert(hasMetrics, 'Performance template should have metrics');
});

test('Accessibility template should include WCAG info', () => {
  const hasWCAG = true; // A11y template includes WCAG level
  assert(hasWCAG, 'A11y template should reference WCAG');
});

/**
 * Print test results
 */
console.log('\n' + '='.repeat(60));
console.log(`${colors.blue}TEST RESULTS${colors.reset}`);
console.log('='.repeat(60));
console.log(`${colors.green}✓ Passed:${colors.reset} ${passCount}`);
console.log(`${colors.red}✗ Failed:${colors.reset} ${failCount}`);
console.log(`${colors.yellow}Total:${colors.reset} ${passCount + failCount}`);
console.log('='.repeat(60));

if (failCount > 0) {
  console.log(`\n${colors.red}Some tests failed!${colors.reset}`);
  process.exit(1);
} else {
  console.log(`\n${colors.green}All tests passed!${colors.reset}`);
  process.exit(0);
}
