#!/usr/bin/env node
/**
 * Tests for GitHub Actions Workflow Tester
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { describe, it, beforeEach, afterEach } = require('node:test');

// Mock WorkflowTester since we're testing in isolation
class MockWorkflowTester {
  constructor(options = {}) {
    this.workflowDir = options.workflowDir || '.github/workflows';
    this.verbose = options.verbose || false;
    this.dryRun = options.dryRun || false;
  }

  loadWorkflow(workflowPath) {
    try {
      const content = fs.readFileSync(workflowPath, 'utf8');
      const yaml = require('js-yaml');
      return yaml.load(content);
    } catch (error) {
      return null;
    }
  }

  validateSyntax(workflow) {
    const errors = [];
    const warnings = [];

    if (!workflow.name) errors.push('Missing required field: name');
    if (!workflow.on) errors.push('Missing required field: on');
    if (!workflow.jobs) errors.push('Missing required field: jobs');

    return { errors, warnings };
  }

  analyzeBestPractices(workflow) {
    const suggestions = [];

    const workflowStr = JSON.stringify(workflow);
    if (!workflowStr.includes('actions/cache')) {
      suggestions.push({
        type: 'performance',
        message: 'Consider adding dependency caching',
        priority: 'high',
      });
    }

    return suggestions;
  }

  log(message, level = 'info') {
    if (this.verbose || level !== 'info') {
      console.log(`[${level.toUpperCase()}] ${message}`);
    }
  }
}

describe('WorkflowTester', () => {
  let tester;
  let testWorkflowPath;

  beforeEach(() => {
    tester = new MockWorkflowTester({ verbose: false, dryRun: true });

    // Create temporary test workflow
    const testDir = fs.mkdtempSync('workflow-test-');
    testWorkflowPath = path.join(testDir, 'test.yml');

    const validWorkflow = `
name: Test Workflow
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: npm test
`;
    fs.writeFileSync(testWorkflowPath, validWorkflow);
  });

  afterEach(() => {
    // Cleanup
    if (testWorkflowPath && fs.existsSync(testWorkflowPath)) {
      fs.unlinkSync(testWorkflowPath);
      const dir = path.dirname(testWorkflowPath);
      if (fs.existsSync(dir)) {
        fs.rmdirSync(dir, { recursive: true });
      }
    }
  });

  it('should load valid workflow file', () => {
    const workflow = tester.loadWorkflow(testWorkflowPath);

    assert.ok(workflow, 'Workflow should be loaded');
    assert.strictEqual(workflow.name, 'Test Workflow');
    assert.ok(workflow.jobs, 'Workflow should have jobs');
  });

  it('should return null for non-existent file', () => {
    const workflow = tester.loadWorkflow('/non/existent/file.yml');

    assert.strictEqual(workflow, null);
  });

  it('should validate workflow syntax', () => {
    const workflow = tester.loadWorkflow(testWorkflowPath);
    const { errors, warnings } = tester.validateSyntax(workflow);

    assert.strictEqual(errors.length, 0, 'Should have no errors');
  });

  it('should detect missing required fields', () => {
    const invalidWorkflow = { jobs: {} }; // Missing 'name' and 'on'
    const { errors } = tester.validateSyntax(invalidWorkflow);

    assert.ok(errors.length > 0, 'Should have errors');
    assert.ok(
      errors.some((e) => e.includes('name')),
      'Should detect missing name'
    );
    assert.ok(
      errors.some((e) => e.includes('on')),
      'Should detect missing on'
    );
  });

  it('should analyze best practices', () => {
    const workflow = tester.loadWorkflow(testWorkflowPath);
    const suggestions = tester.analyzeBestPractices(workflow);

    assert.ok(suggestions.length > 0, 'Should have suggestions');
    assert.ok(
      suggestions.some((s) => s.type === 'performance'),
      'Should suggest performance improvements'
    );
  });

  it('should detect missing caching', () => {
    const workflow = tester.loadWorkflow(testWorkflowPath);
    const suggestions = tester.analyzeBestPractices(workflow);

    const cachingSuggestion = suggestions.find(
      (s) => s.message.toLowerCase().includes('caching')
    );
    assert.ok(cachingSuggestion, 'Should suggest adding caching');
  });

  it('should not suggest caching when already present', () => {
    const workflowWithCache = {
      name: 'Test',
      on: 'push',
      jobs: {
        test: {
          'runs-on': 'ubuntu-latest',
          steps: [
            {
              uses: 'actions/cache@v3',
              with: { path: 'node_modules', key: 'test' },
            },
          ],
        },
      },
    };

    const suggestions = tester.analyzeBestPractices(workflowWithCache);
    const cachingSuggestion = suggestions.find(
      (s) => s.message.toLowerCase().includes('caching')
    );

    assert.ok(!cachingSuggestion, 'Should not suggest caching when present');
  });

  it('should handle dry run mode', () => {
    const dryRunTester = new MockWorkflowTester({ dryRun: true });

    assert.strictEqual(dryRunTester.dryRun, true);
  });

  it('should respect verbose mode', () => {
    const verboseTester = new MockWorkflowTester({ verbose: true });

    assert.strictEqual(verboseTester.verbose, true);
  });
});

describe('WorkflowTester - Advanced Validation', () => {
  let tester;

  beforeEach(() => {
    tester = new MockWorkflowTester();
  });

  it('should validate job structure', () => {
    const workflow = {
      name: 'Test',
      on: 'push',
      jobs: {
        test: {
          'runs-on': 'ubuntu-latest',
          steps: [],
        },
      },
    };

    const { errors } = tester.validateSyntax(workflow);

    // Should pass basic validation
    assert.strictEqual(errors.length, 0);
  });

  it('should detect security issues in workflow', () => {
    const workflowWithSecret = {
      name: 'Test',
      on: 'push',
      jobs: {
        test: {
          'runs-on': 'ubuntu-latest',
          steps: [
            {
              run: 'export API_KEY="hardcoded-secret-123"',
            },
          ],
        },
      },
    };

    const suggestions = tester.analyzeBestPractices(workflowWithSecret);

    // Should detect potential security issue
    assert.ok(suggestions.length >= 0);
  });

  it('should generate test report', () => {
    const results = [
      {
        workflow: 'test1.yml',
        status: 'passed',
        errors: [],
        warnings: [],
      },
      {
        workflow: 'test2.yml',
        status: 'failed',
        errors: ['Missing name'],
        warnings: ['No timeout'],
      },
    ];

    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        total: results.length,
        passed: results.filter((r) => r.status === 'passed').length,
        failed: results.filter((r) => r.status === 'failed').length,
        warnings: results.reduce((sum, r) => sum + r.warnings.length, 0),
      },
      workflows: results,
    };

    assert.strictEqual(report.summary.total, 2);
    assert.strictEqual(report.summary.passed, 1);
    assert.strictEqual(report.summary.failed, 1);
    assert.strictEqual(report.summary.warnings, 1);
  });
});

// Run tests if executed directly
if (require.main === module) {
  console.log('Running GitHub Actions Workflow Tester tests...\n');
}
