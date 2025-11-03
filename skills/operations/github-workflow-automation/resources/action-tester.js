#!/usr/bin/env node
/**
 * GitHub Actions Workflow Tester
 * Test and validate workflows locally before committing
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

class WorkflowTester {
  constructor(options = {}) {
    this.workflowDir = options.workflowDir || '.github/workflows';
    this.verbose = options.verbose || false;
    this.dryRun = options.dryRun || false;
  }

  /**
   * Load workflow file
   */
  loadWorkflow(workflowPath) {
    try {
      const content = fs.readFileSync(workflowPath, 'utf8');
      return yaml.load(content);
    } catch (error) {
      this.log(`Error loading workflow: ${error.message}`, 'error');
      return null;
    }
  }

  /**
   * Validate workflow syntax
   */
  validateSyntax(workflow, workflowPath) {
    const errors = [];
    const warnings = [];

    // Check required fields
    if (!workflow.name) {
      errors.push('Missing required field: name');
    }

    if (!workflow.on) {
      errors.push('Missing required field: on');
    }

    if (!workflow.jobs || Object.keys(workflow.jobs).length === 0) {
      errors.push('No jobs defined');
    }

    // Check job structure
    if (workflow.jobs) {
      Object.entries(workflow.jobs).forEach(([jobName, job]) => {
        if (!job['runs-on']) {
          errors.push(`Job '${jobName}': Missing 'runs-on' field`);
        }

        if (!job.steps || job.steps.length === 0) {
          errors.push(`Job '${jobName}': No steps defined`);
        }

        // Check for missing timeouts
        if (!job['timeout-minutes']) {
          warnings.push(`Job '${jobName}': No timeout configured`);
        }

        // Validate step structure
        if (job.steps) {
          job.steps.forEach((step, index) => {
            if (!step.name && !step.uses && !step.run) {
              errors.push(`Job '${jobName}', step ${index}: Invalid step structure`);
            }
          });
        }
      });
    }

    return { errors, warnings };
  }

  /**
   * Analyze workflow for best practices
   */
  analyzeBestPractices(workflow) {
    const suggestions = [];

    // Check for caching
    const hasCaching = JSON.stringify(workflow).includes('actions/cache');
    if (!hasCaching) {
      suggestions.push({
        type: 'performance',
        message: 'Consider adding dependency caching to improve performance',
        priority: 'high',
      });
    }

    // Check for secrets hardcoding
    const workflowStr = JSON.stringify(workflow);
    const secretPatterns = [
      /password\s*[:=]\s*["'][^"']+["']/i,
      /api[_-]?key\s*[:=]\s*["'][^"']+["']/i,
      /token\s*[:=]\s*["'][^"']+["']/i,
    ];

    secretPatterns.forEach((pattern) => {
      if (pattern.test(workflowStr) && !workflowStr.includes('secrets.')) {
        suggestions.push({
          type: 'security',
          message: 'Potential hardcoded secret detected - use GitHub secrets',
          priority: 'critical',
        });
      }
    });

    // Check for explicit permissions
    if (!workflow.permissions) {
      suggestions.push({
        type: 'security',
        message: 'Set explicit permissions for better security',
        priority: 'medium',
      });
    }

    // Check for matrix optimization
    if (workflow.jobs) {
      Object.entries(workflow.jobs).forEach(([jobName, job]) => {
        if (job.strategy?.matrix && !job.strategy['fail-fast']) {
          suggestions.push({
            type: 'reliability',
            message: `Job '${jobName}': Configure fail-fast behavior in matrix strategy`,
            priority: 'medium',
          });
        }
      });
    }

    // Check for conditional execution
    const hasConditionals = workflowStr.includes('"if":');
    if (!hasConditionals) {
      suggestions.push({
        type: 'cost',
        message: 'Consider adding conditional execution to skip unnecessary jobs',
        priority: 'low',
      });
    }

    return suggestions;
  }

  /**
   * Test workflow with act (local GitHub Actions runner)
   */
  async testWithAct(workflowPath) {
    if (this.dryRun) {
      this.log('Dry run mode - skipping actual execution', 'info');
      return true;
    }

    try {
      // Check if act is installed
      execSync('act --version', { stdio: 'ignore' });
    } catch (error) {
      this.log('act not installed. Install from: https://github.com/nektos/act', 'warning');
      return false;
    }

    return new Promise((resolve) => {
      this.log(`Testing workflow with act: ${workflowPath}`, 'info');

      const actProcess = spawn('act', ['-n', '--workflows', workflowPath], {
        stdio: 'inherit',
      });

      actProcess.on('close', (code) => {
        if (code === 0) {
          this.log('Workflow test passed', 'success');
          resolve(true);
        } else {
          this.log(`Workflow test failed with code ${code}`, 'error');
          resolve(false);
        }
      });
    });
  }

  /**
   * Generate test report
   */
  generateReport(results) {
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        total: results.length,
        passed: results.filter((r) => r.status === 'passed').length,
        failed: results.filter((r) => r.status === 'failed').length,
        warnings: results.reduce((sum, r) => sum + (r.warnings?.length || 0), 0),
      },
      workflows: results,
    };

    return report;
  }

  /**
   * Test all workflows in directory
   */
  async testAllWorkflows() {
    const workflows = fs
      .readdirSync(this.workflowDir)
      .filter((f) => f.endsWith('.yml') || f.endsWith('.yaml'));

    if (workflows.length === 0) {
      this.log('No workflows found', 'warning');
      return;
    }

    const results = [];

    for (const workflowFile of workflows) {
      const workflowPath = path.join(this.workflowDir, workflowFile);
      this.log(`\nTesting workflow: ${workflowFile}`, 'info');

      const workflow = this.loadWorkflow(workflowPath);
      if (!workflow) {
        results.push({
          workflow: workflowFile,
          status: 'failed',
          errors: ['Failed to load workflow file'],
        });
        continue;
      }

      // Validate syntax
      const { errors, warnings } = this.validateSyntax(workflow, workflowPath);
      if (errors.length > 0) {
        this.log(`Syntax errors found:`, 'error');
        errors.forEach((err) => this.log(`  - ${err}`, 'error'));
        results.push({
          workflow: workflowFile,
          status: 'failed',
          errors,
          warnings,
        });
        continue;
      }

      if (warnings.length > 0) {
        this.log(`Warnings:`, 'warning');
        warnings.forEach((warn) => this.log(`  - ${warn}`, 'warning'));
      }

      // Analyze best practices
      const suggestions = this.analyzeBestPractices(workflow);
      if (suggestions.length > 0) {
        this.log(`Best practice suggestions:`, 'info');
        suggestions.forEach((sug) => {
          const icon = sug.priority === 'critical' ? '🔴' : sug.priority === 'high' ? '🟡' : '🔵';
          this.log(`  ${icon} [${sug.type}] ${sug.message}`, 'info');
        });
      }

      // Test with act if available
      const actPassed = await this.testWithAct(workflowPath);

      results.push({
        workflow: workflowFile,
        status: errors.length === 0 && actPassed ? 'passed' : 'failed',
        errors,
        warnings,
        suggestions,
      });
    }

    // Generate and display report
    const report = this.generateReport(results);
    this.displayReport(report);

    return report;
  }

  /**
   * Display test report
   */
  displayReport(report) {
    console.log('\n' + '='.repeat(60));
    console.log('GitHub Actions Workflow Test Report');
    console.log('='.repeat(60));
    console.log(`Timestamp: ${report.timestamp}`);
    console.log(`\nSummary:`);
    console.log(`  Total workflows: ${report.summary.total}`);
    console.log(`  ✓ Passed: ${report.summary.passed}`);
    console.log(`  ✗ Failed: ${report.summary.failed}`);
    console.log(`  ⚠ Warnings: ${report.summary.warnings}`);
    console.log('='.repeat(60) + '\n');

    // Save report to file
    const reportPath = 'workflow-test-report.json';
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    this.log(`Report saved to: ${reportPath}`, 'success');
  }

  /**
   * Log message with color
   */
  log(message, level = 'info') {
    const colors = {
      info: '\x1b[34m',    // Blue
      success: '\x1b[32m', // Green
      warning: '\x1b[33m', // Yellow
      error: '\x1b[31m',   // Red
      reset: '\x1b[0m',
    };

    const icons = {
      info: 'ℹ',
      success: '✓',
      warning: '⚠',
      error: '✗',
    };

    if (this.verbose || level !== 'info') {
      console.log(`${colors[level]}${icons[level]} ${message}${colors.reset}`);
    }
  }
}

// CLI execution
if (require.main === module) {
  const args = process.argv.slice(2);
  const options = {
    verbose: args.includes('--verbose') || args.includes('-v'),
    dryRun: args.includes('--dry-run'),
    workflowDir: args.includes('--workflow-dir')
      ? args[args.indexOf('--workflow-dir') + 1]
      : '.github/workflows',
  };

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
GitHub Actions Workflow Tester

Usage: node action-tester.js [OPTIONS]

Options:
  --workflow-dir DIR    Workflow directory (default: .github/workflows)
  --verbose, -v         Verbose output
  --dry-run             Dry run mode (skip act execution)
  --help, -h            Show this help message

Examples:
  node action-tester.js
  node action-tester.js --verbose
  node action-tester.js --workflow-dir custom/workflows
    `);
    process.exit(0);
  }

  const tester = new WorkflowTester(options);
  tester.testAllWorkflows().catch((error) => {
    console.error('Test execution failed:', error);
    process.exit(1);
  });
}

module.exports = WorkflowTester;
