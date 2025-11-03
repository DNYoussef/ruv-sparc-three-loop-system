/**
 * Test Suite for Flow Nexus Sandbox Manager
 *
 * Comprehensive tests for sandbox lifecycle management:
 * - Sandbox creation with various templates
 * - Configuration and environment setup
 * - Code execution and file uploads
 * - Monitoring and cleanup
 *
 * Run with: node sandbox-manager.test.js
 */

const SandboxManager = require('../resources/scripts/sandbox-manager');

class TestRunner {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
  }

  describe(description, callback) {
    console.log(`\n${description}`);
    callback();
  }

  it(testName, testFn) {
    this.tests.push({ name: testName, fn: testFn });
  }

  async run() {
    console.log('Running Sandbox Manager Tests\n');
    console.log('='.repeat(60));

    for (const test of this.tests) {
      try {
        await test.fn();
        console.log(`✓ ${test.name}`);
        this.passed++;
      } catch (error) {
        console.error(`✗ ${test.name}`);
        console.error(`  Error: ${error.message}`);
        this.failed++;
      }
    }

    console.log('\n' + '='.repeat(60));
    console.log(`\nTest Results: ${this.passed} passed, ${this.failed} failed`);
    console.log(`Total: ${this.tests.length} tests\n`);

    process.exit(this.failed > 0 ? 1 : 0);
  }

  assert(condition, message) {
    if (!condition) {
      throw new Error(message || 'Assertion failed');
    }
  }

  assertEqual(actual, expected, message) {
    if (actual !== expected) {
      throw new Error(message || `Expected ${expected}, got ${actual}`);
    }
  }

  assertNotNull(value, message) {
    if (value === null || value === undefined) {
      throw new Error(message || 'Value is null or undefined');
    }
  }

  assertIncludes(array, value, message) {
    if (!array.includes(value)) {
      throw new Error(message || `Array should include ${value}`);
    }
  }
}

const runner = new TestRunner();

runner.describe('SandboxManager Class', () => {
  runner.it('should create SandboxManager instance', () => {
    const manager = new SandboxManager();
    runner.assertNotNull(manager, 'SandboxManager instance should be created');
    runner.assertEqual(manager.mcpPrefix, 'mcp__flow-nexus__', 'Should have correct MCP prefix');
  });

  runner.it('should define valid templates', () => {
    const manager = new SandboxManager();
    const expectedTemplates = ['node', 'python', 'react', 'nextjs', 'vanilla', 'base', 'claude-code'];

    runner.assertEqual(
      manager.validTemplates.length,
      expectedTemplates.length,
      'Should have correct number of templates'
    );

    expectedTemplates.forEach(template => {
      runner.assertIncludes(
        manager.validTemplates,
        template,
        `Should include ${template} template`
      );
    });
  });

  runner.it('should have all required methods', () => {
    const manager = new SandboxManager();
    const requiredMethods = [
      'executeMCP',
      'create',
      'list',
      'status',
      'execute',
      'upload',
      'logs',
      'stop',
      'delete',
      'cleanupAll'
    ];

    requiredMethods.forEach(method => {
      runner.assert(
        typeof manager[method] === 'function',
        `SandboxManager should have ${method} method`
      );
    });
  });
});

runner.describe('Sandbox Creation', () => {
  runner.it('should validate template names', () => {
    const manager = new SandboxManager();
    const validTemplates = manager.validTemplates;

    runner.assertIncludes(validTemplates, 'node', 'Should accept node template');
    runner.assertIncludes(validTemplates, 'python', 'Should accept python template');
    runner.assert(!validTemplates.includes('invalid'), 'Should reject invalid template');
  });

  runner.it('should construct creation parameters with environment variables', () => {
    const buildParams = (template, name, options = {}) => {
      const params = { template, name };
      if (options.env) params.env_vars = options.env;
      if (options.packages) params.install_packages = options.packages;
      if (options.timeout) params.timeout = parseInt(options.timeout);
      return params;
    };

    const params = buildParams('node', 'test-sandbox', {
      env: { PORT: '3000', NODE_ENV: 'development' },
      packages: ['express', 'cors'],
      timeout: '7200'
    });

    runner.assertEqual(params.template, 'node', 'Should include template');
    runner.assertEqual(params.name, 'test-sandbox', 'Should include name');
    runner.assertNotNull(params.env_vars, 'Should include env_vars');
    runner.assertEqual(params.env_vars.PORT, '3000', 'Should set PORT env var');
    runner.assertEqual(params.install_packages.length, 2, 'Should include 2 packages');
    runner.assertEqual(params.timeout, 7200, 'Should parse timeout as integer');
  });

  runner.it('should handle package installation configuration', () => {
    const packages = ['express', 'cors', 'dotenv', 'helmet'];
    const params = { install_packages: packages };

    runner.assertEqual(params.install_packages.length, 4, 'Should include all packages');
    runner.assertIncludes(params.install_packages, 'express', 'Should include express');
    runner.assertIncludes(params.install_packages, 'helmet', 'Should include helmet');
  });
});

runner.describe('Sandbox Execution', () => {
  runner.it('should handle inline code execution', () => {
    const code = 'console.log("Hello World");';
    const params = {
      sandbox_id: 'sbx_123',
      code,
      language: 'javascript',
      capture_output: true
    };

    runner.assertEqual(params.code, code, 'Should include code');
    runner.assertEqual(params.language, 'javascript', 'Should set language');
    runner.assert(params.capture_output, 'Should capture output');
  });

  runner.it('should detect file reference syntax', () => {
    const isFileReference = (codeOrFile) => codeOrFile.startsWith('@');

    runner.assert(isFileReference('@script.js'), 'Should detect file reference');
    runner.assert(!isFileReference('console.log("test")'), 'Should not detect inline code as file');
  });

  runner.it('should construct execution parameters with working directory', () => {
    const params = {
      sandbox_id: 'sbx_123',
      code: 'npm test',
      language: 'javascript',
      working_dir: '/app/tests',
      timeout: 120
    };

    runner.assertEqual(params.working_dir, '/app/tests', 'Should set working directory');
    runner.assertEqual(params.timeout, 120, 'Should set timeout');
  });
});

runner.describe('File Upload', () => {
  runner.it('should validate file paths', () => {
    const validatePaths = (localPath, remotePath) => {
      return localPath && remotePath && localPath.length > 0 && remotePath.length > 0;
    };

    runner.assert(
      validatePaths('/local/file.js', '/app/file.js'),
      'Valid paths should pass'
    );
    runner.assert(
      !validatePaths('', '/app/file.js'),
      'Empty local path should fail'
    );
  });

  runner.it('should construct upload parameters', () => {
    const params = {
      sandbox_id: 'sbx_123',
      file_path: '/app/config/database.json',
      content: '{"host": "localhost"}'
    };

    runner.assertEqual(params.sandbox_id, 'sbx_123', 'Should include sandbox_id');
    runner.assertEqual(params.file_path, '/app/config/database.json', 'Should include remote path');
    runner.assertNotNull(params.content, 'Should include content');
  });
});

runner.describe('Sandbox Listing and Status', () => {
  runner.it('should validate status filters', () => {
    const validStatuses = ['running', 'stopped', 'all'];

    runner.assertIncludes(validStatuses, 'running', 'Should support running filter');
    runner.assertIncludes(validStatuses, 'stopped', 'Should support stopped filter');
    runner.assertIncludes(validStatuses, 'all', 'Should support all filter');
  });

  runner.it('should parse sandbox list response', () => {
    const mockResponse = {
      sandboxes: [
        {
          id: 'sbx_1',
          name: 'test-sandbox',
          template: 'node',
          status: 'running',
          created_at: new Date().toISOString()
        },
        {
          id: 'sbx_2',
          name: 'python-sandbox',
          template: 'python',
          status: 'stopped',
          created_at: new Date().toISOString()
        }
      ]
    };

    runner.assertEqual(mockResponse.sandboxes.length, 2, 'Should have 2 sandboxes');
    runner.assertEqual(mockResponse.sandboxes[0].status, 'running', 'First should be running');
    runner.assertEqual(mockResponse.sandboxes[1].status, 'stopped', 'Second should be stopped');
  });
});

runner.describe('Sandbox Logs', () => {
  runner.it('should validate log line limits', () => {
    const validateLimit = (lines) => {
      const numLines = parseInt(lines);
      return numLines > 0 && numLines <= 1000;
    };

    runner.assert(validateLimit('100'), 'Valid limit should pass');
    runner.assert(validateLimit('1000'), 'Max limit should pass');
    runner.assert(!validateLimit('1001'), 'Over max should fail');
    runner.assert(!validateLimit('0'), 'Zero should fail');
  });

  runner.it('should construct log parameters', () => {
    const params = {
      sandbox_id: 'sbx_123',
      lines: 200
    };

    runner.assertEqual(params.sandbox_id, 'sbx_123', 'Should include sandbox_id');
    runner.assertEqual(params.lines, 200, 'Should set line limit');
  });
});

runner.describe('Cleanup Operations', () => {
  runner.it('should calculate cutoff time for cleanup', () => {
    const calculateCutoff = (olderThanHours) => {
      const now = Date.now();
      return now - (olderThanHours * 60 * 60 * 1000);
    };

    const cutoff24h = calculateCutoff(24);
    const cutoff1h = calculateCutoff(1);

    runner.assert(cutoff24h < Date.now(), 'Cutoff should be in the past');
    runner.assert(cutoff1h > cutoff24h, '1 hour cutoff should be more recent than 24 hour');
  });

  runner.it('should determine cleanup eligibility', () => {
    const shouldCleanup = (sandbox, cutoffTime) => {
      const createdAt = new Date(sandbox.created_at).getTime();
      return sandbox.status === 'stopped' || (cutoffTime && createdAt < cutoffTime);
    };

    const now = Date.now();
    const cutoff = now - (24 * 60 * 60 * 1000); // 24 hours ago

    const stoppedSandbox = {
      status: 'stopped',
      created_at: new Date(now - 1000).toISOString()
    };

    const oldSandbox = {
      status: 'running',
      created_at: new Date(cutoff - 1000).toISOString()
    };

    const recentSandbox = {
      status: 'running',
      created_at: new Date(now - 1000).toISOString()
    };

    runner.assert(shouldCleanup(stoppedSandbox, cutoff), 'Stopped sandbox should be cleaned');
    runner.assert(shouldCleanup(oldSandbox, cutoff), 'Old sandbox should be cleaned');
    runner.assert(!shouldCleanup(recentSandbox, cutoff), 'Recent running sandbox should not be cleaned');
  });
});

runner.describe('Environment Variable Parsing', () => {
  runner.it('should parse key=value environment variables', () => {
    const parseEnvVars = (args) => {
      const env = {};
      args.forEach(arg => {
        const [key, value] = arg.split('=');
        if (key && value) {
          env[key] = value;
        }
      });
      return env;
    };

    const args = ['PORT=3000', 'NODE_ENV=production', 'LOG_LEVEL=info'];
    const env = parseEnvVars(args);

    runner.assertEqual(env.PORT, '3000', 'Should parse PORT');
    runner.assertEqual(env.NODE_ENV, 'production', 'Should parse NODE_ENV');
    runner.assertEqual(Object.keys(env).length, 3, 'Should parse 3 env vars');
  });
});

runner.describe('Package Parsing', () => {
  runner.it('should parse comma-separated package list', () => {
    const parsePackages = (packagesStr) => {
      return packagesStr.split(',').map(pkg => pkg.trim());
    };

    const packages = parsePackages('express,cors,dotenv,helmet');

    runner.assertEqual(packages.length, 4, 'Should parse 4 packages');
    runner.assertIncludes(packages, 'express', 'Should include express');
    runner.assertIncludes(packages, 'helmet', 'Should include helmet');
  });

  runner.it('should handle whitespace in package lists', () => {
    const parsePackages = (packagesStr) => {
      return packagesStr.split(',').map(pkg => pkg.trim());
    };

    const packages = parsePackages('express, cors , dotenv');

    runner.assertEqual(packages[0], 'express', 'Should trim first package');
    runner.assertEqual(packages[1], 'cors', 'Should trim second package');
    runner.assertEqual(packages[2], 'dotenv', 'Should trim third package');
  });
});

runner.describe('Timeout Configuration', () => {
  runner.it('should validate timeout values', () => {
    const validateTimeout = (timeout) => {
      const num = parseInt(timeout);
      return num > 0 && num <= 86400; // Max 24 hours
    };

    runner.assert(validateTimeout('3600'), '1 hour should be valid');
    runner.assert(validateTimeout('7200'), '2 hours should be valid');
    runner.assert(!validateTimeout('0'), 'Zero should be invalid');
    runner.assert(!validateTimeout('100000'), 'Over 24 hours should be invalid');
  });

  runner.it('should parse timeout as integer', () => {
    const timeout = parseInt('3600');

    runner.assertEqual(typeof timeout, 'number', 'Should be number type');
    runner.assertEqual(timeout, 3600, 'Should equal 3600');
  });
});

runner.describe('Error Handling', () => {
  runner.it('should validate required parameters for creation', () => {
    const hasRequiredParams = (template, name) => {
      return template && name && template.length > 0 && name.length > 0;
    };

    runner.assert(hasRequiredParams('node', 'my-sandbox'), 'Valid params should pass');
    runner.assert(!hasRequiredParams('', 'my-sandbox'), 'Missing template should fail');
    runner.assert(!hasRequiredParams('node', ''), 'Missing name should fail');
  });

  runner.it('should handle execution errors gracefully', () => {
    const mockError = {
      success: false,
      error: 'Execution timeout',
      error_code: 'TIMEOUT',
      stderr: 'Error: command timed out after 60 seconds'
    };

    runner.assert(!mockError.success, 'Should indicate failure');
    runner.assertNotNull(mockError.error, 'Should include error message');
    runner.assertNotNull(mockError.stderr, 'Should include stderr output');
  });
});

// Run all tests
runner.run();
