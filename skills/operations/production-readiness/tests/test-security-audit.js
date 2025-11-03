#!/usr/bin/env node
/**
 * Tests for Security Audit Script
 */

const assert = require('assert');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');
const SecurityAuditor = require('../resources/security-audit.js');

describe('SecurityAuditor', function() {
  this.timeout(10000); // Increase timeout for file operations

  let testDir;
  let auditor;

  beforeEach(async function() {
    // Create temporary test directory
    testDir = await fs.mkdtemp(path.join(os.tmpdir(), 'security-test-'));
    auditor = new SecurityAuditor(testDir, { deep: false });
  });

  afterEach(async function() {
    // Clean up test directory
    try {
      await fs.rm(testDir, { recursive: true, force: true });
    } catch (error) {
      // Ignore cleanup errors
    }
  });

  describe('Initialization', function() {
    it('should initialize with correct options', function() {
      assert.strictEqual(auditor.targetPath, testDir);
      assert.strictEqual(auditor.options.checkSecrets, true);
      assert.strictEqual(auditor.options.checkDependencies, true);
      assert.strictEqual(auditor.options.deep, false);
    });

    it('should initialize findings object', function() {
      assert.ok(auditor.findings);
      assert.ok(Array.isArray(auditor.findings.critical));
      assert.ok(Array.isArray(auditor.findings.high));
      assert.ok(Array.isArray(auditor.findings.medium));
      assert.ok(Array.isArray(auditor.findings.low));
    });
  });

  describe('Hardcoded Secrets Detection', function() {
    it('should detect hardcoded API keys', async function() {
      const srcDir = path.join(testDir, 'src');
      await fs.mkdir(srcDir, { recursive: true });

      const badCode = `
        const config = {
          apiKey: 'sk-abcdef123456789012345678901234567890123456',
          password: 'mysecretpassword123'
        };
      `;

      await fs.writeFile(path.join(srcDir, 'config.js'), badCode);

      await auditor.checkHardcodedSecrets();

      const criticalFindings = auditor.findings.critical;
      assert.ok(criticalFindings.length > 0, 'Should detect hardcoded secrets');
      assert.ok(criticalFindings.some(f => f.type.includes('API Key')));
    });

    it('should not flag secrets in test files', async function() {
      const testDir = path.join(testDir, 'tests');
      await fs.mkdir(testDir, { recursive: true });

      const testCode = `
        test('API key validation', () => {
          const fakeKey = 'sk-test123456789012345678901234567890123456';
          expect(validateApiKey(fakeKey)).toBe(true);
        });
      `;

      await fs.writeFile(path.join(testDir, 'api.test.js'), testCode);

      await auditor.checkHardcodedSecrets();

      // Should not detect secrets in test files
      const criticalFindings = auditor.findings.critical;
      assert.strictEqual(criticalFindings.length, 0, 'Should ignore test files');
    });

    it('should detect AWS access keys', async function() {
      const srcDir = path.join(testDir, 'src');
      await fs.mkdir(srcDir, { recursive: true });

      const badCode = `
        const awsConfig = {
          accessKeyId: 'AKIAIOSFODNN7EXAMPLE',
          secretAccessKey: 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
        };
      `;

      await fs.writeFile(path.join(srcDir, 'aws-config.js'), badCode);

      await auditor.checkHardcodedSecrets();

      const criticalFindings = auditor.findings.critical;
      assert.ok(criticalFindings.some(f => f.type.includes('AWS')));
    });
  });

  describe('Security Headers Check', function() {
    it('should detect missing security headers', async function() {
      const srcDir = path.join(testDir, 'src');
      await fs.mkdir(srcDir, { recursive: true });

      const serverCode = `
        const express = require('express');
        const app = express();

        app.get('/', (req, res) => {
          res.send('Hello World');
        });
      `;

      await fs.writeFile(path.join(srcDir, 'server.js'), serverCode);

      await auditor.checkSecurityHeaders();

      const mediumFindings = auditor.findings.medium;
      assert.ok(mediumFindings.some(f => f.type === 'Missing Security Headers'));
    });

    it('should pass when security headers are configured', async function() {
      const srcDir = path.join(testDir, 'src');
      await fs.mkdir(srcDir, { recursive: true });

      const serverCode = `
        const express = require('express');
        const helmet = require('helmet');
        const app = express();

        app.use(helmet());
        app.use((req, res, next) => {
          res.setHeader('X-Content-Type-Options', 'nosniff');
          res.setHeader('X-Frame-Options', 'DENY');
          res.setHeader('X-XSS-Protection', '1; mode=block');
          res.setHeader('Strict-Transport-Security', 'max-age=31536000');
          res.setHeader('Content-Security-Policy', "default-src 'self'");
          next();
        });
      `;

      await fs.writeFile(path.join(srcDir, 'server.js'), serverCode);

      await auditor.checkSecurityHeaders();

      const mediumFindings = auditor.findings.medium.filter(f => f.type === 'Missing Security Headers');
      assert.strictEqual(mediumFindings.length, 0, 'Should not report missing headers when configured');
    });
  });

  describe('SQL Injection Detection', function() {
    it('should detect potential SQL injection', async function() {
      const srcDir = path.join(testDir, 'src');
      await fs.mkdir(srcDir, { recursive: true });

      const vulnerableCode = `
        app.get('/users/:id', (req, res) => {
          const query = 'SELECT * FROM users WHERE id = ' + req.params.id;
          db.execute(query);
        });
      `;

      await fs.writeFile(path.join(srcDir, 'users.js'), vulnerableCode);

      await auditor.checkSQLInjectionRisks();

      const highFindings = auditor.findings.high;
      assert.ok(highFindings.some(f => f.type === 'Potential SQL Injection'));
    });

    it('should not flag parameterized queries', async function() {
      const srcDir = path.join(testDir, 'src');
      await fs.mkdir(srcDir, { recursive: true });

      const safeCode = `
        app.get('/users/:id', (req, res) => {
          const query = 'SELECT * FROM users WHERE id = ?';
          db.execute(query, [req.params.id]);
        });
      `;

      await fs.writeFile(path.join(srcDir, 'users.js'), safeCode);

      await auditor.checkSQLInjectionRisks();

      const highFindings = auditor.findings.high.filter(f => f.type === 'Potential SQL Injection');
      assert.strictEqual(highFindings.length, 0, 'Should not flag parameterized queries');
    });
  });

  describe('XSS Detection', function() {
    it('should detect dangerouslySetInnerHTML', async function() {
      const srcDir = path.join(testDir, 'src');
      await fs.mkdir(srcDir, { recursive: true });

      const vulnerableCode = `
        function UserComment({ comment }) {
          return <div dangerouslySetInnerHTML={{ __html: comment.text }} />;
        }
      `;

      await fs.writeFile(path.join(srcDir, 'Comment.jsx'), vulnerableCode);

      await auditor.checkXSSVulnerabilities();

      const mediumFindings = auditor.findings.medium;
      assert.ok(mediumFindings.some(f => f.type === 'Potential XSS Vulnerability'));
    });
  });

  describe('Authentication Pattern Check', function() {
    it('should detect weak JWT secrets', async function() {
      const srcDir = path.join(testDir, 'src');
      await fs.mkdir(srcDir, { recursive: true });

      const weakAuthCode = `
        const jwt = require('jsonwebtoken');
        const token = jwt.sign({ userId: 123 }, 'weak', { expiresIn: '1h' });
      `;

      await fs.writeFile(path.join(srcDir, 'auth.js'), weakAuthCode);

      await auditor.checkAuthenticationPatterns();

      const highFindings = auditor.findings.high;
      assert.ok(highFindings.some(f => f.type === 'Authentication Security Issues'));
    });

    it('should detect insecure session configuration', async function() {
      const srcDir = path.join(testDir, 'src');
      await fs.mkdir(srcDir, { recursive: true });

      const insecureSessionCode = `
        const session = require('express-session');
        app.use(session({
          secret: 'keyboard cat',
          cookie: { secure: false, httpOnly: false }
        }));
      `;

      await fs.writeFile(path.join(srcDir, 'session.js'), insecureSessionCode);

      await auditor.checkAuthenticationPatterns();

      const highFindings = auditor.findings.high;
      assert.ok(highFindings.some(f => f.issues && f.issues.length > 0));
    });
  });

  describe('Report Generation', function() {
    it('should generate complete report', async function() {
      const report = auditor.generateReport();

      assert.ok(report.timestamp);
      assert.ok(report.summary);
      assert.strictEqual(typeof report.summary.critical, 'number');
      assert.strictEqual(typeof report.summary.high, 'number');
      assert.strictEqual(typeof report.summary.medium, 'number');
      assert.ok(report.findings);
      assert.strictEqual(typeof report.passed, 'boolean');
    });

    it('should mark as passed when no critical/high issues', async function() {
      const report = auditor.generateReport();
      assert.strictEqual(report.passed, true);
    });

    it('should mark as failed when critical issues exist', async function() {
      auditor.addFinding('critical', {
        type: 'Test Critical Issue',
        recommendation: 'Fix it'
      });

      const report = auditor.generateReport();
      assert.strictEqual(report.passed, false);
    });
  });

  describe('Helper Methods', function() {
    it('should add findings correctly', function() {
      auditor.addFinding('critical', { type: 'Test', message: 'Test critical' });
      auditor.addFinding('high', { type: 'Test', message: 'Test high' });

      assert.strictEqual(auditor.findings.critical.length, 1);
      assert.strictEqual(auditor.findings.high.length, 1);
    });

    it('should count findings correctly', function() {
      auditor.addFinding('critical', { type: 'Test1' });
      auditor.addFinding('high', { type: 'Test2' });
      auditor.addFinding('medium', { type: 'Test3' });

      const count = auditor.countFindings(['critical', 'high']);
      assert.strictEqual(count, 2);
    });

    it('should filter findings by type', function() {
      auditor.addFinding('critical', { type: 'SQL Injection' });
      auditor.addFinding('high', { type: 'XSS' });
      auditor.addFinding('critical', { type: 'SQL Injection' });

      const sqlFindings = auditor.findingsByType('SQL Injection');
      assert.strictEqual(sqlFindings.length, 2);
    });
  });

  describe('Full Audit Integration', function() {
    it('should run complete audit without errors', async function() {
      // Create minimal project structure
      await fs.mkdir(path.join(testDir, 'src'), { recursive: true });
      await fs.writeFile(
        path.join(testDir, 'package.json'),
        JSON.stringify({ name: 'test', dependencies: {} })
      );

      const report = await auditor.audit();

      assert.ok(report);
      assert.ok(report.summary);
      assert.ok(report.findings);
      assert.strictEqual(typeof report.passed, 'boolean');
    });
  });
});

// Run tests if this is the main module
if (require.main === module) {
  const Mocha = require('mocha');
  const mocha = new Mocha();

  mocha.suite.emit('pre-require', global, null, mocha);

  // Run the tests
  mocha.run(failures => {
    process.exitCode = failures ? 1 : 0;
  });
}

module.exports = { SecurityAuditor };
