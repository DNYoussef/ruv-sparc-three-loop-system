#!/usr/bin/env node
/**
 * Security Audit Script
 * Comprehensive security vulnerability scanning for production deployment
 */

const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');

class SecurityAuditor {
  constructor(targetPath, options = {}) {
    this.targetPath = targetPath;
    this.options = {
      checkSecrets: options.checkSecrets !== false,
      checkDependencies: options.checkDependencies !== false,
      checkHeaders: options.checkHeaders !== false,
      checkAuth: options.checkAuth !== false,
      deep: options.deep || false,
    };
    this.findings = {
      critical: [],
      high: [],
      medium: [],
      low: [],
      info: [],
    };
  }

  /**
   * Run complete security audit
   */
  async audit() {
    console.log('\n========================================');
    console.log('Security Deep-Dive Audit');
    console.log('========================================\n');

    await this.checkHardcodedSecrets();
    await this.checkDependencyVulnerabilities();
    await this.checkSecurityHeaders();
    await this.checkAuthenticationPatterns();
    await this.checkSQLInjectionRisks();
    await this.checkXSSVulnerabilities();
    await this.checkCSRFProtection();
    await this.checkFileUploadSecurity();

    return this.generateReport();
  }

  /**
   * CRITICAL: Check for hardcoded secrets
   */
  async checkHardcodedSecrets() {
    if (!this.options.checkSecrets) return;

    console.log('[1/8] Scanning for hardcoded secrets...');

    const secretPatterns = [
      { pattern: /(['"]?)(api[_-]?key|apikey)(['"]?\s*[:=]\s*['"])[a-zA-Z0-9_-]{20,}/, severity: 'critical', type: 'API Key' },
      { pattern: /(['"]?)(password|passwd|pwd)(['"]?\s*[:=]\s*['"])[^'"]{8,}/, severity: 'critical', type: 'Password' },
      { pattern: /(['"]?)(secret|token)(['"]?\s*[:=]\s*['"])[a-zA-Z0-9_-]{20,}/, severity: 'critical', type: 'Secret Token' },
      { pattern: /sk-[a-zA-Z0-9]{48}/, severity: 'critical', type: 'OpenAI API Key' },
      { pattern: /AKIA[0-9A-Z]{16}/, severity: 'critical', type: 'AWS Access Key' },
      { pattern: /ghp_[a-zA-Z0-9]{36}/, severity: 'critical', type: 'GitHub Token' },
      { pattern: /Bearer\s+[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+/, severity: 'high', type: 'JWT Token' },
    ];

    const excludeDirs = ['node_modules', '.git', 'dist', 'build', 'coverage'];
    const files = await this.findFiles(this.targetPath, ['.js', '.ts', '.jsx', '.tsx', '.json', '.env'], excludeDirs);

    for (const file of files) {
      // Skip example and test files
      if (file.includes('test') || file.includes('example') || file.includes('mock')) {
        continue;
      }

      const content = await fs.readFile(file, 'utf-8');
      const lines = content.split('\n');

      for (let i = 0; i < lines.length; i++) {
        for (const { pattern, severity, type } of secretPatterns) {
          if (pattern.test(lines[i])) {
            this.addFinding(severity, {
              type: `Hardcoded ${type}`,
              file: path.relative(this.targetPath, file),
              line: i + 1,
              snippet: lines[i].trim().substring(0, 80),
              recommendation: `Move ${type} to environment variables`,
            });
          }
        }
      }
    }

    console.log(`   Found ${this.countFindings(['critical', 'high'])} potential secrets`);
  }

  /**
   * HIGH: Check npm dependencies for vulnerabilities
   */
  async checkDependencyVulnerabilities() {
    if (!this.options.checkDependencies) return;

    console.log('[2/8] Auditing dependencies...');

    const packageJsonPath = path.join(this.targetPath, 'package.json');
    if (!await this.fileExists(packageJsonPath)) {
      console.log('   No package.json found - skipping');
      return;
    }

    try {
      const result = execSync('npm audit --json', { cwd: this.targetPath, encoding: 'utf-8' });
      const auditData = JSON.parse(result);

      const vulns = auditData.metadata?.vulnerabilities || {};
      const critical = vulns.critical || 0;
      const high = vulns.high || 0;
      const medium = vulns.medium || 0;
      const low = vulns.low || 0;

      if (critical > 0) {
        this.addFinding('critical', {
          type: 'Critical Dependency Vulnerabilities',
          count: critical,
          recommendation: 'Run `npm audit fix` immediately and review breaking changes',
        });
      }

      if (high > 0) {
        this.addFinding('high', {
          type: 'High Severity Dependency Vulnerabilities',
          count: high,
          recommendation: 'Run `npm audit fix` and update dependencies',
        });
      }

      console.log(`   Critical: ${critical}, High: ${high}, Medium: ${medium}, Low: ${low}`);
    } catch (error) {
      // npm audit returns non-zero on vulnerabilities
      if (error.stdout) {
        try {
          const auditData = JSON.parse(error.stdout);
          const vulns = auditData.metadata?.vulnerabilities || {};
          const critical = vulns.critical || 0;
          const high = vulns.high || 0;

          if (critical > 0) {
            this.addFinding('critical', {
              type: 'Critical Dependency Vulnerabilities',
              count: critical,
              recommendation: 'Run `npm audit fix` immediately',
            });
          }

          if (high > 0) {
            this.addFinding('high', {
              type: 'High Severity Dependency Vulnerabilities',
              count: high,
              recommendation: 'Run `npm audit fix` and update dependencies',
            });
          }
        } catch (parseError) {
          console.error('   Error parsing audit results');
        }
      }
    }
  }

  /**
   * MEDIUM: Check for security headers
   */
  async checkSecurityHeaders() {
    if (!this.options.checkHeaders) return;

    console.log('[3/8] Checking security headers...');

    const files = await this.findFiles(this.targetPath, ['.js', '.ts'], ['node_modules', '.git', 'dist', 'build']);
    const requiredHeaders = [
      'X-Content-Type-Options',
      'X-Frame-Options',
      'X-XSS-Protection',
      'Strict-Transport-Security',
      'Content-Security-Policy',
    ];

    let foundHeaders = new Set();

    for (const file of files) {
      const content = await fs.readFile(file, 'utf-8');

      for (const header of requiredHeaders) {
        if (content.includes(header)) {
          foundHeaders.add(header);
        }
      }
    }

    const missingHeaders = requiredHeaders.filter(h => !foundHeaders.has(h));

    if (missingHeaders.length > 0) {
      this.addFinding('medium', {
        type: 'Missing Security Headers',
        missing: missingHeaders,
        recommendation: 'Implement security headers using helmet.js or equivalent middleware',
      });
    }

    console.log(`   ${foundHeaders.size}/${requiredHeaders.length} security headers configured`);
  }

  /**
   * HIGH: Check authentication patterns
   */
  async checkAuthenticationPatterns() {
    if (!this.options.checkAuth) return;

    console.log('[4/8] Analyzing authentication patterns...');

    const files = await this.findFiles(this.targetPath, ['.js', '.ts'], ['node_modules', '.git', 'dist', 'build']);
    let hasAuth = false;
    let hasJWT = false;
    let hasSession = false;
    let issues = [];

    for (const file of files) {
      const content = await fs.readFile(file, 'utf-8');

      // Check for authentication
      if (content.includes('passport') || content.includes('authenticate') || content.includes('auth')) {
        hasAuth = true;
      }

      // Check for JWT
      if (content.includes('jsonwebtoken') || content.includes('jwt')) {
        hasJWT = true;

        // Check for weak JWT secrets
        if (/jwt.*sign.*secret.*['"][^'"]{1,15}['"]/.test(content)) {
          issues.push('Potentially weak JWT secret detected (< 16 characters)');
        }
      }

      // Check for session management
      if (content.includes('express-session') || content.includes('cookie-session')) {
        hasSession = true;

        // Check for insecure session config
        if (content.includes('secure: false') || !content.includes('secure: true')) {
          issues.push('Session cookies not configured as secure-only');
        }

        if (content.includes('httpOnly: false') || !content.includes('httpOnly: true')) {
          issues.push('Session cookies not configured as httpOnly');
        }
      }

      // Check for password hashing
      if (content.includes('password') && !content.includes('bcrypt') && !content.includes('scrypt') && !content.includes('argon2')) {
        const lines = content.split('\n');
        for (let i = 0; i < lines.length; i++) {
          if (lines[i].includes('password') && lines[i].includes('===') || lines[i].includes('==')) {
            issues.push(`Potential plaintext password comparison at ${path.relative(this.targetPath, file)}:${i + 1}`);
          }
        }
      }
    }

    if (hasAuth && issues.length > 0) {
      this.addFinding('high', {
        type: 'Authentication Security Issues',
        issues,
        recommendation: 'Review authentication implementation for security best practices',
      });
    }

    console.log(`   Auth: ${hasAuth ? 'Yes' : 'No'}, JWT: ${hasJWT ? 'Yes' : 'No'}, Issues: ${issues.length}`);
  }

  /**
   * HIGH: Check for SQL injection risks
   */
  async checkSQLInjectionRisks() {
    console.log('[5/8] Checking SQL injection risks...');

    const files = await this.findFiles(this.targetPath, ['.js', '.ts'], ['node_modules', '.git', 'dist', 'build']);
    const sqlPatterns = [
      /execute.*\+.*req\./,
      /query.*\+.*params/,
      /\$\{.*req\..*\}.*SELECT/i,
      /\$\{.*req\..*\}.*INSERT/i,
      /\$\{.*req\..*\}.*UPDATE/i,
      /\$\{.*req\..*\}.*DELETE/i,
    ];

    for (const file of files) {
      const content = await fs.readFile(file, 'utf-8');
      const lines = content.split('\n');

      for (let i = 0; i < lines.length; i++) {
        for (const pattern of sqlPatterns) {
          if (pattern.test(lines[i])) {
            this.addFinding('high', {
              type: 'Potential SQL Injection',
              file: path.relative(this.targetPath, file),
              line: i + 1,
              snippet: lines[i].trim().substring(0, 80),
              recommendation: 'Use parameterized queries or ORM instead of string concatenation',
            });
          }
        }
      }
    }

    console.log(`   Found ${this.findingsByType('Potential SQL Injection').length} potential SQL injection risks`);
  }

  /**
   * MEDIUM: Check for XSS vulnerabilities
   */
  async checkXSSVulnerabilities() {
    console.log('[6/8] Checking XSS vulnerabilities...');

    const files = await this.findFiles(this.targetPath, ['.js', '.ts', '.jsx', '.tsx'], ['node_modules', '.git', 'dist', 'build']);
    const xssPatterns = [
      /dangerouslySetInnerHTML/,
      /innerHTML.*=.*req\./,
      /\.html\(.*req\./,
    ];

    for (const file of files) {
      const content = await fs.readFile(file, 'utf-8');
      const lines = content.split('\n');

      for (let i = 0; i < lines.length; i++) {
        for (const pattern of xssPatterns) {
          if (pattern.test(lines[i])) {
            this.addFinding('medium', {
              type: 'Potential XSS Vulnerability',
              file: path.relative(this.targetPath, file),
              line: i + 1,
              snippet: lines[i].trim().substring(0, 80),
              recommendation: 'Sanitize user input before rendering HTML',
            });
          }
        }
      }
    }

    console.log(`   Found ${this.findingsByType('Potential XSS Vulnerability').length} potential XSS risks`);
  }

  /**
   * MEDIUM: Check CSRF protection
   */
  async checkCSRFProtection() {
    console.log('[7/8] Checking CSRF protection...');

    const files = await this.findFiles(this.targetPath, ['.js', '.ts'], ['node_modules', '.git', 'dist', 'build']);
    let hasCSRF = false;

    for (const file of files) {
      const content = await fs.readFile(file, 'utf-8');

      if (content.includes('csurf') || content.includes('csrf') || content.includes('csrfToken')) {
        hasCSRF = true;
        break;
      }
    }

    if (!hasCSRF) {
      this.addFinding('medium', {
        type: 'Missing CSRF Protection',
        recommendation: 'Implement CSRF protection for state-changing operations using csurf middleware',
      });
    }

    console.log(`   CSRF protection: ${hasCSRF ? 'Enabled' : 'Not detected'}`);
  }

  /**
   * MEDIUM: Check file upload security
   */
  async checkFileUploadSecurity() {
    console.log('[8/8] Checking file upload security...');

    const files = await this.findFiles(this.targetPath, ['.js', '.ts'], ['node_modules', '.git', 'dist', 'build']);
    let hasFileUpload = false;
    let issues = [];

    for (const file of files) {
      const content = await fs.readFile(file, 'utf-8');

      if (content.includes('multer') || content.includes('formidable') || content.includes('upload')) {
        hasFileUpload = true;

        // Check for file type validation
        if (!content.includes('fileFilter') && !content.includes('mimetype')) {
          issues.push('File type validation not detected');
        }

        // Check for file size limits
        if (!content.includes('limits') && !content.includes('maxFileSize')) {
          issues.push('File size limits not configured');
        }

        // Check for file name sanitization
        if (!content.includes('sanitize') && !content.includes('normalize')) {
          issues.push('File name sanitization not detected');
        }
      }
    }

    if (hasFileUpload && issues.length > 0) {
      this.addFinding('medium', {
        type: 'File Upload Security Issues',
        issues,
        recommendation: 'Implement file type validation, size limits, and filename sanitization',
      });
    }

    console.log(`   File upload: ${hasFileUpload ? 'Enabled' : 'Not detected'}, Issues: ${issues.length}`);
  }

  /**
   * Helper: Add finding
   */
  addFinding(severity, details) {
    this.findings[severity].push(details);
  }

  /**
   * Helper: Count findings by severity
   */
  countFindings(severities) {
    return severities.reduce((sum, severity) => sum + this.findings[severity].length, 0);
  }

  /**
   * Helper: Get findings by type
   */
  findingsByType(type) {
    const all = [...this.findings.critical, ...this.findings.high, ...this.findings.medium, ...this.findings.low];
    return all.filter(f => f.type === type);
  }

  /**
   * Helper: Find files by extension
   */
  async findFiles(dir, extensions, excludeDirs = []) {
    const files = [];

    async function scan(currentDir) {
      const entries = await fs.readdir(currentDir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(currentDir, entry.name);

        if (entry.isDirectory()) {
          if (!excludeDirs.includes(entry.name)) {
            await scan(fullPath);
          }
        } else if (entry.isFile()) {
          const ext = path.extname(entry.name);
          if (extensions.includes(ext)) {
            files.push(fullPath);
          }
        }
      }
    }

    await scan(dir);
    return files;
  }

  /**
   * Helper: Check file exists
   */
  async fileExists(filePath) {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Generate security report
   */
  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        critical: this.findings.critical.length,
        high: this.findings.high.length,
        medium: this.findings.medium.length,
        low: this.findings.low.length,
        info: this.findings.info.length,
        total: this.countFindings(['critical', 'high', 'medium', 'low', 'info']),
      },
      findings: this.findings,
      passed: this.findings.critical.length === 0 && this.findings.high.length === 0,
    };

    console.log('\n========================================');
    console.log('Security Audit Report');
    console.log('========================================\n');
    console.log(`Critical: ${report.summary.critical}`);
    console.log(`High:     ${report.summary.high}`);
    console.log(`Medium:   ${report.summary.medium}`);
    console.log(`Low:      ${report.summary.low}`);
    console.log(`Total:    ${report.summary.total}`);
    console.log('\n========================================');
    console.log(report.passed ? '✅ SECURITY CHECK PASSED' : '❌ SECURITY ISSUES FOUND');
    console.log('========================================\n');

    return report;
  }

  /**
   * Save report to file
   */
  async saveReport(report, outputPath) {
    await fs.writeFile(outputPath, JSON.stringify(report, null, 2));
    console.log(`Report saved to: ${outputPath}`);
  }
}

// CLI entry point
if (require.main === module) {
  const targetPath = process.argv[2] || process.cwd();
  const options = {
    deep: process.argv.includes('--deep'),
  };

  const auditor = new SecurityAuditor(targetPath, options);

  auditor.audit().then(report => {
    const outputPath = path.join(process.cwd(), `security-audit-${Date.now()}.json`);
    auditor.saveReport(report, outputPath);
    process.exit(report.passed ? 0 : 1);
  }).catch(error => {
    console.error('Audit failed:', error);
    process.exit(1);
  });
}

module.exports = SecurityAuditor;
