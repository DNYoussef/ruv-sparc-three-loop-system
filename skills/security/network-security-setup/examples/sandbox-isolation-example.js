#!/usr/bin/env node
/**
 * Example: Complete Sandbox Network Isolation Setup
 *
 * This comprehensive example demonstrates:
 * - Network isolation configuration
 * - Firewall rules setup
 * - Domain whitelist management
 * - Security policy enforcement
 * - Monitoring and logging
 *
 * Use Case: Configuring secure network isolation for Claude Code sandbox
 */

const fs = require('fs').promises;
const path = require('path');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

// Configuration
const CONFIG = {
  isolation: {
    mode: 'trusted', // Options: 'none', 'trusted', 'custom'
    defaultAction: 'deny',
    enabledProtocols: ['https', 'ssh'],
  },
  paths: {
    configDir: '/etc/network-security',
    trustedDomainsFile: '/etc/network-security/trusted-domains.conf',
    firewallRulesFile: '/etc/network-security/firewall-rules.json',
    logFile: '/var/log/network-security/isolation.log',
  },
  monitoring: {
    enabled: true,
    logLevel: 'info',
    alertOnBlocked: true,
    metricsInterval: 60000, // 1 minute
  },
};

/**
 * Logger utility
 */
class Logger {
  constructor(logFile) {
    this.logFile = logFile;
  }

  async log(level, message, metadata = {}) {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level,
      message,
      ...metadata,
    };

    const logLine = `[${timestamp}] [${level}] ${message}\n`;
    console.log(logLine.trim());

    // Append to log file
    try {
      await fs.appendFile(this.logFile, logLine);
    } catch (error) {
      console.error(`Failed to write to log file: ${error.message}`);
    }
  }

  info(message, metadata) { return this.log('INFO', message, metadata); }
  warn(message, metadata) { return this.log('WARN', message, metadata); }
  error(message, metadata) { return this.log('ERROR', message, metadata); }
  success(message, metadata) { return this.log('SUCCESS', message, metadata); }
}

/**
 * Sandbox Isolation Manager
 */
class SandboxIsolationManager {
  constructor(config) {
    this.config = config;
    this.logger = new Logger(config.paths.logFile);
    this.metrics = {
      allowedConnections: 0,
      blockedConnections: 0,
      totalAttempts: 0,
    };
  }

  /**
   * Initialize sandbox isolation
   */
  async initialize() {
    await this.logger.info('Initializing sandbox network isolation...');

    try {
      // Create directories
      await this.createDirectories();

      // Configure trusted domains
      await this.configureTrustedDomains();

      // Setup firewall rules
      await this.setupFirewallRules();

      // Initialize monitoring
      if (this.config.monitoring.enabled) {
        await this.initializeMonitoring();
      }

      await this.logger.success('Sandbox isolation initialized successfully');
    } catch (error) {
      await this.logger.error('Initialization failed', { error: error.message });
      throw error;
    }
  }

  /**
   * Create necessary directories
   */
  async createDirectories() {
    await this.logger.info('Creating configuration directories...');

    const dirs = [
      this.config.paths.configDir,
      path.join(this.config.paths.configDir, 'rules'),
      path.join(this.config.paths.configDir, 'logs'),
    ];

    for (const dir of dirs) {
      try {
        await fs.mkdir(dir, { recursive: true, mode: 0o755 });
        await this.logger.info(`Created directory: ${dir}`);
      } catch (error) {
        if (error.code !== 'EEXIST') {
          throw error;
        }
      }
    }
  }

  /**
   * Configure trusted domains whitelist
   */
  async configureTrustedDomains() {
    await this.logger.info('Configuring trusted domains...');

    const trustedDomains = [
      // Package Registries
      '*.npmjs.org',
      'registry.npmjs.org',
      '*.yarnpkg.com',
      '*.pypi.org',
      'pypi.python.org',

      // Container Registries
      '*.docker.io',
      'registry.hub.docker.com',
      'ghcr.io',

      // Source Control
      '*.github.com',
      'api.github.com',
      'raw.githubusercontent.com',

      // CDNs
      '*.cloudfront.net',
      'cdn.jsdelivr.net',
      'unpkg.com',
    ];

    const configContent = `# Trusted Domains Configuration
# Generated: ${new Date().toISOString()}
# Mode: ${this.config.isolation.mode}
#
# Syntax:
#   - One domain per line
#   - Wildcards supported: *.example.com
#   - Comments start with #

${trustedDomains.map(domain => domain).join('\n')}
`;

    await fs.writeFile(
      this.config.paths.trustedDomainsFile,
      configContent,
      'utf-8'
    );

    await this.logger.success(
      `Configured ${trustedDomains.length} trusted domains`
    );
  }

  /**
   * Setup firewall rules
   */
  async setupFirewallRules() {
    await this.logger.info('Setting up firewall rules...');

    const firewallRules = {
      version: '1.0.0',
      metadata: {
        name: 'Sandbox Network Isolation Rules',
        created: new Date().toISOString(),
        mode: this.config.isolation.mode,
      },
      default_policy: this.config.isolation.defaultAction,
      rules: [
        {
          id: 'rule-001',
          name: 'Allow Loopback',
          description: 'Allow all loopback traffic',
          enabled: true,
          priority: 100,
          action: 'allow',
          direction: 'both',
          protocol: 'any',
          source: {
            addresses: ['127.0.0.0/8', '::1/128'],
          },
          destination: {
            addresses: ['127.0.0.0/8', '::1/128'],
          },
        },
        {
          id: 'rule-002',
          name: 'Allow Established Connections',
          description: 'Allow established and related connections',
          enabled: true,
          priority: 90,
          action: 'allow',
          direction: 'both',
          protocol: 'any',
          state: ['ESTABLISHED', 'RELATED'],
        },
        {
          id: 'rule-003',
          name: 'Allow DNS',
          description: 'Allow DNS queries',
          enabled: true,
          priority: 80,
          action: 'allow',
          direction: 'outbound',
          protocol: 'udp',
          destination: {
            ports: [53],
          },
        },
        {
          id: 'rule-004',
          name: 'Allow HTTPS to Trusted Domains',
          description: 'Allow HTTPS to whitelisted domains',
          enabled: true,
          priority: 70,
          action: 'allow',
          direction: 'outbound',
          protocol: 'tcp',
          destination: {
            ports: [443],
          },
          metadata: {
            check_trusted_domains: true,
          },
        },
        {
          id: 'rule-999',
          name: 'Deny All Outbound',
          description: 'Default deny for all other outbound traffic',
          enabled: true,
          priority: 1,
          action: 'deny',
          direction: 'outbound',
          logging: {
            enabled: true,
            level: 'warning',
            prefix: 'BLOCKED',
          },
        },
      ],
    };

    await fs.writeFile(
      this.config.paths.firewallRulesFile,
      JSON.stringify(firewallRules, null, 2),
      'utf-8'
    );

    await this.logger.success(
      `Configured ${firewallRules.rules.length} firewall rules`
    );
  }

  /**
   * Initialize monitoring and metrics collection
   */
  async initializeMonitoring() {
    await this.logger.info('Initializing network monitoring...');

    // Start metrics collection interval
    this.metricsInterval = setInterval(
      () => this.collectMetrics(),
      this.config.monitoring.metricsInterval
    );

    await this.logger.success('Monitoring initialized');
  }

  /**
   * Collect network metrics
   */
  async collectMetrics() {
    const timestamp = new Date().toISOString();

    await this.logger.info('Collecting network metrics', {
      timestamp,
      metrics: this.metrics,
    });

    // Check for anomalies
    if (this.metrics.blockedConnections > 100) {
      await this.logger.warn('High number of blocked connections detected', {
        count: this.metrics.blockedConnections,
      });
    }

    // Calculate success rate
    const successRate = this.metrics.totalAttempts > 0
      ? (this.metrics.allowedConnections / this.metrics.totalAttempts) * 100
      : 0;

    await this.logger.info('Connection success rate', {
      rate: `${successRate.toFixed(2)}%`,
    });
  }

  /**
   * Test network connectivity
   */
  async testConnectivity() {
    await this.logger.info('Testing network connectivity...');

    const testDomains = [
      'registry.npmjs.org',
      'api.github.com',
      'malicious-site.com', // Should be blocked
    ];

    for (const domain of testDomains) {
      try {
        const { stdout, stderr } = await execPromise(
          `curl -s -o /dev/null -w "%{http_code}" --max-time 5 https://${domain}`,
          { timeout: 6000 }
        );

        const statusCode = parseInt(stdout.trim(), 10);

        if (statusCode >= 200 && statusCode < 400) {
          await this.logger.success(`✓ ${domain}: Accessible (${statusCode})`);
          this.metrics.allowedConnections++;
        } else {
          await this.logger.warn(`✗ ${domain}: Not accessible (${statusCode})`);
        }

        this.metrics.totalAttempts++;
      } catch (error) {
        await this.logger.error(`✗ ${domain}: Connection failed`, {
          error: error.message,
        });
        this.metrics.blockedConnections++;
        this.metrics.totalAttempts++;
      }
    }
  }

  /**
   * Validate configuration
   */
  async validateConfiguration() {
    await this.logger.info('Validating configuration...');

    const checks = [
      {
        name: 'Trusted domains file exists',
        check: async () => {
          await fs.access(this.config.paths.trustedDomainsFile);
          return true;
        },
      },
      {
        name: 'Firewall rules file exists',
        check: async () => {
          await fs.access(this.config.paths.firewallRulesFile);
          return true;
        },
      },
      {
        name: 'Log directory writable',
        check: async () => {
          const logDir = path.dirname(this.config.paths.logFile);
          await fs.access(logDir, fs.constants.W_OK);
          return true;
        },
      },
      {
        name: 'Firewall rules valid JSON',
        check: async () => {
          const content = await fs.readFile(
            this.config.paths.firewallRulesFile,
            'utf-8'
          );
          JSON.parse(content);
          return true;
        },
      },
    ];

    let passed = 0;
    let failed = 0;

    for (const check of checks) {
      try {
        await check.check();
        await this.logger.success(`✓ ${check.name}`);
        passed++;
      } catch (error) {
        await this.logger.error(`✗ ${check.name}`, { error: error.message });
        failed++;
      }
    }

    await this.logger.info('Validation complete', {
      passed,
      failed,
      total: checks.length,
    });

    return failed === 0;
  }

  /**
   * Generate security report
   */
  async generateSecurityReport() {
    await this.logger.info('Generating security report...');

    const report = {
      timestamp: new Date().toISOString(),
      configuration: {
        mode: this.config.isolation.mode,
        defaultAction: this.config.isolation.defaultAction,
      },
      metrics: this.metrics,
      status: 'OPERATIONAL',
    };

    const reportPath = path.join(
      this.config.paths.configDir,
      `security-report-${Date.now()}.json`
    );

    await fs.writeFile(reportPath, JSON.stringify(report, null, 2), 'utf-8');

    await this.logger.success(`Security report generated: ${reportPath}`);

    return report;
  }

  /**
   * Cleanup and shutdown
   */
  async shutdown() {
    await this.logger.info('Shutting down sandbox isolation...');

    // Stop monitoring
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
    }

    // Generate final report
    await this.generateSecurityReport();

    await this.logger.success('Shutdown complete');
  }
}

/**
 * Main execution
 */
async function main() {
  console.log('=========================================');
  console.log('Sandbox Network Isolation Example');
  console.log('=========================================');

  const manager = new SandboxIsolationManager(CONFIG);

  try {
    // Initialize isolation
    await manager.initialize();

    // Validate configuration
    const isValid = await manager.validateConfiguration();

    if (!isValid) {
      throw new Error('Configuration validation failed');
    }

    // Test connectivity
    await manager.testConnectivity();

    // Run for a short duration (in real usage, this would run continuously)
    console.log('\nRunning for 10 seconds...');
    await new Promise(resolve => setTimeout(resolve, 10000));

    // Shutdown
    await manager.shutdown();

    console.log('\n=========================================');
    console.log('Example completed successfully');
    console.log('=========================================');
  } catch (error) {
    console.error(`\nFatal error: ${error.message}`);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}

module.exports = { SandboxIsolationManager };
