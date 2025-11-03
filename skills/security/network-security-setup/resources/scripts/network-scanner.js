#!/usr/bin/env node
/**
 * Network Scanner for Security Setup Validation
 * Tests connectivity and validates network isolation policies
 */

const dns = require('dns').promises;
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);
const fs = require('fs').promises;
const path = require('path');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

class NetworkScanner {
  constructor(options = {}) {
    this.configFile = options.configFile || '/etc/network-security/trusted-domains.conf';
    this.timeout = options.timeout || 5000;
    this.parallel = options.parallel || 5;
    this.verbose = options.verbose || false;
    this.results = {
      trusted: [],
      blocked: [],
      errors: [],
      summary: {},
    };
  }

  /**
   * Log message with timestamp and color
   */
  log(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const levelColors = {
      INFO: colors.blue,
      SUCCESS: colors.green,
      ERROR: colors.red,
      WARN: colors.yellow,
    };
    const color = levelColors[level] || colors.reset;

    console.log(`${color}[${timestamp}] [${level}]${colors.reset} ${message}`);

    if (data && this.verbose) {
      console.log(JSON.stringify(data, null, 2));
    }
  }

  /**
   * Parse trusted domains from config file
   */
  async parseTrustedDomains() {
    this.log('INFO', `Parsing trusted domains from ${this.configFile}`);

    try {
      const content = await fs.readFile(this.configFile, 'utf-8');

      // Extract domains, removing comments and empty lines
      const domains = content
        .split('\n')
        .map(line => line.split('#')[0].trim())
        .filter(line => line.length > 0);

      this.log('INFO', `Found ${domains.length} trusted domains`);
      return domains;
    } catch (error) {
      this.log('ERROR', `Failed to read config file: ${error.message}`);
      throw error;
    }
  }

  /**
   * Resolve domain to IP addresses
   */
  async resolveDomain(domain) {
    // Remove wildcard prefix
    const cleanDomain = domain.replace(/^\*\./, '');

    try {
      const [ipv4Addresses, ipv6Addresses] = await Promise.allSettled([
        dns.resolve4(cleanDomain),
        dns.resolve6(cleanDomain),
      ]);

      const addresses = [];

      if (ipv4Addresses.status === 'fulfilled') {
        addresses.push(...ipv4Addresses.value);
      }

      if (ipv6Addresses.status === 'fulfilled') {
        addresses.push(...ipv6Addresses.value);
      }

      return addresses;
    } catch (error) {
      this.log('WARN', `Failed to resolve ${domain}: ${error.message}`);
      return [];
    }
  }

  /**
   * Test HTTP/HTTPS connectivity to domain
   */
  async testHttpConnectivity(domain, protocol = 'https') {
    const cleanDomain = domain.replace(/^\*\./, '');
    const url = `${protocol}://${cleanDomain}`;

    try {
      const { stdout, stderr } = await execPromise(
        `curl -s -o /dev/null -w "%{http_code}" --max-time ${this.timeout / 1000} "${url}"`,
        { timeout: this.timeout }
      );

      const statusCode = parseInt(stdout.trim(), 10);
      const success = statusCode >= 200 && statusCode < 400;

      return {
        success,
        statusCode,
        protocol,
        url,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        protocol,
        url,
      };
    }
  }

  /**
   * Test TCP connectivity to domain and port
   */
  async testTcpConnectivity(domain, port = 443) {
    const cleanDomain = domain.replace(/^\*\./, '');

    try {
      const { stdout, stderr } = await execPromise(
        `nc -z -w ${this.timeout / 1000} ${cleanDomain} ${port}`,
        { timeout: this.timeout }
      );

      return { success: true, port };
    } catch (error) {
      return { success: false, port, error: error.message };
    }
  }

  /**
   * Test ICMP ping connectivity
   */
  async testPingConnectivity(domain) {
    const cleanDomain = domain.replace(/^\*\./, '');

    try {
      const { stdout, stderr } = await execPromise(
        `ping -c 1 -W ${this.timeout / 1000} ${cleanDomain}`,
        { timeout: this.timeout }
      );

      // Parse ping statistics
      const match = stdout.match(/(\d+)% packet loss/);
      const packetLoss = match ? parseInt(match[1], 10) : 100;

      return {
        success: packetLoss === 0,
        packetLoss,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  /**
   * Comprehensive connectivity test for a domain
   */
  async testDomain(domain) {
    this.log('INFO', `Testing domain: ${domain}`);

    const result = {
      domain,
      timestamp: new Date().toISOString(),
      dns: { addresses: [] },
      http: {},
      https: {},
      tcp: {},
      ping: {},
      overall: false,
    };

    try {
      // DNS resolution
      result.dns.addresses = await this.resolveDomain(domain);
      result.dns.success = result.dns.addresses.length > 0;

      if (!result.dns.success) {
        this.log('WARN', `DNS resolution failed for ${domain}`);
        return result;
      }

      // HTTP connectivity
      result.http = await this.testHttpConnectivity(domain, 'http');

      // HTTPS connectivity
      result.https = await this.testHttpConnectivity(domain, 'https');

      // TCP connectivity (port 443)
      result.tcp = await this.testTcpConnectivity(domain, 443);

      // Ping connectivity
      result.ping = await this.testPingConnectivity(domain);

      // Overall success if any test passes
      result.overall = (
        result.http.success ||
        result.https.success ||
        result.tcp.success ||
        result.ping.success
      );

      if (result.overall) {
        this.log('SUCCESS', `✓ ${domain}: Accessible`);
        this.results.trusted.push(result);
      } else {
        this.log('ERROR', `✗ ${domain}: Not accessible`);
        this.results.blocked.push(result);
      }
    } catch (error) {
      this.log('ERROR', `Failed to test ${domain}: ${error.message}`);
      result.error = error.message;
      this.results.errors.push(result);
    }

    return result;
  }

  /**
   * Test blocked domains (should fail)
   */
  async testBlockedDomains() {
    this.log('INFO', 'Testing blocked domains (should fail)...');

    const blockedDomains = [
      'malicious-site.com',
      'untrusted-registry.net',
      'random-website.org',
    ];

    const results = [];

    for (const domain of blockedDomains) {
      const result = await this.testDomain(domain);

      if (!result.overall) {
        this.log('SUCCESS', `✓ ${domain}: Correctly blocked`);
      } else {
        this.log('WARN', `✗ ${domain}: NOT blocked (potential security issue)`);
      }

      results.push(result);
    }

    return results;
  }

  /**
   * Run comprehensive network scan
   */
  async scan() {
    this.log('INFO', '========================================');
    this.log('INFO', 'Network Security Scanner');
    this.log('INFO', '========================================');

    try {
      // Parse trusted domains
      const trustedDomains = await this.parseTrustedDomains();

      // Test trusted domains in parallel batches
      this.log('INFO', `Testing ${trustedDomains.length} trusted domains...`);

      for (let i = 0; i < trustedDomains.length; i += this.parallel) {
        const batch = trustedDomains.slice(i, i + this.parallel);
        await Promise.all(batch.map(domain => this.testDomain(domain)));
      }

      // Test blocked domains
      const blockedResults = await this.testBlockedDomains();

      // Generate summary
      this.results.summary = {
        total_tested: trustedDomains.length,
        accessible: this.results.trusted.length,
        blocked: this.results.blocked.length,
        errors: this.results.errors.length,
        success_rate: (
          (this.results.trusted.length / trustedDomains.length) * 100
        ).toFixed(2),
        blocked_domains_tested: blockedResults.length,
        blocked_domains_correct: blockedResults.filter(r => !r.overall).length,
      };

      // Display summary
      this.log('INFO', '');
      this.log('INFO', '========================================');
      this.log('INFO', 'Scan Summary');
      this.log('INFO', '========================================');
      this.log('INFO', `Total Domains Tested: ${this.results.summary.total_tested}`);
      this.log('SUCCESS', `Accessible: ${this.results.summary.accessible}`);
      this.log('ERROR', `Blocked: ${this.results.summary.blocked}`);
      this.log('WARN', `Errors: ${this.results.summary.errors}`);
      this.log('INFO', `Success Rate: ${this.results.summary.success_rate}%`);
      this.log('INFO', '');
      this.log('INFO', `Blocked Domains Tested: ${this.results.summary.blocked_domains_tested}`);
      this.log('SUCCESS', `Correctly Blocked: ${this.results.summary.blocked_domains_correct}`);
      this.log('INFO', '========================================');

      return this.results;
    } catch (error) {
      this.log('ERROR', `Scan failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Export results to JSON file
   */
  async exportResults(outputPath) {
    this.log('INFO', `Exporting results to ${outputPath}`);

    try {
      await fs.writeFile(
        outputPath,
        JSON.stringify(this.results, null, 2),
        'utf-8'
      );
      this.log('SUCCESS', `Results exported to ${outputPath}`);
    } catch (error) {
      this.log('ERROR', `Failed to export results: ${error.message}`);
      throw error;
    }
  }
}

// CLI interface
async function main() {
  const args = process.argv.slice(2);

  const options = {
    configFile: process.env.CONFIG_FILE || '/etc/network-security/trusted-domains.conf',
    timeout: parseInt(process.env.TIMEOUT || '5000', 10),
    parallel: parseInt(process.env.PARALLEL || '5', 10),
    verbose: args.includes('--verbose'),
  };

  const scanner = new NetworkScanner(options);

  try {
    const results = await scanner.scan();

    // Export results if output path specified
    const outputIndex = args.indexOf('--output');
    if (outputIndex !== -1 && args[outputIndex + 1]) {
      await scanner.exportResults(args[outputIndex + 1]);
    }

    // Exit with error code if any tests failed
    const exitCode = results.summary.blocked > 0 ? 1 : 0;
    process.exit(exitCode);
  } catch (error) {
    console.error(`Fatal error: ${error.message}`);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}

module.exports = NetworkScanner;
