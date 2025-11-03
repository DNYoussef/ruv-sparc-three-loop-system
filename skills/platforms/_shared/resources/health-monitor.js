#!/usr/bin/env node
/**
 * Health Monitor for Flow Nexus Platform
 * Monitors service health, collects metrics, and triggers alerts
 */

const axios = require('axios');
const { EventEmitter } = require('events');
const fs = require('fs').promises;
const path = require('path');

/**
 * Health status enumeration
 */
const HealthStatus = {
  HEALTHY: 'healthy',
  DEGRADED: 'degraded',
  UNHEALTHY: 'unhealthy',
  UNKNOWN: 'unknown'
};

/**
 * Alert severity levels
 */
const AlertSeverity = {
  INFO: 'info',
  WARNING: 'warning',
  CRITICAL: 'critical'
};

/**
 * Health check configuration
 */
class HealthCheck {
  constructor(config) {
    this.name = config.name;
    this.type = config.type || 'http'; // http, tcp, command
    this.endpoint = config.endpoint;
    this.interval = config.interval || 30000; // 30 seconds
    this.timeout = config.timeout || 5000; // 5 seconds
    this.retries = config.retries || 3;
    this.threshold = config.threshold || 2; // failures before unhealthy

    this.consecutiveFailures = 0;
    this.lastCheck = null;
    this.lastStatus = HealthStatus.UNKNOWN;
    this.metrics = [];
  }

  async execute() {
    const startTime = Date.now();
    let success = false;
    let error = null;

    try {
      switch (this.type) {
        case 'http':
          success = await this.httpCheck();
          break;
        case 'tcp':
          success = await this.tcpCheck();
          break;
        case 'command':
          success = await this.commandCheck();
          break;
        default:
          throw new Error(`Unknown check type: ${this.type}`);
      }
    } catch (err) {
      error = err.message;
      success = false;
    }

    const responseTime = Date.now() - startTime;

    // Update metrics
    this.lastCheck = new Date();
    this.metrics.push({
      timestamp: this.lastCheck,
      success,
      responseTime,
      error
    });

    // Keep only last 100 metrics
    if (this.metrics.length > 100) {
      this.metrics = this.metrics.slice(-100);
    }

    // Update status
    if (success) {
      this.consecutiveFailures = 0;
      this.lastStatus = HealthStatus.HEALTHY;
    } else {
      this.consecutiveFailures++;
      if (this.consecutiveFailures >= this.threshold) {
        this.lastStatus = HealthStatus.UNHEALTHY;
      } else {
        this.lastStatus = HealthStatus.DEGRADED;
      }
    }

    return {
      name: this.name,
      status: this.lastStatus,
      success,
      responseTime,
      error,
      consecutiveFailures: this.consecutiveFailures
    };
  }

  async httpCheck() {
    try {
      const response = await axios.get(this.endpoint, {
        timeout: this.timeout,
        validateStatus: (status) => status >= 200 && status < 500
      });
      return response.status >= 200 && response.status < 300;
    } catch (err) {
      return false;
    }
  }

  async tcpCheck() {
    const net = require('net');
    return new Promise((resolve) => {
      const [host, port] = this.endpoint.split(':');
      const socket = new net.Socket();

      socket.setTimeout(this.timeout);
      socket.on('connect', () => {
        socket.destroy();
        resolve(true);
      });
      socket.on('timeout', () => {
        socket.destroy();
        resolve(false);
      });
      socket.on('error', () => {
        resolve(false);
      });

      socket.connect(parseInt(port), host);
    });
  }

  async commandCheck() {
    const { exec } = require('child_process');
    return new Promise((resolve) => {
      exec(this.endpoint, { timeout: this.timeout }, (error) => {
        resolve(!error);
      });
    });
  }

  getMetricsSummary() {
    if (this.metrics.length === 0) {
      return null;
    }

    const successCount = this.metrics.filter(m => m.success).length;
    const avgResponseTime = this.metrics.reduce((sum, m) => sum + m.responseTime, 0) / this.metrics.length;

    return {
      name: this.name,
      checks: this.metrics.length,
      successRate: (successCount / this.metrics.length * 100).toFixed(2) + '%',
      avgResponseTime: avgResponseTime.toFixed(2) + 'ms',
      lastCheck: this.lastCheck,
      currentStatus: this.lastStatus
    };
  }
}

/**
 * Health Monitor
 */
class HealthMonitor extends EventEmitter {
  constructor(configPath = 'platform/config/health-checks.json') {
    super();
    this.configPath = configPath;
    this.checks = new Map();
    this.intervals = new Map();
    this.alerts = [];
    this.running = false;
  }

  async loadConfig() {
    try {
      const configData = await fs.readFile(this.configPath, 'utf8');
      const config = JSON.parse(configData);

      for (const checkConfig of config.checks || []) {
        const check = new HealthCheck(checkConfig);
        this.checks.set(check.name, check);
      }

      console.log(`Loaded ${this.checks.size} health checks`);
    } catch (err) {
      console.warn('Config file not found, creating default:', this.configPath);
      await this.createDefaultConfig();
      await this.loadConfig();
    }
  }

  async createDefaultConfig() {
    const defaultConfig = {
      version: '1.0.0',
      checks: [
        {
          name: 'api',
          type: 'http',
          endpoint: 'http://localhost:3000/health',
          interval: 30000,
          timeout: 5000,
          threshold: 2
        },
        {
          name: 'database',
          type: 'tcp',
          endpoint: 'localhost:5432',
          interval: 60000,
          timeout: 3000
        },
        {
          name: 'redis',
          type: 'command',
          endpoint: 'redis-cli ping',
          interval: 60000,
          timeout: 3000
        }
      ]
    };

    await fs.mkdir(path.dirname(this.configPath), { recursive: true });
    await fs.writeFile(this.configPath, JSON.stringify(defaultConfig, null, 2));
  }

  async start() {
    if (this.running) {
      console.log('Health monitor already running');
      return;
    }

    await this.loadConfig();
    this.running = true;

    // Start health checks
    for (const [name, check] of this.checks) {
      this.startCheck(name, check);
    }

    console.log('Health monitor started');
    this.emit('started');
  }

  stop() {
    if (!this.running) {
      return;
    }

    // Stop all intervals
    for (const interval of this.intervals.values()) {
      clearInterval(interval);
    }

    this.intervals.clear();
    this.running = false;

    console.log('Health monitor stopped');
    this.emit('stopped');
  }

  startCheck(name, check) {
    // Run immediately
    this.runCheck(name, check);

    // Schedule recurring checks
    const interval = setInterval(() => {
      this.runCheck(name, check);
    }, check.interval);

    this.intervals.set(name, interval);
  }

  async runCheck(name, check) {
    try {
      const result = await check.execute();
      this.emit('check', result);

      // Generate alerts
      if (result.status === HealthStatus.UNHEALTHY) {
        this.alert(AlertSeverity.CRITICAL, `Service unhealthy: ${name}`, result);
      } else if (result.status === HealthStatus.DEGRADED) {
        this.alert(AlertSeverity.WARNING, `Service degraded: ${name}`, result);
      }
    } catch (err) {
      console.error(`Error running check ${name}:`, err.message);
      this.alert(AlertSeverity.CRITICAL, `Check failed: ${name}`, { error: err.message });
    }
  }

  alert(severity, message, data) {
    const alert = {
      severity,
      message,
      data,
      timestamp: new Date()
    };

    this.alerts.push(alert);

    // Keep only last 1000 alerts
    if (this.alerts.length > 1000) {
      this.alerts = this.alerts.slice(-1000);
    }

    this.emit('alert', alert);

    // Log to console based on severity
    const logLevel = severity === AlertSeverity.CRITICAL ? 'error' :
                     severity === AlertSeverity.WARNING ? 'warn' : 'info';
    console[logLevel](`[${severity.toUpperCase()}] ${message}`);
  }

  getStatus() {
    const status = {
      running: this.running,
      checks: {},
      overall: HealthStatus.HEALTHY
    };

    let hasUnhealthy = false;
    let hasDegraded = false;

    for (const [name, check] of this.checks) {
      const summary = check.getMetricsSummary();
      status.checks[name] = summary;

      if (check.lastStatus === HealthStatus.UNHEALTHY) {
        hasUnhealthy = true;
      } else if (check.lastStatus === HealthStatus.DEGRADED) {
        hasDegraded = true;
      }
    }

    // Determine overall status
    if (hasUnhealthy) {
      status.overall = HealthStatus.UNHEALTHY;
    } else if (hasDegraded) {
      status.overall = HealthStatus.DEGRADED;
    }

    return status;
  }

  getMetrics() {
    const metrics = {};

    for (const [name, check] of this.checks) {
      metrics[name] = check.metrics;
    }

    return metrics;
  }

  getAlerts(severity = null, limit = 100) {
    let alerts = this.alerts;

    if (severity) {
      alerts = alerts.filter(a => a.severity === severity);
    }

    return alerts.slice(-limit);
  }
}

// CLI interface
async function main() {
  const command = process.argv[2] || 'start';
  const monitor = new HealthMonitor();

  // Event handlers
  monitor.on('check', (result) => {
    if (!result.success) {
      console.log(`[${result.status}] ${result.name}: ${result.error || 'Check failed'}`);
    }
  });

  monitor.on('alert', (alert) => {
    // Alerts are already logged in the alert() method
  });

  switch (command) {
    case 'start':
      await monitor.start();

      // Keep running
      process.on('SIGINT', () => {
        console.log('\nShutting down...');
        monitor.stop();
        process.exit(0);
      });
      break;

    case 'status':
      await monitor.loadConfig();

      // Run all checks once
      for (const [name, check] of monitor.checks) {
        await check.execute();
      }

      const status = monitor.getStatus();
      console.log(JSON.stringify(status, null, 2));
      break;

    case 'metrics':
      await monitor.loadConfig();

      // Run all checks once
      for (const [name, check] of monitor.checks) {
        await check.execute();
      }

      const metrics = monitor.getMetrics();
      console.log(JSON.stringify(metrics, null, 2));
      break;

    default:
      console.log('Usage: health-monitor.js <command>');
      console.log('\nCommands:');
      console.log('  start   - Start health monitoring (default)');
      console.log('  status  - Get current health status');
      console.log('  metrics - Get detailed metrics');
      process.exit(1);
  }
}

// Export for library use
module.exports = { HealthMonitor, HealthCheck, HealthStatus, AlertSeverity };

// Run CLI if executed directly
if (require.main === module) {
  main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
}
