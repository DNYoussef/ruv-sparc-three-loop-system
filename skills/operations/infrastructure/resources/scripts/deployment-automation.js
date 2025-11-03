#!/usr/bin/env node
/**
 * Deployment Automation and CI/CD Orchestration Script
 * Purpose: Orchestrate deployment pipelines with blue-green, canary strategies
 * Version: 2.0.0
 * Last Updated: 2025-11-02
 */

const fs = require('fs').promises;
const path = require('path');
const { exec } = require('child_process');
const util = require('util');
const yaml = require('js-yaml');
const axios = require('axios');

const execPromise = util.promisify(exec);

// Configuration
const CONFIG = {
  stateDir: process.env.DEPLOYMENT_STATE_DIR || path.join(process.env.HOME, '.deployment-automation'),
  logLevel: process.env.LOG_LEVEL || 'info',
  maxRollbacks: parseInt(process.env.MAX_ROLLBACKS) || 5,
  healthCheckTimeout: parseInt(process.env.HEALTH_CHECK_TIMEOUT) || 300000, // 5 minutes
  healthCheckInterval: parseInt(process.env.HEALTH_CHECK_INTERVAL) || 5000, // 5 seconds
};

// Logger utility
class Logger {
  static levels = { debug: 0, info: 1, warn: 2, error: 3 };

  static log(level, message, data = null) {
    if (Logger.levels[level] >= Logger.levels[CONFIG.logLevel]) {
      const timestamp = new Date().toISOString();
      const logMessage = `[${timestamp}] [${level.toUpperCase()}] ${message}`;
      console.log(logMessage);
      if (data) {
        console.log(JSON.stringify(data, null, 2));
      }
    }
  }

  static debug(message, data) { Logger.log('debug', message, data); }
  static info(message, data) { Logger.log('info', message, data); }
  static warn(message, data) { Logger.log('warn', message, data); }
  static error(message, data) { Logger.log('error', message, data); }
}

// Deployment state management
class DeploymentState {
  constructor(environment, application) {
    this.environment = environment;
    this.application = application;
    this.stateFile = path.join(CONFIG.stateDir, `${environment}-${application}.json`);
  }

  async load() {
    try {
      const data = await fs.readFile(this.stateFile, 'utf8');
      return JSON.parse(data);
    } catch (error) {
      if (error.code === 'ENOENT') {
        return this.getDefaultState();
      }
      throw error;
    }
  }

  async save(state) {
    await fs.mkdir(CONFIG.stateDir, { recursive: true });
    await fs.writeFile(this.stateFile, JSON.stringify(state, null, 2));
    Logger.info(`Deployment state saved for ${this.environment}/${this.application}`);
  }

  getDefaultState() {
    return {
      environment: this.environment,
      application: this.application,
      deployments: [],
      currentVersion: null,
      previousVersions: [],
      createdAt: new Date().toISOString(),
    };
  }

  async recordDeployment(deployment) {
    const state = await this.load();

    state.deployments.push({
      ...deployment,
      timestamp: new Date().toISOString(),
    });

    // Keep only last 100 deployments
    if (state.deployments.length > 100) {
      state.deployments = state.deployments.slice(-100);
    }

    if (deployment.status === 'success') {
      if (state.currentVersion) {
        state.previousVersions.unshift(state.currentVersion);
        state.previousVersions = state.previousVersions.slice(0, CONFIG.maxRollbacks);
      }
      state.currentVersion = deployment.version;
    }

    await this.save(state);
  }
}

// Health check utilities
class HealthChecker {
  static async checkHttp(url, expectedStatus = 200) {
    try {
      const response = await axios.get(url, { timeout: 10000 });
      return response.status === expectedStatus;
    } catch (error) {
      Logger.debug(`HTTP health check failed for ${url}`, error.message);
      return false;
    }
  }

  static async checkCommand(command) {
    try {
      await execPromise(command);
      return true;
    } catch (error) {
      Logger.debug(`Command health check failed: ${command}`, error.message);
      return false;
    }
  }

  static async waitForHealthy(checks, timeout = CONFIG.healthCheckTimeout) {
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      Logger.debug('Running health checks...');

      const results = await Promise.all(
        checks.map(async (check) => {
          if (check.type === 'http') {
            return HealthChecker.checkHttp(check.url, check.expectedStatus);
          } else if (check.type === 'command') {
            return HealthChecker.checkCommand(check.command);
          }
          return false;
        })
      );

      if (results.every((result) => result === true)) {
        Logger.info('All health checks passed');
        return true;
      }

      await new Promise((resolve) => setTimeout(resolve, CONFIG.healthCheckInterval));
    }

    Logger.error('Health checks timed out');
    return false;
  }
}

// Deployment strategies
class DeploymentStrategy {
  static async blueGreen(config) {
    Logger.info('Starting Blue-Green deployment', { application: config.application });

    const state = new DeploymentState(config.environment, config.application);

    // Deploy to green environment
    Logger.info('Deploying to green environment...');
    await DeploymentStrategy.executeDeployment(config, 'green');

    // Health check green environment
    const greenHealthy = await HealthChecker.waitForHealthy(config.healthChecks.green);

    if (!greenHealthy) {
      Logger.error('Green environment health checks failed');
      await state.recordDeployment({
        version: config.version,
        strategy: 'blue-green',
        status: 'failed',
        reason: 'Health checks failed',
      });
      throw new Error('Deployment failed: Green environment unhealthy');
    }

    // Switch traffic to green
    Logger.info('Switching traffic to green environment...');
    await DeploymentStrategy.switchTraffic(config, 'blue', 'green');

    // Health check after traffic switch
    const postSwitchHealthy = await HealthChecker.waitForHealthy(config.healthChecks.green);

    if (!postSwitchHealthy) {
      Logger.error('Post-switch health checks failed, rolling back...');
      await DeploymentStrategy.switchTraffic(config, 'green', 'blue');
      await state.recordDeployment({
        version: config.version,
        strategy: 'blue-green',
        status: 'failed',
        reason: 'Post-switch health checks failed',
      });
      throw new Error('Deployment failed: Post-switch health checks failed');
    }

    // Clean up blue environment
    Logger.info('Cleaning up blue environment...');
    await DeploymentStrategy.cleanup(config, 'blue');

    await state.recordDeployment({
      version: config.version,
      strategy: 'blue-green',
      status: 'success',
    });

    Logger.info('Blue-Green deployment completed successfully');
  }

  static async canary(config) {
    Logger.info('Starting Canary deployment', { application: config.application });

    const state = new DeploymentState(config.environment, config.application);
    const stages = config.canary.stages || [10, 25, 50, 100]; // Traffic percentages

    // Deploy canary version
    Logger.info('Deploying canary version...');
    await DeploymentStrategy.executeDeployment(config, 'canary');

    for (const percentage of stages) {
      Logger.info(`Routing ${percentage}% traffic to canary...`);
      await DeploymentStrategy.setTrafficSplit(config, percentage);

      // Monitor canary
      const duration = config.canary.stageDuration || 300000; // 5 minutes
      Logger.info(`Monitoring canary for ${duration / 1000} seconds...`);

      const healthy = await DeploymentStrategy.monitorCanary(config, duration);

      if (!healthy) {
        Logger.error(`Canary unhealthy at ${percentage}% traffic, rolling back...`);
        await DeploymentStrategy.setTrafficSplit(config, 0);
        await state.recordDeployment({
          version: config.version,
          strategy: 'canary',
          status: 'failed',
          reason: `Canary unhealthy at ${percentage}% traffic`,
        });
        throw new Error('Canary deployment failed');
      }
    }

    // Promote canary to production
    Logger.info('Promoting canary to production...');
    await DeploymentStrategy.promoteCanary(config);

    await state.recordDeployment({
      version: config.version,
      strategy: 'canary',
      status: 'success',
    });

    Logger.info('Canary deployment completed successfully');
  }

  static async rollingUpdate(config) {
    Logger.info('Starting Rolling Update deployment', { application: config.application });

    const state = new DeploymentState(config.environment, config.application);
    const instances = config.rolling.instances || [];
    const batchSize = config.rolling.batchSize || 1;

    for (let i = 0; i < instances.length; i += batchSize) {
      const batch = instances.slice(i, i + batchSize);
      Logger.info(`Updating batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(instances.length / batchSize)}`);

      // Update instances in batch
      await Promise.all(
        batch.map((instance) => DeploymentStrategy.updateInstance(config, instance))
      );

      // Health check batch
      const batchHealthy = await HealthChecker.waitForHealthy(config.healthChecks.instances);

      if (!batchHealthy) {
        Logger.error(`Batch ${Math.floor(i / batchSize) + 1} health checks failed, rolling back...`);
        await state.recordDeployment({
          version: config.version,
          strategy: 'rolling-update',
          status: 'failed',
          reason: `Batch ${Math.floor(i / batchSize) + 1} health checks failed`,
        });
        throw new Error('Rolling update failed');
      }
    }

    await state.recordDeployment({
      version: config.version,
      strategy: 'rolling-update',
      status: 'success',
    });

    Logger.info('Rolling Update deployment completed successfully');
  }

  // Helper methods for deployment strategies
  static async executeDeployment(config, target) {
    const command = config.commands.deploy.replace('{{target}}', target).replace('{{version}}', config.version);
    Logger.debug(`Executing deployment command: ${command}`);
    const { stdout, stderr } = await execPromise(command);
    if (stderr) Logger.debug('Deployment stderr:', stderr);
    return stdout;
  }

  static async switchTraffic(config, from, to) {
    const command = config.commands.switchTraffic.replace('{{from}}', from).replace('{{to}}', to);
    Logger.debug(`Switching traffic: ${command}`);
    await execPromise(command);
  }

  static async setTrafficSplit(config, percentage) {
    const command = config.commands.setTrafficSplit.replace('{{percentage}}', percentage);
    Logger.debug(`Setting traffic split: ${command}`);
    await execPromise(command);
  }

  static async cleanup(config, target) {
    if (config.commands.cleanup) {
      const command = config.commands.cleanup.replace('{{target}}', target);
      Logger.debug(`Cleaning up: ${command}`);
      await execPromise(command);
    }
  }

  static async updateInstance(config, instance) {
    const command = config.commands.updateInstance.replace('{{instance}}', instance).replace('{{version}}', config.version);
    Logger.debug(`Updating instance ${instance}: ${command}`);
    await execPromise(command);
  }

  static async monitorCanary(config, duration) {
    const checks = config.healthChecks.canary;
    const interval = 30000; // Check every 30 seconds
    const iterations = Math.floor(duration / interval);

    for (let i = 0; i < iterations; i++) {
      const healthy = await HealthChecker.waitForHealthy(checks, interval);
      if (!healthy) return false;
    }

    return true;
  }

  static async promoteCanary(config) {
    const command = config.commands.promoteCanary;
    Logger.debug(`Promoting canary: ${command}`);
    await execPromise(command);
  }
}

// Rollback functionality
class Rollback {
  static async execute(environment, application, targetVersion = null) {
    Logger.info(`Rolling back ${application} in ${environment}`);

    const state = new DeploymentState(environment, application);
    const stateData = await state.load();

    const version = targetVersion || stateData.previousVersions[0];

    if (!version) {
      throw new Error('No previous version available for rollback');
    }

    Logger.info(`Rolling back to version ${version}`);

    // Load deployment config for rollback
    const configPath = path.join(process.cwd(), 'deployment-config.yaml');
    const configData = await fs.readFile(configPath, 'utf8');
    const config = yaml.load(configData);
    config.version = version;

    // Execute rollback using appropriate strategy
    const strategy = config.strategy || 'blue-green';
    await DeploymentStrategy[strategy](config);

    await state.recordDeployment({
      version,
      strategy,
      status: 'success',
      isRollback: true,
    });

    Logger.info(`Rollback to version ${version} completed successfully`);
  }
}

// CLI
async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  try {
    if (command === 'deploy') {
      const configPath = args[1] || 'deployment-config.yaml';
      const configData = await fs.readFile(configPath, 'utf8');
      const config = yaml.load(configData);

      const strategy = config.strategy || 'blue-green';
      Logger.info(`Using deployment strategy: ${strategy}`);

      if (DeploymentStrategy[strategy]) {
        await DeploymentStrategy[strategy](config);
      } else {
        throw new Error(`Unknown deployment strategy: ${strategy}`);
      }
    } else if (command === 'rollback') {
      const environment = args[1];
      const application = args[2];
      const version = args[3]; // Optional

      if (!environment || !application) {
        throw new Error('Usage: deployment-automation.js rollback <environment> <application> [version]');
      }

      await Rollback.execute(environment, application, version);
    } else if (command === 'status') {
      const environment = args[1];
      const application = args[2];

      if (!environment || !application) {
        throw new Error('Usage: deployment-automation.js status <environment> <application>');
      }

      const state = new DeploymentState(environment, application);
      const stateData = await state.load();
      console.log(JSON.stringify(stateData, null, 2));
    } else {
      console.log('Usage:');
      console.log('  deployment-automation.js deploy [config-file]');
      console.log('  deployment-automation.js rollback <environment> <application> [version]');
      console.log('  deployment-automation.js status <environment> <application>');
      process.exit(1);
    }
  } catch (error) {
    Logger.error('Deployment automation failed', { error: error.message, stack: error.stack });
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { DeploymentStrategy, HealthChecker, DeploymentState, Rollback };
