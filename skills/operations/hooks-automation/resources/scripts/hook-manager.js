#!/usr/bin/env node
/**
 * hook-manager.js - Manage hook lifecycle and execution
 * Usage: node hook-manager.js <command> [options]
 * Commands: start, stop, list, enable, disable, metrics
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const os = require('os');

// Configuration
const HOOKS_DIR = process.env.HOOKS_DIR || path.join(os.homedir(), '.claude-flow', 'hooks');
const STATE_FILE = path.join(HOOKS_DIR, '.hook-manager-state.json');

// Colors for terminal output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

class HookManager {
  constructor() {
    this.state = this.loadState();
    this.hooks = ['pre-task', 'post-edit', 'post-task', 'session'];
  }

  // Load manager state from disk
  loadState() {
    try {
      if (fs.existsSync(STATE_FILE)) {
        const data = fs.readFileSync(STATE_FILE, 'utf8');
        return JSON.parse(data);
      }
    } catch (error) {
      this.logError(`Failed to load state: ${error.message}`);
    }

    return {
      enabled: {},
      metrics: {},
      lastRun: {},
      running: false
    };
  }

  // Save manager state to disk
  saveState() {
    try {
      fs.writeFileSync(STATE_FILE, JSON.stringify(this.state, null, 2));
    } catch (error) {
      this.logError(`Failed to save state: ${error.message}`);
    }
  }

  // Logging methods
  logInfo(msg) {
    console.log(`${colors.green}[INFO]${colors.reset} ${msg}`);
  }

  logWarn(msg) {
    console.log(`${colors.yellow}[WARN]${colors.reset} ${msg}`);
  }

  logError(msg) {
    console.log(`${colors.red}[ERROR]${colors.reset} ${msg}`);
  }

  logDebug(msg) {
    if (process.env.DEBUG) {
      console.log(`${colors.blue}[DEBUG]${colors.reset} ${msg}`);
    }
  }

  // Check if hooks directory exists
  checkHooksDir() {
    if (!fs.existsSync(HOOKS_DIR)) {
      this.logError(`Hooks directory not found: ${HOOKS_DIR}`);
      this.logInfo('Run hook-installer.sh to set up hooks first');
      return false;
    }
    return true;
  }

  // Start hook manager
  start() {
    this.logInfo('Starting hook manager...');

    if (!this.checkHooksDir()) {
      return;
    }

    this.state.running = true;
    this.saveState();

    // Initialize all hooks as enabled by default
    this.hooks.forEach(hook => {
      if (!(hook in this.state.enabled)) {
        this.state.enabled[hook] = true;
      }
      if (!(hook in this.state.metrics)) {
        this.state.metrics[hook] = {
          executions: 0,
          successes: 0,
          failures: 0,
          totalDuration: 0
        };
      }
    });

    this.saveState();
    this.logInfo('Hook manager started successfully');
    this.list();
  }

  // Stop hook manager
  stop() {
    this.logInfo('Stopping hook manager...');
    this.state.running = false;
    this.saveState();
    this.logInfo('Hook manager stopped');
  }

  // List all hooks and their status
  list() {
    this.logInfo('Registered hooks:');
    console.log();

    this.hooks.forEach(hook => {
      const enabled = this.state.enabled[hook] !== false;
      const status = enabled ? `${colors.green}enabled${colors.reset}` : `${colors.red}disabled${colors.reset}`;
      const hookDir = path.join(HOOKS_DIR, hook);
      const exists = fs.existsSync(hookDir) ? '✓' : '✗';

      console.log(`  ${exists} ${hook.padEnd(15)} ${status}`);

      const metrics = this.state.metrics[hook];
      if (metrics && metrics.executions > 0) {
        const avgDuration = (metrics.totalDuration / metrics.executions).toFixed(2);
        console.log(`    └─ Executions: ${metrics.executions}, Success rate: ${((metrics.successes / metrics.executions) * 100).toFixed(1)}%, Avg duration: ${avgDuration}ms`);
      }
    });
    console.log();
  }

  // Enable specific hook
  enable(hookName) {
    if (!this.hooks.includes(hookName)) {
      this.logError(`Unknown hook: ${hookName}`);
      this.logInfo(`Available hooks: ${this.hooks.join(', ')}`);
      return;
    }

    this.state.enabled[hookName] = true;
    this.saveState();
    this.logInfo(`Enabled hook: ${hookName}`);
  }

  // Disable specific hook
  disable(hookName) {
    if (!this.hooks.includes(hookName)) {
      this.logError(`Unknown hook: ${hookName}`);
      this.logInfo(`Available hooks: ${this.hooks.join(', ')}`);
      return;
    }

    this.state.enabled[hookName] = false;
    this.saveState();
    this.logInfo(`Disabled hook: ${hookName}`);
  }

  // Execute a hook
  async execute(hookName, args = []) {
    if (!this.state.enabled[hookName]) {
      this.logDebug(`Hook ${hookName} is disabled, skipping`);
      return { success: false, skipped: true };
    }

    const hookScript = path.join(HOOKS_DIR, hookName, 'run.sh');
    if (!fs.existsSync(hookScript)) {
      this.logWarn(`Hook script not found: ${hookScript}`);
      return { success: false, error: 'Script not found' };
    }

    const startTime = Date.now();

    try {
      this.logDebug(`Executing hook: ${hookName} with args: ${args.join(' ')}`);

      const result = execSync(`bash "${hookScript}" ${args.join(' ')}`, {
        encoding: 'utf8',
        timeout: 30000 // 30 second timeout
      });

      const duration = Date.now() - startTime;

      // Update metrics
      const metrics = this.state.metrics[hookName];
      metrics.executions++;
      metrics.successes++;
      metrics.totalDuration += duration;
      this.state.lastRun[hookName] = new Date().toISOString();
      this.saveState();

      this.logDebug(`Hook ${hookName} completed in ${duration}ms`);
      return { success: true, output: result, duration };

    } catch (error) {
      const duration = Date.now() - startTime;

      // Update metrics
      const metrics = this.state.metrics[hookName];
      metrics.executions++;
      metrics.failures++;
      metrics.totalDuration += duration;
      this.saveState();

      this.logError(`Hook ${hookName} failed: ${error.message}`);
      return { success: false, error: error.message, duration };
    }
  }

  // Display metrics for all hooks
  showMetrics() {
    this.logInfo('Hook execution metrics:');
    console.log();

    this.hooks.forEach(hook => {
      const metrics = this.state.metrics[hook];
      if (!metrics || metrics.executions === 0) {
        console.log(`${colors.cyan}${hook}${colors.reset}: No executions yet`);
        return;
      }

      const successRate = ((metrics.successes / metrics.executions) * 100).toFixed(1);
      const avgDuration = (metrics.totalDuration / metrics.executions).toFixed(2);
      const lastRun = this.state.lastRun[hook] || 'Never';

      console.log(`${colors.cyan}${hook}${colors.reset}:`);
      console.log(`  Total executions: ${metrics.executions}`);
      console.log(`  Successes: ${metrics.successes}`);
      console.log(`  Failures: ${metrics.failures}`);
      console.log(`  Success rate: ${successRate}%`);
      console.log(`  Average duration: ${avgDuration}ms`);
      console.log(`  Total duration: ${metrics.totalDuration}ms`);
      console.log(`  Last run: ${lastRun}`);
      console.log();
    });
  }

  // Reset metrics
  resetMetrics() {
    this.logInfo('Resetting metrics...');
    this.hooks.forEach(hook => {
      this.state.metrics[hook] = {
        executions: 0,
        successes: 0,
        failures: 0,
        totalDuration: 0
      };
    });
    this.state.lastRun = {};
    this.saveState();
    this.logInfo('Metrics reset successfully');
  }

  // Test all hooks
  async testAll() {
    this.logInfo('Testing all hooks...');
    console.log();

    for (const hook of this.hooks) {
      if (!this.state.enabled[hook]) {
        this.logDebug(`Skipping disabled hook: ${hook}`);
        continue;
      }

      const result = await this.execute(hook, ['test']);

      if (result.success) {
        console.log(`${colors.green}✓${colors.reset} ${hook} - OK (${result.duration}ms)`);
      } else if (result.skipped) {
        console.log(`${colors.yellow}⊘${colors.reset} ${hook} - Skipped (disabled)`);
      } else {
        console.log(`${colors.red}✗${colors.reset} ${hook} - Failed: ${result.error}`);
      }
    }
    console.log();
  }
}

// CLI interface
function main() {
  const args = process.argv.slice(2);
  const command = args[0];
  const manager = new HookManager();

  switch (command) {
    case 'start':
      manager.start();
      break;

    case 'stop':
      manager.stop();
      break;

    case 'list':
      manager.list();
      break;

    case 'enable':
      if (!args[1]) {
        manager.logError('Please specify hook name');
        manager.logInfo(`Usage: ${process.argv[1]} enable <hook-name>`);
        process.exit(1);
      }
      manager.enable(args[1]);
      break;

    case 'disable':
      if (!args[1]) {
        manager.logError('Please specify hook name');
        manager.logInfo(`Usage: ${process.argv[1]} disable <hook-name>`);
        process.exit(1);
      }
      manager.disable(args[1]);
      break;

    case 'metrics':
      manager.showMetrics();
      break;

    case 'reset-metrics':
      manager.resetMetrics();
      break;

    case 'test':
      manager.testAll().catch(err => {
        manager.logError(`Test failed: ${err.message}`);
        process.exit(1);
      });
      break;

    case 'execute':
      if (!args[1]) {
        manager.logError('Please specify hook name');
        manager.logInfo(`Usage: ${process.argv[1]} execute <hook-name> [args...]`);
        process.exit(1);
      }
      manager.execute(args[1], args.slice(2)).then(result => {
        if (result.success) {
          console.log(result.output);
          process.exit(0);
        } else {
          manager.logError(result.error || 'Execution failed');
          process.exit(1);
        }
      });
      break;

    default:
      console.log('Hook Manager - Manage Claude Flow hooks');
      console.log();
      console.log('Usage: node hook-manager.js <command> [options]');
      console.log();
      console.log('Commands:');
      console.log('  start              Start hook manager');
      console.log('  stop               Stop hook manager');
      console.log('  list               List all hooks and their status');
      console.log('  enable <hook>      Enable specific hook');
      console.log('  disable <hook>     Disable specific hook');
      console.log('  execute <hook>     Execute specific hook');
      console.log('  metrics            Show execution metrics');
      console.log('  reset-metrics      Reset all metrics');
      console.log('  test               Test all hooks');
      console.log();
      console.log('Available hooks: pre-task, post-edit, post-task, session');
      process.exit(command ? 1 : 0);
  }
}

// Run CLI
if (require.main === module) {
  main();
}

module.exports = HookManager;
