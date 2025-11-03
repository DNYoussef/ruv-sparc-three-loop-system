#!/usr/bin/env node

/**
 * Flow Nexus Swarm Monitoring & Analytics Script
 *
 * Real-time monitoring of swarm health, agent performance, task execution,
 * and resource utilization with comprehensive metrics and alerts.
 *
 * Usage:
 *   node swarm-monitor.js --swarm-id <id>
 *   node swarm-monitor.js --all --watch
 *   node swarm-monitor.js --export metrics.json
 *
 * @requires flow-nexus MCP server
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m'
};

/**
 * Parse command-line arguments
 */
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    swarmId: null,
    all: false,
    watch: false,
    interval: 5000, // 5 seconds
    export: null,
    metrics: true,
    agents: true,
    tasks: true,
    verbose: false
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--swarm-id':
      case '--id':
        config.swarmId = args[++i];
        break;
      case '--all':
        config.all = true;
        break;
      case '--watch':
      case '-w':
        config.watch = true;
        break;
      case '--interval':
        config.interval = parseInt(args[++i], 10) * 1000;
        break;
      case '--export':
        config.export = args[++i];
        break;
      case '--no-metrics':
        config.metrics = false;
        break;
      case '--no-agents':
        config.agents = false;
        break;
      case '--no-tasks':
        config.tasks = false;
        break;
      case '--verbose':
      case '-v':
        config.verbose = true;
        break;
      case '--help':
      case '-h':
        printHelp();
        process.exit(0);
    }
  }

  return config;
}

/**
 * Print help message
 */
function printHelp() {
  console.log(`
${colors.bright}Flow Nexus Swarm Monitoring${colors.reset}

${colors.cyan}Usage:${colors.reset}
  node swarm-monitor.js [options]

${colors.cyan}Options:${colors.reset}
  --swarm-id <id>       Monitor specific swarm
  --all                 Monitor all active swarms
  --watch, -w           Continuous monitoring (auto-refresh)
  --interval <sec>      Refresh interval in seconds (default: 5)
  --export <file>       Export metrics to JSON file
  --no-metrics          Skip metrics display
  --no-agents           Skip agent status display
  --no-tasks            Skip task status display
  --verbose, -v         Verbose output
  --help, -h            Show this help message

${colors.cyan}Examples:${colors.reset}
  ${colors.yellow}# Monitor specific swarm${colors.reset}
  node swarm-monitor.js --swarm-id swarm_123

  ${colors.yellow}# Watch all swarms with auto-refresh${colors.reset}
  node swarm-monitor.js --all --watch --interval 10

  ${colors.yellow}# Export metrics to file${colors.reset}
  node swarm-monitor.js --swarm-id swarm_123 --export metrics.json
`);
}

/**
 * Log with color
 */
function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

/**
 * Execute MCP tool
 */
function executeMCP(tool, params) {
  const cmd = `npx flow-nexus@latest mcp call ${tool} '${JSON.stringify(params)}'`;

  try {
    const result = execSync(cmd, { encoding: 'utf-8', stdio: 'pipe' });
    return JSON.parse(result);
  } catch (error) {
    throw new Error(`MCP call failed: ${error.message}`);
  }
}

/**
 * Clear terminal screen
 */
function clearScreen() {
  process.stdout.write('\x1Bc');
}

/**
 * Get swarm status
 */
function getSwarmStatus(swarmId) {
  return executeMCP('swarm_status', { swarm_id: swarmId });
}

/**
 * List all swarms
 */
function listSwarms() {
  return executeMCP('swarm_list', { status: 'active' });
}

/**
 * Display swarm header
 */
function displaySwarmHeader(swarm) {
  log(`\n${colors.bright}═══════════════════════════════════════════════════════════${colors.reset}`, 'cyan');
  log(`  Swarm: ${swarm.name || swarm.swarm_id}`, 'bright');
  log(`  ID: ${swarm.swarm_id}`, 'blue');
  log(`  Topology: ${swarm.topology} | Strategy: ${swarm.strategy}`, 'blue');
  log(`  Status: ${getHealthColor(swarm.status)}`, 'reset');
  log(`  Uptime: ${formatDuration(swarm.uptime_ms)}`, 'blue');
  log(`${colors.bright}═══════════════════════════════════════════════════════════${colors.reset}\n`, 'cyan');
}

/**
 * Display agent status
 */
function displayAgentStatus(agents) {
  if (!agents || agents.length === 0) {
    log(`  ${colors.yellow}No agents available${colors.reset}`, 'yellow');
    return;
  }

  log(`${colors.bright}Agents (${agents.length}):${colors.reset}`, 'cyan');

  const agentsByType = {};
  agents.forEach(agent => {
    if (!agentsByType[agent.type]) {
      agentsByType[agent.type] = [];
    }
    agentsByType[agent.type].push(agent);
  });

  Object.keys(agentsByType).forEach(type => {
    const typeAgents = agentsByType[type];
    const activeCount = typeAgents.filter(a => a.status === 'active').length;
    const busyCount = typeAgents.filter(a => a.status === 'busy').length;

    log(`\n  ${colors.bright}${type}${colors.reset} (${typeAgents.length})`, 'magenta');

    typeAgents.forEach(agent => {
      const statusIcon = getStatusIcon(agent.status);
      const workloadBar = getWorkloadBar(agent.workload || 0);

      log(`    ${statusIcon} ${agent.name}`, 'blue');
      log(`       ${workloadBar} ${Math.round((agent.workload || 0) * 100)}%`, 'dim');

      if (agent.current_task) {
        log(`       Task: ${agent.current_task}`, 'dim');
      }
    });
  });

  log('');
}

/**
 * Display metrics
 */
function displayMetrics(metrics) {
  if (!metrics) {
    return;
  }

  log(`${colors.bright}Performance Metrics:${colors.reset}`, 'cyan');

  log(`  Tasks Completed: ${metrics.tasks_completed || 0}`, 'blue');
  log(`  Tasks Running: ${metrics.tasks_running || 0}`, 'blue');
  log(`  Tasks Queued: ${metrics.tasks_queued || 0}`, 'blue');

  if (metrics.success_rate !== undefined) {
    const successColor = metrics.success_rate > 90 ? 'green' : metrics.success_rate > 70 ? 'yellow' : 'red';
    log(`  Success Rate: ${colors[successColor]}${metrics.success_rate.toFixed(1)}%${colors.reset}`, 'reset');
  }

  if (metrics.avg_task_duration_ms) {
    log(`  Avg Task Duration: ${formatDuration(metrics.avg_task_duration_ms)}`, 'blue');
  }

  if (metrics.throughput) {
    log(`  Throughput: ${metrics.throughput.toFixed(2)} tasks/min`, 'blue');
  }

  log('');
}

/**
 * Display task queue
 */
function displayTaskQueue(tasks) {
  if (!tasks || tasks.length === 0) {
    return;
  }

  log(`${colors.bright}Active Tasks (${tasks.length}):${colors.reset}`, 'cyan');

  tasks.slice(0, 10).forEach((task, index) => {
    const statusIcon = getStatusIcon(task.status);
    const progress = task.progress ? ` (${task.progress}%)` : '';

    log(`  ${index + 1}. ${statusIcon} ${task.name || task.task_id}${progress}`, 'blue');

    if (task.assigned_agent) {
      log(`     Agent: ${task.assigned_agent}`, 'dim');
    }

    if (task.duration_ms) {
      log(`     Duration: ${formatDuration(task.duration_ms)}`, 'dim');
    }
  });

  if (tasks.length > 10) {
    log(`  ... and ${tasks.length - 10} more tasks`, 'dim');
  }

  log('');
}

/**
 * Monitor single swarm
 */
function monitorSwarm(swarmId, config) {
  try {
    const status = getSwarmStatus(swarmId);

    displaySwarmHeader(status);

    if (config.agents && status.agents) {
      displayAgentStatus(status.agents);
    }

    if (config.metrics && status.metrics) {
      displayMetrics(status.metrics);
    }

    if (config.tasks && status.tasks) {
      displayTaskQueue(status.tasks);
    }

    log(`${colors.dim}Last updated: ${new Date().toLocaleTimeString()}${colors.reset}`, 'dim');

    return status;
  } catch (error) {
    log(`Error monitoring swarm: ${error.message}`, 'red');
    return null;
  }
}

/**
 * Monitor all swarms
 */
function monitorAllSwarms(config) {
  try {
    const swarms = listSwarms();

    if (!swarms.swarms || swarms.swarms.length === 0) {
      log(`\n${colors.yellow}No active swarms found${colors.reset}\n`, 'yellow');
      return;
    }

    log(`\n${colors.bright}Active Swarms: ${swarms.swarms.length}${colors.reset}\n`, 'cyan');

    swarms.swarms.forEach((swarm, index) => {
      const status = getSwarmStatus(swarm.swarm_id);

      log(`${index + 1}. ${status.name || status.swarm_id}`, 'bright');
      log(`   Topology: ${status.topology} | Agents: ${status.active_agents}/${status.max_agents}`, 'blue');
      log(`   Status: ${getHealthColor(status.status)} | Uptime: ${formatDuration(status.uptime_ms)}`, 'reset');

      if (status.metrics && status.metrics.tasks_running > 0) {
        log(`   Running Tasks: ${status.metrics.tasks_running}`, 'blue');
      }

      log('');
    });

    log(`${colors.dim}Last updated: ${new Date().toLocaleTimeString()}${colors.reset}\n`, 'dim');
  } catch (error) {
    log(`Error monitoring swarms: ${error.message}`, 'red');
  }
}

/**
 * Export metrics to file
 */
function exportMetrics(data, filename) {
  try {
    const filepath = path.resolve(process.cwd(), filename);
    const exportData = {
      timestamp: new Date().toISOString(),
      data: data
    };

    fs.writeFileSync(filepath, JSON.stringify(exportData, null, 2), 'utf-8');
    log(`\n✓ Metrics exported to: ${filepath}`, 'green');
  } catch (error) {
    log(`Error exporting metrics: ${error.message}`, 'red');
  }
}

/**
 * Helper: Format duration
 */
function formatDuration(ms) {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${(ms / 60000).toFixed(1)}m`;
  return `${(ms / 3600000).toFixed(1)}h`;
}

/**
 * Helper: Get health color
 */
function getHealthColor(status) {
  const statusColors = {
    healthy: 'green',
    operational: 'green',
    degraded: 'yellow',
    warning: 'yellow',
    error: 'red',
    offline: 'red'
  };

  const color = statusColors[status] || 'reset';
  return `${colors[color]}${status}${colors.reset}`;
}

/**
 * Helper: Get status icon
 */
function getStatusIcon(status) {
  const icons = {
    active: '●',
    busy: '◉',
    idle: '○',
    running: '▶',
    completed: '✓',
    failed: '✗',
    pending: '◷'
  };

  const icon = icons[status] || '?';
  const iconColors = {
    active: 'green',
    busy: 'yellow',
    idle: 'dim',
    running: 'cyan',
    completed: 'green',
    failed: 'red',
    pending: 'yellow'
  };

  const color = iconColors[status] || 'reset';
  return `${colors[color]}${icon}${colors.reset}`;
}

/**
 * Helper: Get workload bar
 */
function getWorkloadBar(workload) {
  const barLength = 20;
  const filled = Math.round(workload * barLength);
  const empty = barLength - filled;

  const bar = '█'.repeat(filled) + '░'.repeat(empty);
  const color = workload > 0.8 ? 'red' : workload > 0.5 ? 'yellow' : 'green';

  return `${colors[color]}${bar}${colors.reset}`;
}

/**
 * Main execution
 */
async function main() {
  const config = parseArgs();

  if (config.watch) {
    // Continuous monitoring mode
    log(`${colors.cyan}Starting continuous monitoring (Press Ctrl+C to exit)${colors.reset}\n`);

    setInterval(() => {
      clearScreen();

      if (config.all) {
        monitorAllSwarms(config);
      } else if (config.swarmId) {
        const status = monitorSwarm(config.swarmId, config);

        if (config.export && status) {
          exportMetrics(status, config.export);
        }
      } else {
        log(`Error: Specify --swarm-id or --all`, 'red');
        process.exit(1);
      }
    }, config.interval);
  } else {
    // Single snapshot
    if (config.all) {
      monitorAllSwarms(config);
    } else if (config.swarmId) {
      const status = monitorSwarm(config.swarmId, config);

      if (config.export && status) {
        exportMetrics(status, config.export);
      }
    } else {
      log(`Error: Specify --swarm-id or --all`, 'red');
      printHelp();
      process.exit(1);
    }
  }
}

// Execute if run directly
if (require.main === module) {
  main();
}

module.exports = { monitorSwarm, monitorAllSwarms, exportMetrics };
