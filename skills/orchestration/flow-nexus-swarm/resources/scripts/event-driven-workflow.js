#!/usr/bin/env node

/**
 * Event-Driven Workflow Automation Script
 *
 * Creates and manages event-driven workflows with message queue processing,
 * GitHub integration, scheduled triggers, and intelligent retry mechanisms.
 *
 * Usage:
 *   node event-driven-workflow.js create --name "CI/CD" --trigger github_push
 *   node event-driven-workflow.js execute --workflow-id <id> --async
 *   node event-driven-workflow.js monitor --workflow-id <id>
 *
 * @requires flow-nexus MCP server
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
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
  const command = args[0];
  const config = {
    command: command || 'help',
    name: null,
    workflowId: null,
    trigger: null,
    steps: null,
    async: false,
    priority: 'medium',
    retryPolicy: 'exponential_backoff',
    maxRetries: 3,
    timeout: 300000, // 5 minutes
    verbose: false
  };

  for (let i = 1; i < args.length; i++) {
    switch (args[i]) {
      case '--name':
        config.name = args[++i];
        break;
      case '--workflow-id':
      case '--id':
        config.workflowId = args[++i];
        break;
      case '--trigger':
        config.trigger = args[++i];
        break;
      case '--steps':
        config.steps = args[++i];
        break;
      case '--async':
        config.async = true;
        break;
      case '--priority':
        config.priority = args[++i];
        break;
      case '--retry-policy':
        config.retryPolicy = args[++i];
        break;
      case '--max-retries':
        config.maxRetries = parseInt(args[++i], 10);
        break;
      case '--timeout':
        config.timeout = parseInt(args[++i], 10);
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
${colors.bright}Event-Driven Workflow Automation${colors.reset}

${colors.cyan}Commands:${colors.reset}
  create              Create a new workflow
  execute             Execute a workflow
  monitor             Monitor workflow status
  list                List all workflows
  queue               Check message queue status
  help                Show this help message

${colors.cyan}Options:${colors.reset}
  --name <name>          Workflow name
  --workflow-id <id>     Workflow ID
  --trigger <type>       Event trigger (github_push, github_pr, schedule, manual)
  --steps <file>         Workflow steps definition file (JSON)
  --async                Async execution with message queue
  --priority <level>     Priority level (low, medium, high, critical)
  --retry-policy <type>  Retry policy (none, linear, exponential_backoff)
  --max-retries <num>    Maximum retry attempts (default: 3)
  --timeout <ms>         Workflow timeout in milliseconds (default: 300000)
  --verbose, -v          Verbose output

${colors.cyan}Examples:${colors.reset}
  ${colors.yellow}# Create CI/CD workflow${colors.reset}
  node event-driven-workflow.js create --name "CI/CD Pipeline" \\
    --trigger github_push --steps ci-cd-steps.json

  ${colors.yellow}# Execute workflow asynchronously${colors.reset}
  node event-driven-workflow.js execute --id workflow_123 --async

  ${colors.yellow}# Monitor workflow execution${colors.reset}
  node event-driven-workflow.js monitor --id workflow_123

  ${colors.yellow}# Check message queue status${colors.reset}
  node event-driven-workflow.js queue
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
 * Load workflow steps from file
 */
function loadSteps(stepsFile) {
  try {
    const stepsPath = path.resolve(process.cwd(), stepsFile);
    const stepsData = fs.readFileSync(stepsPath, 'utf-8');
    return JSON.parse(stepsData);
  } catch (error) {
    throw new Error(`Failed to load steps file: ${error.message}`);
  }
}

/**
 * Create workflow
 */
async function createWorkflow(config) {
  log(`\n${colors.bright}Creating Workflow${colors.reset}`, 'cyan');

  if (!config.name) {
    throw new Error('Workflow name is required (--name)');
  }

  if (!config.steps) {
    throw new Error('Workflow steps file is required (--steps)');
  }

  const steps = loadSteps(config.steps);
  log(`  Name: ${config.name}`, 'blue');
  log(`  Steps: ${steps.length}`, 'blue');
  log(`  Trigger: ${config.trigger || 'manual'}`, 'blue');
  log(`  Priority: ${config.priority}`, 'blue');

  try {
    const workflowDef = {
      name: config.name,
      description: `Event-driven workflow: ${config.name}`,
      steps: steps,
      triggers: config.trigger ? [config.trigger] : [],
      metadata: {
        priority: getPriorityValue(config.priority),
        retry_policy: config.retryPolicy,
        max_retries: config.maxRetries,
        timeout: config.timeout
      }
    };

    const result = executeMCP('workflow_create', workflowDef);

    log(`  ✓ Workflow created successfully`, 'green');
    log(`  Workflow ID: ${result.workflow_id}`, 'green');

    if (config.verbose) {
      log(JSON.stringify(result, null, 2), 'reset');
    }

    return result;
  } catch (error) {
    log(`  ✗ Failed to create workflow: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * Execute workflow
 */
async function executeWorkflow(config) {
  log(`\n${colors.bright}Executing Workflow${colors.reset}`, 'cyan');

  if (!config.workflowId) {
    throw new Error('Workflow ID is required (--workflow-id)');
  }

  log(`  Workflow ID: ${config.workflowId}`, 'blue');
  log(`  Execution Mode: ${config.async ? 'Async (Message Queue)' : 'Sync'}`, 'blue');

  try {
    const result = executeMCP('workflow_execute', {
      workflow_id: config.workflowId,
      input_data: {},
      async: config.async
    });

    log(`  ✓ Workflow execution started`, 'green');

    if (result.execution_id) {
      log(`  Execution ID: ${result.execution_id}`, 'green');
    }

    if (config.async) {
      log(`\n  ${colors.cyan}Monitor execution with:${colors.reset}`);
      log(`  node event-driven-workflow.js monitor --id ${config.workflowId}`, 'yellow');
    }

    if (config.verbose) {
      log(JSON.stringify(result, null, 2), 'reset');
    }

    return result;
  } catch (error) {
    log(`  ✗ Failed to execute workflow: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * Monitor workflow status
 */
async function monitorWorkflow(config) {
  log(`\n${colors.bright}Monitoring Workflow${colors.reset}`, 'cyan');

  if (!config.workflowId) {
    throw new Error('Workflow ID is required (--workflow-id)');
  }

  try {
    const status = executeMCP('workflow_status', {
      workflow_id: config.workflowId,
      include_metrics: true
    });

    log(`\n  Workflow: ${status.name}`, 'blue');
    log(`  Status: ${getStatusColor(status.status)}`, 'reset');
    log(`  Progress: ${status.completed_steps}/${status.total_steps} steps`, 'blue');

    if (status.current_step) {
      log(`  Current Step: ${status.current_step}`, 'blue');
    }

    if (status.metrics) {
      log(`\n  ${colors.bright}Metrics:${colors.reset}`, 'cyan');
      log(`    Duration: ${status.metrics.duration_ms}ms`, 'blue');
      log(`    Success Rate: ${status.metrics.success_rate}%`, 'blue');

      if (status.metrics.error_count > 0) {
        log(`    Errors: ${status.metrics.error_count}`, 'red');
      }
    }

    if (config.verbose) {
      log(`\n${JSON.stringify(status, null, 2)}`, 'reset');
    }

    return status;
  } catch (error) {
    log(`  ✗ Failed to get workflow status: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * List workflows
 */
async function listWorkflows(config) {
  log(`\n${colors.bright}Workflows${colors.reset}`, 'cyan');

  try {
    const result = executeMCP('workflow_list', {
      limit: 20,
      offset: 0
    });

    if (!result.workflows || result.workflows.length === 0) {
      log(`  No workflows found`, 'yellow');
      return;
    }

    log(`  Total: ${result.total}`, 'blue');
    log(`  Showing: ${result.workflows.length}\n`, 'blue');

    result.workflows.forEach((workflow, index) => {
      log(`  ${index + 1}. ${workflow.name}`, 'bright');
      log(`     ID: ${workflow.workflow_id}`, 'blue');
      log(`     Status: ${getStatusColor(workflow.status)}`, 'reset');
      log(`     Triggers: ${workflow.triggers.join(', ')}`, 'blue');
      log(`     Created: ${new Date(workflow.created_at).toLocaleString()}`, 'blue');
      log('');
    });

    return result;
  } catch (error) {
    log(`  ✗ Failed to list workflows: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * Check message queue status
 */
async function checkQueueStatus(config) {
  log(`\n${colors.bright}Message Queue Status${colors.reset}`, 'cyan');

  try {
    const status = executeMCP('workflow_queue_status', {
      include_messages: config.verbose
    });

    log(`  Total Queues: ${status.total_queues}`, 'blue');
    log(`  Total Messages: ${status.total_messages}`, 'blue');
    log(`  Processing: ${status.processing}`, 'blue');
    log(`  Pending: ${status.pending}`, 'blue');

    if (status.queues) {
      log(`\n  ${colors.bright}Queue Breakdown:${colors.reset}`, 'cyan');

      status.queues.forEach(queue => {
        log(`    ${queue.name}: ${queue.message_count} messages`, 'blue');
      });
    }

    if (config.verbose && status.messages) {
      log(`\n  ${colors.bright}Pending Messages:${colors.reset}`, 'cyan');

      status.messages.slice(0, 10).forEach((msg, index) => {
        log(`    ${index + 1}. ${msg.workflow_id} - ${msg.status}`, 'blue');
      });
    }

    return status;
  } catch (error) {
    log(`  ✗ Failed to get queue status: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * Helper: Get priority value
 */
function getPriorityValue(priority) {
  const priorities = { low: 1, medium: 5, high: 8, critical: 10 };
  return priorities[priority] || 5;
}

/**
 * Helper: Get colored status text
 */
function getStatusColor(status) {
  const statusColors = {
    running: 'cyan',
    completed: 'green',
    failed: 'red',
    pending: 'yellow'
  };

  const color = statusColors[status] || 'reset';
  return `${colors[color]}${status}${colors.reset}`;
}

/**
 * Main execution
 */
async function main() {
  const config = parseArgs();

  try {
    switch (config.command) {
      case 'create':
        await createWorkflow(config);
        break;
      case 'execute':
        await executeWorkflow(config);
        break;
      case 'monitor':
        await monitorWorkflow(config);
        break;
      case 'list':
        await listWorkflows(config);
        break;
      case 'queue':
        await checkQueueStatus(config);
        break;
      case 'help':
        printHelp();
        break;
      default:
        log(`Unknown command: ${config.command}`, 'red');
        log(`Run with --help for usage information\n`, 'yellow');
        process.exit(1);
    }

    process.exit(0);
  } catch (error) {
    log(`\n${colors.red}Error: ${error.message}${colors.reset}\n`, 'red');
    process.exit(1);
  }
}

// Execute if run directly
if (require.main === module) {
  main();
}

module.exports = { createWorkflow, executeWorkflow, monitorWorkflow, listWorkflows, checkQueueStatus };
