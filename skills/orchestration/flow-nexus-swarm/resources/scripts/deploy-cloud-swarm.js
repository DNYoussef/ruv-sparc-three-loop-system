#!/usr/bin/env node

/**
 * Flow Nexus Cloud Swarm Deployment Script
 *
 * Deploys a fully-configured AI agent swarm to Flow Nexus cloud infrastructure
 * with customizable topology, agent types, and workflow automation.
 *
 * Usage:
 *   node deploy-cloud-swarm.js --config swarm-config.json
 *   node deploy-cloud-swarm.js --topology hierarchical --agents 8 --workflow ci-cd
 *
 * @requires flow-nexus MCP server
 * @requires Active Flow Nexus authentication
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// ANSI color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

/**
 * Parse command-line arguments
 */
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    configFile: null,
    topology: 'hierarchical',
    maxAgents: 8,
    strategy: 'balanced',
    workflow: null,
    name: `swarm-${Date.now()}`,
    verbose: false
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--config':
        config.configFile = args[++i];
        break;
      case '--topology':
        config.topology = args[++i];
        break;
      case '--agents':
        config.maxAgents = parseInt(args[++i], 10);
        break;
      case '--strategy':
        config.strategy = args[++i];
        break;
      case '--workflow':
        config.workflow = args[++i];
        break;
      case '--name':
        config.name = args[++i];
        break;
      case '--verbose':
      case '-v':
        config.verbose = true;
        break;
      case '--help':
      case '-h':
        printHelp();
        process.exit(0);
      default:
        console.error(`${colors.red}Unknown argument: ${args[i]}${colors.reset}`);
        process.exit(1);
    }
  }

  return config;
}

/**
 * Print help message
 */
function printHelp() {
  console.log(`
${colors.bright}Flow Nexus Cloud Swarm Deployment${colors.reset}

${colors.cyan}Usage:${colors.reset}
  node deploy-cloud-swarm.js [options]

${colors.cyan}Options:${colors.reset}
  --config <file>       Load configuration from JSON file
  --topology <type>     Swarm topology (hierarchical, mesh, ring, star)
  --agents <num>        Maximum number of agents (default: 8)
  --strategy <type>     Distribution strategy (balanced, specialized, adaptive)
  --workflow <file>     Workflow definition file to deploy
  --name <name>         Swarm name (default: swarm-<timestamp>)
  --verbose, -v         Verbose output
  --help, -h            Show this help message

${colors.cyan}Examples:${colors.reset}
  ${colors.yellow}# Deploy with default settings${colors.reset}
  node deploy-cloud-swarm.js

  ${colors.yellow}# Deploy with custom topology and agents${colors.reset}
  node deploy-cloud-swarm.js --topology mesh --agents 5

  ${colors.yellow}# Deploy from configuration file${colors.reset}
  node deploy-cloud-swarm.js --config swarm-config.json

  ${colors.yellow}# Deploy with workflow automation${colors.reset}
  node deploy-cloud-swarm.js --workflow ci-cd-pipeline.json
`);
}

/**
 * Log message with color
 */
function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

/**
 * Execute MCP tool via npx flow-nexus CLI
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
 * Load configuration from file
 */
function loadConfig(configFile) {
  try {
    const configPath = path.resolve(process.cwd(), configFile);
    const configData = fs.readFileSync(configPath, 'utf-8');
    return JSON.parse(configData);
  } catch (error) {
    throw new Error(`Failed to load config file: ${error.message}`);
  }
}

/**
 * Initialize swarm with specified configuration
 */
async function initializeSwarm(config) {
  log(`\n${colors.bright}[1/4] Initializing swarm: ${config.name}${colors.reset}`, 'cyan');
  log(`  Topology: ${config.topology}`, 'blue');
  log(`  Max Agents: ${config.maxAgents}`, 'blue');
  log(`  Strategy: ${config.strategy}`, 'blue');

  try {
    const result = executeMCP('swarm_init', {
      topology: config.topology,
      maxAgents: config.maxAgents,
      strategy: config.strategy
    });

    if (config.verbose) {
      log(JSON.stringify(result, null, 2), 'reset');
    }

    log(`  ✓ Swarm initialized successfully`, 'green');
    return result.swarm_id;
  } catch (error) {
    log(`  ✗ Failed to initialize swarm: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * Spawn agents based on configuration
 */
async function spawnAgents(config, swarmId) {
  log(`\n${colors.bright}[2/4] Spawning agents${colors.reset}`, 'cyan');

  const agentTypes = config.agents || [
    { type: 'coordinator', name: 'Coordinator' },
    { type: 'researcher', name: 'Researcher' },
    { type: 'coder', name: 'Coder' },
    { type: 'analyst', name: 'Analyst' }
  ];

  const spawnedAgents = [];

  for (const agent of agentTypes) {
    try {
      log(`  Spawning ${agent.type}: ${agent.name}...`, 'blue');

      const result = executeMCP('agent_spawn', {
        type: agent.type,
        name: agent.name,
        capabilities: agent.capabilities || []
      });

      spawnedAgents.push(result);
      log(`    ✓ ${agent.name} spawned`, 'green');

      if (config.verbose) {
        log(`      Agent ID: ${result.agent_id}`, 'reset');
      }
    } catch (error) {
      log(`    ✗ Failed to spawn ${agent.name}: ${error.message}`, 'red');
    }
  }

  log(`  ✓ ${spawnedAgents.length}/${agentTypes.length} agents spawned successfully`, 'green');
  return spawnedAgents;
}

/**
 * Deploy workflow if specified
 */
async function deployWorkflow(config, swarmId) {
  if (!config.workflow) {
    log(`\n${colors.bright}[3/4] Skipping workflow deployment (not specified)${colors.reset}`, 'yellow');
    return null;
  }

  log(`\n${colors.bright}[3/4] Deploying workflow${colors.reset}`, 'cyan');

  try {
    let workflowDef;

    if (typeof config.workflow === 'string') {
      // Load from file
      const workflowPath = path.resolve(process.cwd(), config.workflow);
      workflowDef = JSON.parse(fs.readFileSync(workflowPath, 'utf-8'));
    } else {
      // Use inline definition
      workflowDef = config.workflow;
    }

    log(`  Workflow: ${workflowDef.name}`, 'blue');
    log(`  Steps: ${workflowDef.steps.length}`, 'blue');

    const result = executeMCP('workflow_create', workflowDef);

    log(`  ✓ Workflow deployed successfully`, 'green');

    if (config.verbose) {
      log(`    Workflow ID: ${result.workflow_id}`, 'reset');
    }

    return result;
  } catch (error) {
    log(`  ✗ Failed to deploy workflow: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * Get swarm status and display summary
 */
async function getSwarmStatus(swarmId) {
  log(`\n${colors.bright}[4/4] Retrieving swarm status${colors.reset}`, 'cyan');

  try {
    const status = executeMCP('swarm_status', { swarm_id: swarmId });

    log(`  ✓ Swarm is operational`, 'green');
    log(`\n${colors.bright}Swarm Summary:${colors.reset}`, 'cyan');
    log(`  ID: ${status.swarm_id}`, 'blue');
    log(`  Topology: ${status.topology}`, 'blue');
    log(`  Active Agents: ${status.active_agents}/${status.max_agents}`, 'blue');
    log(`  Status: ${status.status}`, 'blue');
    log(`  Created: ${new Date(status.created_at).toLocaleString()}`, 'blue');

    return status;
  } catch (error) {
    log(`  ✗ Failed to get swarm status: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * Main deployment orchestration
 */
async function main() {
  log(`${colors.bright}==============================================`, 'cyan');
  log(`  Flow Nexus Cloud Swarm Deployment`, 'cyan');
  log(`  Version 1.0.0`, 'cyan');
  log(`==============================================\n`, 'cyan');

  try {
    // Parse arguments
    let config = parseArgs();

    // Load config file if specified
    if (config.configFile) {
      log(`Loading configuration from: ${config.configFile}`, 'yellow');
      const fileConfig = loadConfig(config.configFile);
      config = { ...config, ...fileConfig };
    }

    // Execute deployment steps
    const swarmId = await initializeSwarm(config);
    const agents = await spawnAgents(config, swarmId);
    const workflow = await deployWorkflow(config, swarmId);
    const status = await getSwarmStatus(swarmId);

    // Success summary
    log(`\n${colors.bright}${colors.green}✓ Deployment completed successfully!${colors.reset}`, 'green');
    log(`\n${colors.cyan}Next Steps:${colors.reset}`);
    log(`  1. Monitor swarm: npx flow-nexus@latest swarm status ${swarmId}`);
    log(`  2. Orchestrate tasks: npx flow-nexus@latest task orchestrate "<task>"`);
    log(`  3. Scale swarm: npx flow-nexus@latest swarm scale --agents <num>`);

    if (workflow) {
      log(`  4. Execute workflow: npx flow-nexus@latest workflow execute ${workflow.workflow_id}`);
    }

    log(`\n${colors.cyan}Documentation:${colors.reset} https://flow-nexus.ruv.io/docs\n`);

    process.exit(0);
  } catch (error) {
    log(`\n${colors.red}${colors.bright}✗ Deployment failed: ${error.message}${colors.reset}`, 'red');
    log(`\nFor help, run: node deploy-cloud-swarm.js --help\n`, 'yellow');
    process.exit(1);
  }
}

// Execute if run directly
if (require.main === module) {
  main();
}

module.exports = { parseArgs, initializeSwarm, spawnAgents, deployWorkflow, getSwarmStatus };
