#!/usr/bin/env node

/**
 * Flow Nexus Template Manager
 *
 * Manage swarm and workflow templates: list, view details, deploy from templates,
 * create custom templates, and share reusable configurations.
 *
 * Usage:
 *   node template-manager.js list --category quickstart
 *   node template-manager.js deploy --template full-stack-dev
 *   node template-manager.js create --from-swarm swarm_123 --name my-template
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
    category: 'all',
    template: null,
    swarmId: null,
    name: null,
    output: null,
    overrides: {},
    verbose: false
  };

  for (let i = 1; i < args.length; i++) {
    switch (args[i]) {
      case '--category':
        config.category = args[++i];
        break;
      case '--template':
        config.template = args[++i];
        break;
      case '--from-swarm':
        config.swarmId = args[++i];
        break;
      case '--name':
        config.name = args[++i];
        break;
      case '--output':
        config.output = args[++i];
        break;
      case '--agents':
        config.overrides.maxAgents = parseInt(args[++i], 10);
        break;
      case '--strategy':
        config.overrides.strategy = args[++i];
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
${colors.bright}Flow Nexus Template Manager${colors.reset}

${colors.cyan}Commands:${colors.reset}
  list                List available templates
  view                View template details
  deploy              Deploy swarm from template
  create              Create custom template from swarm
  export              Export template to file
  help                Show this help message

${colors.cyan}Options:${colors.reset}
  --category <type>     Template category (quickstart, specialized, enterprise, all)
  --template <name>     Template name or ID
  --from-swarm <id>     Create template from existing swarm
  --name <name>         Template name (for create command)
  --output <file>       Output file for export
  --agents <num>        Override max agents
  --strategy <type>     Override strategy
  --verbose, -v         Verbose output

${colors.cyan}Examples:${colors.reset}
  ${colors.yellow}# List all quickstart templates${colors.reset}
  node template-manager.js list --category quickstart

  ${colors.yellow}# View template details${colors.reset}
  node template-manager.js view --template full-stack-dev

  ${colors.yellow}# Deploy from template with overrides${colors.reset}
  node template-manager.js deploy --template code-review --agents 4

  ${colors.yellow}# Create custom template from swarm${colors.reset}
  node template-manager.js create --from-swarm swarm_123 --name my-custom-template

  ${colors.yellow}# Export template to file${colors.reset}
  node template-manager.js export --template full-stack-dev --output template.json
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
 * List templates
 */
async function listTemplates(config) {
  log(`\n${colors.bright}Available Templates${colors.reset}`, 'cyan');

  try {
    const result = executeMCP('swarm_templates_list', {
      category: config.category === 'all' ? undefined : config.category,
      includeStore: true
    });

    if (!result.templates || result.templates.length === 0) {
      log(`  No templates found in category: ${config.category}`, 'yellow');
      return;
    }

    // Group by category
    const byCategory = {};
    result.templates.forEach(template => {
      const cat = template.category || 'uncategorized';
      if (!byCategory[cat]) {
        byCategory[cat] = [];
      }
      byCategory[cat].push(template);
    });

    Object.keys(byCategory).forEach(category => {
      log(`\n  ${colors.bright}${category.toUpperCase()}${colors.reset}`, 'magenta');

      byCategory[category].forEach((template, index) => {
        log(`    ${index + 1}. ${template.name}`, 'bright');
        log(`       ID: ${template.template_id}`, 'blue');
        log(`       ${template.description}`, 'dim');

        if (template.topology) {
          log(`       Topology: ${template.topology} | Agents: ${template.default_agents || 'variable'}`, 'blue');
        }

        if (template.tags && template.tags.length > 0) {
          log(`       Tags: ${template.tags.join(', ')}`, 'dim');
        }

        log('');
      });
    });

    log(`  Total templates: ${result.templates.length}\n`, 'cyan');

    return result;
  } catch (error) {
    log(`  Error listing templates: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * View template details
 */
async function viewTemplate(config) {
  if (!config.template) {
    throw new Error('Template name or ID is required (--template)');
  }

  log(`\n${colors.bright}Template Details${colors.reset}`, 'cyan');

  try {
    const template = executeMCP('template_get', {
      template_name: config.template
    });

    log(`\n  Name: ${template.name}`, 'bright');
    log(`  ID: ${template.template_id}`, 'blue');
    log(`  Category: ${template.category}`, 'blue');
    log(`  Description: ${template.description}`, 'reset');

    if (template.topology) {
      log(`\n  ${colors.bright}Configuration:${colors.reset}`, 'cyan');
      log(`    Topology: ${template.topology}`, 'blue');
      log(`    Default Agents: ${template.default_agents || 'variable'}`, 'blue');
      log(`    Strategy: ${template.strategy || 'balanced'}`, 'blue');
    }

    if (template.agents && template.agents.length > 0) {
      log(`\n  ${colors.bright}Included Agents:${colors.reset}`, 'cyan');

      template.agents.forEach((agent, index) => {
        log(`    ${index + 1}. ${agent.type}: ${agent.name}`, 'blue');

        if (agent.capabilities && agent.capabilities.length > 0) {
          log(`       Capabilities: ${agent.capabilities.join(', ')}`, 'dim');
        }
      });
    }

    if (template.workflows && template.workflows.length > 0) {
      log(`\n  ${colors.bright}Included Workflows:${colors.reset}`, 'cyan');

      template.workflows.forEach((workflow, index) => {
        log(`    ${index + 1}. ${workflow.name}`, 'blue');
        log(`       Steps: ${workflow.steps.length}`, 'dim');
      });
    }

    if (template.tags && template.tags.length > 0) {
      log(`\n  Tags: ${template.tags.join(', ')}`, 'blue');
    }

    if (template.created_at) {
      log(`  Created: ${new Date(template.created_at).toLocaleString()}`, 'dim');
    }

    if (config.verbose) {
      log(`\n${JSON.stringify(template, null, 2)}`, 'reset');
    }

    log('');

    return template;
  } catch (error) {
    log(`  Error viewing template: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * Deploy from template
 */
async function deployTemplate(config) {
  if (!config.template) {
    throw new Error('Template name or ID is required (--template)');
  }

  log(`\n${colors.bright}Deploying from Template${colors.reset}`, 'cyan');
  log(`  Template: ${config.template}`, 'blue');

  if (Object.keys(config.overrides).length > 0) {
    log(`  Overrides:`, 'blue');
    Object.keys(config.overrides).forEach(key => {
      log(`    ${key}: ${config.overrides[key]}`, 'blue');
    });
  }

  try {
    const result = executeMCP('swarm_create_from_template', {
      template_name: config.template,
      overrides: config.overrides
    });

    log(`\n  ✓ Swarm deployed successfully`, 'green');
    log(`  Swarm ID: ${result.swarm_id}`, 'green');
    log(`  Topology: ${result.topology}`, 'blue');
    log(`  Agents: ${result.agents.length}`, 'blue');

    if (config.verbose) {
      log(`\n${JSON.stringify(result, null, 2)}`, 'reset');
    }

    log(`\n  ${colors.cyan}Monitor with:${colors.reset}`);
    log(`  npx flow-nexus@latest swarm status ${result.swarm_id}\n`, 'yellow');

    return result;
  } catch (error) {
    log(`  Error deploying template: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * Create custom template from swarm
 */
async function createTemplate(config) {
  if (!config.swarmId) {
    throw new Error('Source swarm ID is required (--from-swarm)');
  }

  if (!config.name) {
    throw new Error('Template name is required (--name)');
  }

  log(`\n${colors.bright}Creating Custom Template${colors.reset}`, 'cyan');
  log(`  Source Swarm: ${config.swarmId}`, 'blue');
  log(`  Template Name: ${config.name}`, 'blue');

  try {
    // Get swarm configuration
    const swarmStatus = executeMCP('swarm_status', { swarm_id: config.swarmId });

    // Create template object
    const template = {
      name: config.name,
      category: 'custom',
      description: `Custom template created from swarm ${config.swarmId}`,
      topology: swarmStatus.topology,
      strategy: swarmStatus.strategy,
      default_agents: swarmStatus.agents.length,
      agents: swarmStatus.agents.map(agent => ({
        type: agent.type,
        name: agent.name,
        capabilities: agent.capabilities || []
      })),
      created_at: new Date().toISOString()
    };

    log(`\n  Template created with:`, 'blue');
    log(`    Topology: ${template.topology}`, 'blue');
    log(`    Agents: ${template.agents.length}`, 'blue');
    log(`    Strategy: ${template.strategy}`, 'blue');

    // Export to file if output specified
    if (config.output) {
      const outputPath = path.resolve(process.cwd(), config.output);
      fs.writeFileSync(outputPath, JSON.stringify(template, null, 2), 'utf-8');
      log(`\n  ✓ Template saved to: ${outputPath}`, 'green');
    } else {
      log(`\n${JSON.stringify(template, null, 2)}`, 'reset');
    }

    log('');

    return template;
  } catch (error) {
    log(`  Error creating template: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * Export template to file
 */
async function exportTemplate(config) {
  if (!config.template) {
    throw new Error('Template name or ID is required (--template)');
  }

  if (!config.output) {
    throw new Error('Output file is required (--output)');
  }

  log(`\n${colors.bright}Exporting Template${colors.reset}`, 'cyan');

  try {
    const template = executeMCP('template_get', { template_name: config.template });

    const outputPath = path.resolve(process.cwd(), config.output);
    fs.writeFileSync(outputPath, JSON.stringify(template, null, 2), 'utf-8');

    log(`  ✓ Template exported to: ${outputPath}`, 'green');
    log(`  Size: ${(JSON.stringify(template).length / 1024).toFixed(2)} KB\n`, 'blue');

    return template;
  } catch (error) {
    log(`  Error exporting template: ${error.message}`, 'red');
    throw error;
  }
}

/**
 * Main execution
 */
async function main() {
  const config = parseArgs();

  try {
    switch (config.command) {
      case 'list':
        await listTemplates(config);
        break;
      case 'view':
        await viewTemplate(config);
        break;
      case 'deploy':
        await deployTemplate(config);
        break;
      case 'create':
        await createTemplate(config);
        break;
      case 'export':
        await exportTemplate(config);
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

module.exports = { listTemplates, viewTemplate, deployTemplate, createTemplate, exportTemplate };
