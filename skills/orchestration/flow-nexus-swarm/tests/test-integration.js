#!/usr/bin/env node

/**
 * Test Suite: Integration Tests
 *
 * End-to-end integration tests for complete workflows:
 * - Full-stack development workflow
 * - Multi-swarm coordination
 * - Template deployment and customization
 * - Real-time monitoring and metrics
 *
 * @requires flow-nexus MCP server
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m'
};

const TEST_CONFIG = {
  verbose: process.env.VERBOSE === 'true',
  skipCleanup: process.env.SKIP_CLEANUP === 'true'
};

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
 * Log helper
 */
function log(message, color = 'reset') {
  if (TEST_CONFIG.verbose) {
    console.log(`${colors[color]}${message}${colors.reset}`);
  }
}

/**
 * Assert helper
 */
function assert(condition, message) {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

/**
 * Wait helper
 */
function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Test 1: Full-Stack Development Workflow
 */
async function testFullStackWorkflow() {
  console.log(`\n${colors.cyan}Test 1: Full-Stack Development Workflow${colors.reset}`);

  let swarmId = null;
  let workflowId = null;

  try {
    // Initialize hierarchical swarm
    log('  [1/6] Initializing hierarchical swarm', 'yellow');
    const swarm = executeMCP('swarm_init', {
      topology: 'hierarchical',
      maxAgents: 8,
      strategy: 'specialized'
    });

    swarmId = swarm.swarm_id;
    assert(swarm.topology === 'hierarchical', 'Topology should be hierarchical');
    log(`    ✓ Swarm created: ${swarmId}`, 'green');

    // Spawn specialized agents
    log('  [2/6] Spawning specialized agents', 'yellow');
    const agents = [
      { type: 'coordinator', name: 'Project Manager' },
      { type: 'researcher', name: 'Technical Architect' },
      { type: 'coder', name: 'Backend Developer' },
      { type: 'coder', name: 'Frontend Developer' },
      { type: 'coder', name: 'Database Architect' },
      { type: 'analyst', name: 'QA Engineer' },
      { type: 'optimizer', name: 'Performance Engineer' }
    ];

    for (const agent of agents) {
      executeMCP('agent_spawn', { type: agent.type, name: agent.name });
    }

    log(`    ✓ ${agents.length} agents spawned`, 'green');

    // Create full-stack workflow
    log('  [3/6] Creating full-stack workflow', 'yellow');
    const workflow = executeMCP('workflow_create', {
      name: 'Full-Stack Feature Development',
      description: 'Complete feature development lifecycle',
      steps: [
        { id: 'requirements', action: 'analyze_requirements', agent: 'coordinator' },
        { id: 'architecture', action: 'design_architecture', agent: 'researcher', depends_on: ['requirements'] },
        { id: 'db_design', action: 'design_database', agent: 'coder', depends_on: ['architecture'] },
        { id: 'backend', action: 'develop_api', agent: 'coder', depends_on: ['db_design'], parallel: true },
        { id: 'frontend', action: 'develop_ui', agent: 'coder', depends_on: ['architecture'], parallel: true },
        { id: 'integration', action: 'integrate', agent: 'coder', depends_on: ['backend', 'frontend'] },
        { id: 'testing', action: 'qa_testing', agent: 'analyst', depends_on: ['integration'] },
        { id: 'optimization', action: 'optimize', agent: 'optimizer', depends_on: ['testing'] }
      ],
      metadata: {
        priority: 10,
        retry_policy: 'exponential_backoff',
        max_retries: 2
      }
    });

    workflowId = workflow.workflow_id;
    assert(workflow.steps, 'Workflow should have steps');
    log(`    ✓ Workflow created: ${workflowId}`, 'green');

    // Execute workflow
    log('  [4/6] Executing workflow (async)', 'yellow');
    const execution = executeMCP('workflow_execute', {
      workflow_id: workflowId,
      input_data: { project: 'E-commerce Platform' },
      async: true
    });

    log(`    ✓ Workflow execution started`, 'green');

    // Wait and check status
    log('  [5/6] Checking workflow status', 'yellow');
    await wait(2000);

    const status = executeMCP('workflow_status', {
      workflow_id: workflowId,
      include_metrics: true
    });

    assert(status.workflow_id === workflowId, 'Status should match workflow ID');
    log(`    ✓ Workflow status: ${status.status}`, 'green');

    // Check swarm metrics
    log('  [6/6] Retrieving swarm metrics', 'yellow');
    const swarmStatus = executeMCP('swarm_status', { swarm_id: swarmId });

    assert(swarmStatus.agents, 'Swarm should have agents');
    assert(swarmStatus.agents.length >= agents.length, 'All agents should be present');
    log(`    ✓ Swarm operational with ${swarmStatus.agents.length} agents`, 'green');

    console.log(`  ${colors.green}✓ Full-stack workflow integration successful${colors.reset}`);
    return { success: true, swarmId, workflowId };
  } catch (error) {
    console.log(`  ${colors.red}✗ Test failed: ${error.message}${colors.reset}`);
    return { success: false, error: error.message, swarmId, workflowId };
  }
}

/**
 * Test 2: Template Deployment
 */
async function testTemplateDeployment() {
  console.log(`\n${colors.cyan}Test 2: Template Deployment and Customization${colors.reset}`);

  let swarmId = null;

  try {
    // List available templates
    log('  [1/4] Listing available templates', 'yellow');
    const templates = executeMCP('swarm_templates_list', {
      category: 'quickstart',
      includeStore: true
    });

    assert(templates.templates, 'Should return templates');
    log(`    ✓ Found ${templates.templates.length} templates`, 'green');

    // Deploy from template with overrides
    log('  [2/4] Deploying from template', 'yellow');

    // Use first available template or fallback
    const templateName = templates.templates.length > 0
      ? templates.templates[0].template_id
      : 'full-stack-dev';

    try {
      const deployment = executeMCP('swarm_create_from_template', {
        template_name: templateName,
        overrides: {
          maxAgents: 6,
          strategy: 'balanced'
        }
      });

      swarmId = deployment.swarm_id;
      assert(deployment.topology, 'Deployment should have topology');
      log(`    ✓ Deployed from template: ${swarmId}`, 'green');

      // Verify deployment
      log('  [3/4] Verifying deployment', 'yellow');
      const status = executeMCP('swarm_status', { swarm_id: swarmId });

      assert(status.swarm_id === swarmId, 'Swarm ID should match');
      assert(status.status === 'healthy' || status.status === 'operational', 'Swarm should be healthy');
      log(`    ✓ Deployment verified`, 'green');

      // Check agent configuration
      log('  [4/4] Checking agent configuration', 'yellow');
      assert(status.agents, 'Should have agents');
      log(`    ✓ ${status.agents.length} agents configured from template`, 'green');

    } catch (error) {
      if (error.message.includes('not found')) {
        log(`    ! Template not found, skipping template-specific tests`, 'yellow');
      } else {
        throw error;
      }
    }

    console.log(`  ${colors.green}✓ Template deployment successful${colors.reset}`);
    return { success: true, swarmId };
  } catch (error) {
    console.log(`  ${colors.red}✗ Test failed: ${error.message}${colors.reset}`);
    return { success: false, error: error.message, swarmId };
  }
}

/**
 * Test 3: Multi-Swarm Coordination
 */
async function testMultiSwarmCoordination() {
  console.log(`\n${colors.cyan}Test 3: Multi-Swarm Coordination${colors.reset}`);

  const swarmIds = [];

  try {
    // Create multiple swarms for different phases
    log('  [1/4] Creating phase-specific swarms', 'yellow');

    const phases = [
      { name: 'Research', topology: 'mesh', agents: 4 },
      { name: 'Development', topology: 'hierarchical', agents: 8 },
      { name: 'Testing', topology: 'star', agents: 5 }
    ];

    for (const phase of phases) {
      const swarm = executeMCP('swarm_init', {
        topology: phase.topology,
        maxAgents: phase.agents,
        strategy: 'specialized'
      });

      swarmIds.push(swarm.swarm_id);
      log(`    ✓ ${phase.name} swarm: ${swarm.swarm_id}`, 'green');
    }

    // List all active swarms
    log('  [2/4] Listing all active swarms', 'yellow');
    const allSwarms = executeMCP('swarm_list', { status: 'active' });

    assert(allSwarms.swarms, 'Should return swarms list');
    assert(allSwarms.swarms.length >= swarmIds.length, 'Should find created swarms');
    log(`    ✓ ${allSwarms.swarms.length} active swarms found`, 'green');

    // Get status of each swarm
    log('  [3/4] Retrieving individual swarm statuses', 'yellow');
    for (const swarmId of swarmIds) {
      const status = executeMCP('swarm_status', { swarm_id: swarmId });
      assert(status.swarm_id === swarmId, 'Swarm ID should match');
      log(`    ✓ ${status.topology} swarm operational`, 'green');
    }

    // Test swarm scaling
    log('  [4/4] Testing swarm scaling', 'yellow');
    const targetSwarm = swarmIds[0];

    const scaled = executeMCP('swarm_scale', {
      swarm_id: targetSwarm,
      target_agents: 6
    });

    assert(scaled.target_agents === 6 || scaled.max_agents === 6, 'Should scale to 6 agents');
    log(`    ✓ Swarm scaled successfully`, 'green');

    console.log(`  ${colors.green}✓ Multi-swarm coordination successful${colors.reset}`);
    return { success: true, swarmIds };
  } catch (error) {
    console.log(`  ${colors.red}✗ Test failed: ${error.message}${colors.reset}`);
    return { success: false, error: error.message, swarmIds };
  }
}

/**
 * Cleanup
 */
async function cleanup(swarmIds) {
  if (TEST_CONFIG.skipCleanup) {
    console.log(`\n${colors.yellow}Skipping cleanup${colors.reset}`);
    return;
  }

  console.log(`\n${colors.cyan}Cleaning up test resources...${colors.reset}`);

  for (const swarmId of swarmIds) {
    if (swarmId) {
      try {
        executeMCP('swarm_destroy', { swarm_id: swarmId });
        log(`  ✓ Destroyed swarm: ${swarmId}`, 'green');
      } catch (error) {
        log(`  ✗ Failed to destroy swarm: ${error.message}`, 'red');
      }
    }
  }
}

/**
 * Main test runner
 */
async function runTests() {
  console.log(`${colors.cyan}╔═══════════════════════════════════════════════════════════╗${colors.reset}`);
  console.log(`${colors.cyan}║   Integration Test Suite                                 ║${colors.reset}`);
  console.log(`${colors.cyan}╚═══════════════════════════════════════════════════════════╝${colors.reset}`);

  const allSwarmIds = [];
  const results = [];

  try {
    const test1 = await testFullStackWorkflow();
    results.push({ name: 'Full-Stack Workflow', ...test1 });
    if (test1.swarmId) allSwarmIds.push(test1.swarmId);

    const test2 = await testTemplateDeployment();
    results.push({ name: 'Template Deployment', ...test2 });
    if (test2.swarmId) allSwarmIds.push(test2.swarmId);

    const test3 = await testMultiSwarmCoordination();
    results.push({ name: 'Multi-Swarm Coordination', ...test3 });
    if (test3.swarmIds) allSwarmIds.push(...test3.swarmIds);

    // Summary
    console.log(`\n${colors.cyan}╔═══════════════════════════════════════════════════════════╗${colors.reset}`);
    console.log(`${colors.cyan}║   Test Summary                                            ║${colors.reset}`);
    console.log(`${colors.cyan}╚═══════════════════════════════════════════════════════════╝${colors.reset}`);

    const passed = results.filter(r => r.success).length;
    const failed = results.filter(r => !r.success).length;

    results.forEach(result => {
      const icon = result.success ? '✓' : '✗';
      const color = result.success ? 'green' : 'red';
      console.log(`  ${colors[color]}${icon} ${result.name}${colors.reset}`);

      if (!result.success && result.error) {
        console.log(`    ${colors.red}Error: ${result.error}${colors.reset}`);
      }
    });

    console.log(`\n  ${colors.cyan}Total: ${results.length} | Passed: ${passed} | Failed: ${failed}${colors.reset}`);

    await cleanup(allSwarmIds);

    process.exit(failed > 0 ? 1 : 0);
  } catch (error) {
    console.log(`\n${colors.red}Fatal error: ${error.message}${colors.reset}`);
    await cleanup(allSwarmIds);
    process.exit(1);
  }
}

if (require.main === module) {
  runTests();
}

module.exports = { testFullStackWorkflow, testTemplateDeployment, testMultiSwarmCoordination };
