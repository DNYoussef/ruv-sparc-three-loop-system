#!/usr/bin/env node

/**
 * Test Suite: Swarm Deployment
 *
 * Comprehensive tests for cloud swarm deployment including:
 * - Swarm initialization with different topologies
 * - Agent spawning and configuration
 * - Workflow deployment and execution
 * - Error handling and recovery
 *
 * @requires flow-nexus MCP server
 * @requires jest or node:test
 */

const { execSync } = require('child_process');

// Test configuration
const TEST_CONFIG = {
  verbose: process.env.VERBOSE === 'true',
  skipCleanup: process.env.SKIP_CLEANUP === 'true',
  timeout: 30000
};

// ANSI colors for test output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m'
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
 * Log test message
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
 * Test 1: Swarm Initialization with Different Topologies
 */
async function testSwarmTopologies() {
  console.log(`\n${colors.cyan}Test 1: Swarm Initialization with Different Topologies${colors.reset}`);

  const topologies = ['hierarchical', 'mesh', 'ring', 'star'];
  const swarmIds = [];

  try {
    for (const topology of topologies) {
      log(`  Testing topology: ${topology}`, 'yellow');

      const result = executeMCP('swarm_init', {
        topology: topology,
        maxAgents: 5,
        strategy: 'balanced'
      });

      assert(result.swarm_id, 'Swarm ID should be returned');
      assert(result.topology === topology, `Topology should be ${topology}`);
      assert(result.status === 'healthy' || result.status === 'operational', 'Swarm should be healthy');

      swarmIds.push(result.swarm_id);
      log(`    ✓ ${topology} swarm created: ${result.swarm_id}`, 'green');
    }

    console.log(`  ${colors.green}✓ All topologies initialized successfully${colors.reset}`);
    return { success: true, swarmIds };
  } catch (error) {
    console.log(`  ${colors.red}✗ Test failed: ${error.message}${colors.reset}`);
    return { success: false, error: error.message, swarmIds };
  }
}

/**
 * Test 2: Agent Spawning and Configuration
 */
async function testAgentSpawning() {
  console.log(`\n${colors.cyan}Test 2: Agent Spawning and Configuration${colors.reset}`);

  let swarmId = null;

  try {
    // Initialize test swarm
    log('  Initializing test swarm', 'yellow');
    const swarm = executeMCP('swarm_init', {
      topology: 'hierarchical',
      maxAgents: 6,
      strategy: 'specialized'
    });

    swarmId = swarm.swarm_id;
    log(`    Swarm created: ${swarmId}`, 'yellow');

    // Define agents to spawn
    const agentConfigs = [
      { type: 'coordinator', name: 'Test Coordinator', capabilities: ['coordination', 'delegation'] },
      { type: 'researcher', name: 'Test Researcher', capabilities: ['research', 'analysis'] },
      { type: 'coder', name: 'Test Coder', capabilities: ['coding', 'refactoring'] },
      { type: 'analyst', name: 'Test Analyst', capabilities: ['testing', 'quality_assurance'] }
    ];

    const spawnedAgents = [];

    for (const config of agentConfigs) {
      log(`  Spawning ${config.type}: ${config.name}`, 'yellow');

      const result = executeMCP('agent_spawn', {
        type: config.type,
        name: config.name,
        capabilities: config.capabilities
      });

      assert(result.agent_id, 'Agent ID should be returned');
      assert(result.type === config.type, `Agent type should be ${config.type}`);

      spawnedAgents.push(result);
      log(`    ✓ Agent spawned: ${result.agent_id}`, 'green');
    }

    // Verify swarm status
    const status = executeMCP('swarm_status', { swarm_id: swarmId });

    assert(status.agents, 'Swarm status should include agents');
    assert(status.agents.length >= agentConfigs.length, 'All agents should be spawned');

    console.log(`  ${colors.green}✓ All agents spawned successfully (${spawnedAgents.length}/${agentConfigs.length})${colors.reset}`);
    return { success: true, swarmId, agents: spawnedAgents };
  } catch (error) {
    console.log(`  ${colors.red}✗ Test failed: ${error.message}${colors.reset}`);
    return { success: false, error: error.message, swarmId };
  }
}

/**
 * Test 3: Workflow Deployment and Execution
 */
async function testWorkflowDeployment() {
  console.log(`\n${colors.cyan}Test 3: Workflow Deployment and Execution${colors.reset}`);

  let swarmId = null;
  let workflowId = null;

  try {
    // Initialize test swarm
    log('  Initializing test swarm', 'yellow');
    const swarm = executeMCP('swarm_init', {
      topology: 'star',
      maxAgents: 4,
      strategy: 'balanced'
    });

    swarmId = swarm.swarm_id;

    // Spawn required agents
    log('  Spawning agents', 'yellow');
    executeMCP('agent_spawn', { type: 'coordinator', name: 'Workflow Coordinator' });
    executeMCP('agent_spawn', { type: 'coder', name: 'Implementation Agent' });
    executeMCP('agent_spawn', { type: 'analyst', name: 'Testing Agent' });

    // Create test workflow
    log('  Creating test workflow', 'yellow');
    const workflow = executeMCP('workflow_create', {
      name: 'Test Workflow',
      description: 'Test workflow for validation',
      steps: [
        { id: 'step1', action: 'analyze', agent: 'coordinator' },
        { id: 'step2', action: 'implement', agent: 'coder', depends_on: ['step1'] },
        { id: 'step3', action: 'test', agent: 'analyst', depends_on: ['step2'] }
      ],
      triggers: ['manual_trigger'],
      metadata: {
        priority: 5,
        retry_policy: 'linear',
        max_retries: 2
      }
    });

    assert(workflow.workflow_id, 'Workflow ID should be returned');
    workflowId = workflow.workflow_id;
    log(`    ✓ Workflow created: ${workflowId}`, 'green');

    // Execute workflow (async mode for testing)
    log('  Executing workflow (async)', 'yellow');
    const execution = executeMCP('workflow_execute', {
      workflow_id: workflowId,
      input_data: { test_param: 'test_value' },
      async: true
    });

    assert(execution.execution_id || execution.status, 'Execution should start successfully');
    log(`    ✓ Workflow execution started`, 'green');

    // Check workflow status
    log('  Checking workflow status', 'yellow');
    const status = executeMCP('workflow_status', {
      workflow_id: workflowId,
      include_metrics: true
    });

    assert(status.workflow_id === workflowId, 'Status should match workflow ID');
    assert(status.status, 'Workflow should have a status');
    log(`    ✓ Workflow status: ${status.status}`, 'green');

    console.log(`  ${colors.green}✓ Workflow deployment and execution successful${colors.reset}`);
    return { success: true, swarmId, workflowId };
  } catch (error) {
    console.log(`  ${colors.red}✗ Test failed: ${error.message}${colors.reset}`);
    return { success: false, error: error.message, swarmId, workflowId };
  }
}

/**
 * Test 4: Error Handling and Recovery
 */
async function testErrorHandling() {
  console.log(`\n${colors.cyan}Test 4: Error Handling and Recovery${colors.reset}`);

  const errors = [];

  try {
    // Test 1: Invalid topology
    log('  Testing invalid topology error handling', 'yellow');
    try {
      executeMCP('swarm_init', { topology: 'invalid_topology', maxAgents: 5 });
      errors.push('Should have thrown error for invalid topology');
    } catch (error) {
      log(`    ✓ Invalid topology rejected correctly`, 'green');
    }

    // Test 2: Invalid agent type
    log('  Testing invalid agent type error handling', 'yellow');
    try {
      // First create a valid swarm
      const swarm = executeMCP('swarm_init', { topology: 'star', maxAgents: 3 });

      // Try to spawn invalid agent
      executeMCP('agent_spawn', { type: 'invalid_agent_type', name: 'Invalid Agent' });
      errors.push('Should have thrown error for invalid agent type');
    } catch (error) {
      log(`    ✓ Invalid agent type rejected correctly`, 'green');
    }

    // Test 3: Workflow with missing dependencies
    log('  Testing workflow validation error handling', 'yellow');
    try {
      executeMCP('workflow_create', {
        name: 'Invalid Workflow',
        steps: [
          { id: 'step1', action: 'test', depends_on: ['nonexistent_step'] }
        ]
      });
      errors.push('Should have thrown error for invalid workflow dependencies');
    } catch (error) {
      log(`    ✓ Invalid workflow dependencies rejected correctly`, 'green');
    }

    if (errors.length > 0) {
      throw new Error(`Error handling tests failed: ${errors.join(', ')}`);
    }

    console.log(`  ${colors.green}✓ Error handling and recovery working correctly${colors.reset}`);
    return { success: true };
  } catch (error) {
    console.log(`  ${colors.red}✗ Test failed: ${error.message}${colors.reset}`);
    return { success: false, error: error.message };
  }
}

/**
 * Cleanup test resources
 */
async function cleanup(swarmIds) {
  if (TEST_CONFIG.skipCleanup) {
    console.log(`\n${colors.yellow}Skipping cleanup (SKIP_CLEANUP=true)${colors.reset}`);
    return;
  }

  console.log(`\n${colors.cyan}Cleaning up test resources...${colors.reset}`);

  for (const swarmId of swarmIds) {
    try {
      executeMCP('swarm_destroy', { swarm_id: swarmId });
      log(`  ✓ Destroyed swarm: ${swarmId}`, 'green');
    } catch (error) {
      log(`  ✗ Failed to destroy swarm ${swarmId}: ${error.message}`, 'red');
    }
  }
}

/**
 * Main test runner
 */
async function runTests() {
  console.log(`${colors.cyan}╔═══════════════════════════════════════════════════════════╗${colors.reset}`);
  console.log(`${colors.cyan}║   Flow Nexus Swarm Deployment Test Suite                 ║${colors.reset}`);
  console.log(`${colors.cyan}╚═══════════════════════════════════════════════════════════╝${colors.reset}`);

  const allSwarmIds = [];
  const results = [];

  try {
    // Run tests
    const test1 = await testSwarmTopologies();
    results.push({ name: 'Swarm Topologies', ...test1 });
    if (test1.swarmIds) allSwarmIds.push(...test1.swarmIds);

    const test2 = await testAgentSpawning();
    results.push({ name: 'Agent Spawning', ...test2 });
    if (test2.swarmId) allSwarmIds.push(test2.swarmId);

    const test3 = await testWorkflowDeployment();
    results.push({ name: 'Workflow Deployment', ...test3 });
    if (test3.swarmId) allSwarmIds.push(test3.swarmId);

    const test4 = await testErrorHandling();
    results.push({ name: 'Error Handling', ...test4 });

    // Print summary
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

    // Cleanup
    await cleanup(allSwarmIds);

    // Exit with appropriate code
    process.exit(failed > 0 ? 1 : 0);
  } catch (error) {
    console.log(`\n${colors.red}Fatal error: ${error.message}${colors.reset}`);
    await cleanup(allSwarmIds);
    process.exit(1);
  }
}

// Run tests
if (require.main === module) {
  runTests();
}

module.exports = { testSwarmTopologies, testAgentSpawning, testWorkflowDeployment, testErrorHandling };
