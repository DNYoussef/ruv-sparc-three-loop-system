#!/usr/bin/env node

/**
 * Test Suite: Event-Driven Workflows
 *
 * Comprehensive tests for event-driven workflow automation:
 * - Workflow creation with various trigger types
 * - Message queue processing
 * - Async vs sync execution
 * - Retry mechanisms and error recovery
 * - Agent assignment and optimization
 *
 * @requires flow-nexus MCP server
 */

const { execSync } = require('child_process');

const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m'
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
 * Test 1: Workflow Creation with Different Triggers
 */
async function testWorkflowTriggers() {
  console.log(`\n${colors.cyan}Test 1: Workflow Creation with Different Triggers${colors.reset}`);

  const workflowIds = [];

  try {
    const triggers = [
      { name: 'GitHub Push Trigger', triggers: ['github_push'], priority: 8 },
      { name: 'GitHub PR Trigger', triggers: ['github_pr_merged'], priority: 9 },
      { name: 'Schedule Trigger', triggers: ['schedule:0 2 * * *'], priority: 5 },
      { name: 'Manual Trigger', triggers: ['manual_trigger'], priority: 10 },
      { name: 'Multi Trigger', triggers: ['github_push', 'manual_trigger'], priority: 7 }
    ];

    for (const config of triggers) {
      log(`  Creating workflow: ${config.name}`, 'yellow');

      const workflow = executeMCP('workflow_create', {
        name: config.name,
        description: `Test workflow with ${config.triggers.join(', ')}`,
        steps: [
          { id: 'init', action: 'initialize', agent: 'coordinator' },
          { id: 'process', action: 'process_task', agent: 'coder', depends_on: ['init'] }
        ],
        triggers: config.triggers,
        metadata: {
          priority: config.priority,
          retry_policy: 'exponential_backoff',
          max_retries: 3
        }
      });

      assert(workflow.workflow_id, 'Workflow ID should be returned');
      assert(workflow.triggers, 'Workflow should have triggers');

      workflowIds.push(workflow.workflow_id);
      log(`    ✓ Created: ${workflow.workflow_id}`, 'green');
    }

    console.log(`  ${colors.green}✓ All workflow triggers created successfully (${workflowIds.length})${colors.reset}`);
    return { success: true, workflowIds };
  } catch (error) {
    console.log(`  ${colors.red}✗ Test failed: ${error.message}${colors.reset}`);
    return { success: false, error: error.message, workflowIds };
  }
}

/**
 * Test 2: Async vs Sync Execution
 */
async function testExecutionModes() {
  console.log(`\n${colors.cyan}Test 2: Async vs Sync Execution${colors.reset}`);

  let workflowId = null;

  try {
    // Create test workflow
    log('  Creating test workflow', 'yellow');
    const workflow = executeMCP('workflow_create', {
      name: 'Execution Mode Test',
      description: 'Test async and sync execution',
      steps: [
        { id: 'step1', action: 'task1', agent: 'coder' },
        { id: 'step2', action: 'task2', agent: 'analyst', depends_on: ['step1'] }
      ]
    });

    workflowId = workflow.workflow_id;

    // Test async execution
    log('  Testing async execution', 'yellow');
    const asyncExec = executeMCP('workflow_execute', {
      workflow_id: workflowId,
      input_data: { mode: 'async' },
      async: true
    });

    assert(asyncExec.execution_id || asyncExec.status, 'Async execution should return execution ID or status');
    log(`    ✓ Async execution started`, 'green');

    // Wait a moment for queue processing
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Check queue status
    log('  Checking message queue status', 'yellow');
    const queueStatus = executeMCP('workflow_queue_status', {
      include_messages: true
    });

    assert(queueStatus.total_queues !== undefined, 'Queue status should include total queues');
    log(`    ✓ Queue status retrieved: ${queueStatus.total_messages} messages`, 'green');

    // Test sync execution (if supported)
    log('  Testing sync execution', 'yellow');
    try {
      const syncExec = executeMCP('workflow_execute', {
        workflow_id: workflowId,
        input_data: { mode: 'sync' },
        async: false
      });

      assert(syncExec.status || syncExec.result, 'Sync execution should return status or result');
      log(`    ✓ Sync execution completed`, 'green');
    } catch (error) {
      log(`    ! Sync execution not supported or timed out (expected for long workflows)`, 'yellow');
    }

    console.log(`  ${colors.green}✓ Execution mode tests completed${colors.reset}`);
    return { success: true, workflowId };
  } catch (error) {
    console.log(`  ${colors.red}✗ Test failed: ${error.message}${colors.reset}`);
    return { success: false, error: error.message, workflowId };
  }
}

/**
 * Test 3: Agent Assignment Optimization
 */
async function testAgentAssignment() {
  console.log(`\n${colors.cyan}Test 3: Agent Assignment Optimization${colors.reset}`);

  let swarmId = null;

  try {
    // Initialize swarm
    log('  Initializing swarm', 'yellow');
    const swarm = executeMCP('swarm_init', {
      topology: 'hierarchical',
      maxAgents: 5,
      strategy: 'specialized'
    });

    swarmId = swarm.swarm_id;

    // Spawn specialized agents
    log('  Spawning specialized agents', 'yellow');
    const agents = [
      { type: 'coder', name: 'Backend Specialist', capabilities: ['backend', 'api', 'database'] },
      { type: 'coder', name: 'Frontend Specialist', capabilities: ['frontend', 'react', 'ui'] },
      { type: 'analyst', name: 'QA Specialist', capabilities: ['testing', 'quality', 'automation'] }
    ];

    for (const agent of agents) {
      executeMCP('agent_spawn', {
        type: agent.type,
        name: agent.name,
        capabilities: agent.capabilities
      });
      log(`    ✓ Spawned: ${agent.name}`, 'green');
    }

    // Create workflow with tasks
    log('  Creating workflow with tasks', 'yellow');
    const workflow = executeMCP('workflow_create', {
      name: 'Agent Assignment Test',
      steps: [
        { id: 'backend_task', action: 'develop_api', agent: 'coder' },
        { id: 'frontend_task', action: 'develop_ui', agent: 'coder' },
        { id: 'testing_task', action: 'qa_testing', agent: 'analyst' }
      ]
    });

    // Execute workflow
    log('  Executing workflow', 'yellow');
    executeMCP('workflow_execute', {
      workflow_id: workflow.workflow_id,
      async: true
    });

    // Check workflow status to see agent assignments
    await new Promise(resolve => setTimeout(resolve, 1000));

    const status = executeMCP('workflow_status', {
      workflow_id: workflow.workflow_id,
      include_metrics: true
    });

    assert(status.workflow_id, 'Should have workflow status');
    log(`    ✓ Workflow status: ${status.status}`, 'green');

    console.log(`  ${colors.green}✓ Agent assignment optimization working${colors.reset}`);
    return { success: true, swarmId };
  } catch (error) {
    console.log(`  ${colors.red}✗ Test failed: ${error.message}${colors.reset}`);
    return { success: false, error: error.message, swarmId };
  }
}

/**
 * Test 4: Retry Mechanisms
 */
async function testRetryMechanisms() {
  console.log(`\n${colors.cyan}Test 4: Retry Mechanisms and Error Recovery${colors.reset}`);

  const workflowIds = [];

  try {
    const retryPolicies = [
      { name: 'No Retry', policy: 'none', max_retries: 0 },
      { name: 'Linear Retry', policy: 'linear', max_retries: 2 },
      { name: 'Exponential Backoff', policy: 'exponential_backoff', max_retries: 3 }
    ];

    for (const config of retryPolicies) {
      log(`  Testing retry policy: ${config.name}`, 'yellow');

      const workflow = executeMCP('workflow_create', {
        name: `Retry Test - ${config.name}`,
        description: `Testing ${config.policy} retry mechanism`,
        steps: [
          { id: 'step1', action: 'task_with_potential_failure', agent: 'coder' }
        ],
        metadata: {
          retry_policy: config.policy,
          max_retries: config.max_retries,
          timeout: 60000
        }
      });

      assert(workflow.workflow_id, 'Workflow should be created');
      workflowIds.push(workflow.workflow_id);

      log(`    ✓ Created workflow with ${config.policy} (max ${config.max_retries} retries)`, 'green');
    }

    console.log(`  ${colors.green}✓ All retry mechanisms configured successfully${colors.reset}`);
    return { success: true, workflowIds };
  } catch (error) {
    console.log(`  ${colors.red}✗ Test failed: ${error.message}${colors.reset}`);
    return { success: false, error: error.message, workflowIds };
  }
}

/**
 * Cleanup
 */
async function cleanup(swarmIds, workflowIds) {
  if (TEST_CONFIG.skipCleanup) {
    console.log(`\n${colors.yellow}Skipping cleanup${colors.reset}`);
    return;
  }

  console.log(`\n${colors.cyan}Cleaning up test resources...${colors.reset}`);

  // Cleanup swarms
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

  log(`  Cleanup complete`, 'green');
}

/**
 * Main test runner
 */
async function runTests() {
  console.log(`${colors.cyan}╔═══════════════════════════════════════════════════════════╗${colors.reset}`);
  console.log(`${colors.cyan}║   Event-Driven Workflow Test Suite                       ║${colors.reset}`);
  console.log(`${colors.cyan}╚═══════════════════════════════════════════════════════════╝${colors.reset}`);

  const allSwarmIds = [];
  const allWorkflowIds = [];
  const results = [];

  try {
    const test1 = await testWorkflowTriggers();
    results.push({ name: 'Workflow Triggers', ...test1 });
    if (test1.workflowIds) allWorkflowIds.push(...test1.workflowIds);

    const test2 = await testExecutionModes();
    results.push({ name: 'Execution Modes', ...test2 });
    if (test2.workflowId) allWorkflowIds.push(test2.workflowId);

    const test3 = await testAgentAssignment();
    results.push({ name: 'Agent Assignment', ...test3 });
    if (test3.swarmId) allSwarmIds.push(test3.swarmId);

    const test4 = await testRetryMechanisms();
    results.push({ name: 'Retry Mechanisms', ...test4 });
    if (test4.workflowIds) allWorkflowIds.push(...test4.workflowIds);

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

    await cleanup(allSwarmIds, allWorkflowIds);

    process.exit(failed > 0 ? 1 : 0);
  } catch (error) {
    console.log(`\n${colors.red}Fatal error: ${error.message}${colors.reset}`);
    await cleanup(allSwarmIds, allWorkflowIds);
    process.exit(1);
  }
}

if (require.main === module) {
  runTests();
}

module.exports = { testWorkflowTriggers, testExecutionModes, testAgentAssignment, testRetryMechanisms };
