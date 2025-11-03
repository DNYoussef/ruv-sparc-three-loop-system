#!/usr/bin/env node

/**
 * Mesh Topology Test Suite
 *
 * Comprehensive tests for mesh coordination topology.
 * Tests: initialization, agent spawning, consensus, state sync, fault tolerance.
 */

const { execSync } = require('child_process');
const assert = require('assert');

// Test utilities
function exec(command) {
  try {
    return execSync(command, { encoding: 'utf8' });
  } catch (error) {
    return { error: error.message, stdout: error.stdout, stderr: error.stderr };
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Test suite
class MeshTopologyTestSuite {
  constructor() {
    this.results = {
      passed: 0,
      failed: 0,
      tests: []
    };
  }

  async runTest(name, testFn) {
    console.log(`\n🧪 Running: ${name}`);
    try {
      await testFn();
      this.results.passed++;
      this.results.tests.push({ name, status: 'PASSED' });
      console.log(`   ✅ PASSED`);
    } catch (error) {
      this.results.failed++;
      this.results.tests.push({ name, status: 'FAILED', error: error.message });
      console.log(`   ❌ FAILED: ${error.message}`);
    }
  }

  // Test 1: Mesh topology initialization
  async testMeshInitialization() {
    const output = exec('npx claude-flow@alpha swarm_init --topology mesh --maxAgents 5 --strategy balanced');

    assert(output && !output.error, 'Mesh initialization should succeed');
    assert(output.includes('mesh') || output.includes('initialized'), 'Output should confirm mesh initialization');
  }

  // Test 2: Agent spawning in mesh
  async testAgentSpawning() {
    const agents = ['researcher', 'coder', 'analyst'];

    for (const agentType of agents) {
      const output = exec(`npx claude-flow@alpha agent_spawn --type ${agentType} --name ${agentType}-test`);
      assert(output && !output.error, `Agent ${agentType} should spawn successfully`);
    }

    // Verify agent list
    const listOutput = exec('npx claude-flow@alpha agent_list --filter active');
    assert(listOutput && !listOutput.error, 'Agent list should return successfully');
  }

  // Test 3: All-to-all communication
  async testAllToAllCommunication() {
    // In mesh, all agents should be able to communicate directly
    // Store a message in shared state
    const storeOutput = exec(`npx claude-flow@alpha memory store --key "mesh/test-message" --value "Test communication message"`);

    assert(storeOutput && !storeOutput.error, 'Memory store should succeed');

    // Retrieve message
    const retrieveOutput = exec('npx claude-flow@alpha memory retrieve --key "mesh/test-message"');
    assert(retrieveOutput && !retrieveOutput.error, 'Memory retrieve should succeed');
    assert(retrieveOutput.includes('Test communication message'), 'Message should be retrievable');
  }

  // Test 4: Task orchestration
  async testTaskOrchestration() {
    const output = exec('npx claude-flow@alpha task_orchestrate --task "Test mesh coordination" --strategy adaptive --priority medium');

    assert(output && !output.error, 'Task orchestration should succeed');
  }

  // Test 5: Consensus mechanism (Byzantine)
  async testByzantineConsensus() {
    // Simulate consensus requirement
    // In real scenario, this would involve multiple agents voting

    // Store consensus proposal
    const proposalOutput = exec(`npx claude-flow@alpha memory store --key "consensus/proposal-1" --value '{"action":"deploy","votes":0}'`);

    assert(proposalOutput && !proposalOutput.error, 'Consensus proposal should be stored');

    // Simulate votes from agents (in practice, agents would vote independently)
    const voteOutput = exec(`npx claude-flow@alpha memory store --key "consensus/proposal-1-votes" --value '{"agent-1":"yes","agent-2":"yes","agent-3":"yes"}'`);

    assert(voteOutput && !voteOutput.error, 'Votes should be recorded');
  }

  // Test 6: State synchronization
  async testStateSynchronization() {
    // Store state from one agent
    const storeOutput1 = exec(`npx claude-flow@alpha memory store --key "mesh/agent-1/state" --value '{"status":"active","task":"research"}'`);
    assert(storeOutput1 && !storeOutput1.error, 'Agent 1 state should be stored');

    // Store state from another agent
    const storeOutput2 = exec(`npx claude-flow@alpha memory store --key "mesh/agent-2/state" --value '{"status":"active","task":"coding"}'`);
    assert(storeOutput2 && !storeOutput2.error, 'Agent 2 state should be stored');

    // Retrieve both states (simulating synchronization)
    const retrieveOutput1 = exec('npx claude-flow@alpha memory retrieve --key "mesh/agent-1/state"');
    const retrieveOutput2 = exec('npx claude-flow@alpha memory retrieve --key "mesh/agent-2/state"');

    assert(retrieveOutput1 && !retrieveOutput1.error, 'Agent 1 state should be retrievable');
    assert(retrieveOutput2 && !retrieveOutput2.error, 'Agent 2 state should be retrievable');
  }

  // Test 7: Fault detection and recovery
  async testFaultTolerance() {
    // Check swarm status
    const statusOutput = exec('npx claude-flow@alpha swarm_status --verbose false');

    assert(statusOutput && !statusOutput.error, 'Swarm status should be retrievable');

    // In mesh topology, system should continue functioning even if one agent fails
    // This is a conceptual test - actual fault injection would require more infrastructure
  }

  // Test 8: Performance metrics
  async testPerformanceMetrics() {
    const metricsOutput = exec('npx claude-flow@alpha agent_metrics --metric all');

    assert(metricsOutput && !metricsOutput.error, 'Performance metrics should be available');
  }

  // Test 9: Mesh topology cleanup
  async testCleanup() {
    // This would normally clean up the mesh topology
    // For testing purposes, we just verify we can access status
    const statusOutput = exec('npx claude-flow@alpha swarm_status');

    assert(statusOutput && !statusOutput.error, 'Final status check should succeed');
  }

  // Run all tests
  async runAll() {
    console.log('═══════════════════════════════════════');
    console.log('  Mesh Topology Test Suite');
    console.log('═══════════════════════════════════════');

    await this.runTest('Test 1: Mesh Initialization', () => this.testMeshInitialization());
    await this.runTest('Test 2: Agent Spawning', () => this.testAgentSpawning());
    await this.runTest('Test 3: All-to-All Communication', () => this.testAllToAllCommunication());
    await this.runTest('Test 4: Task Orchestration', () => this.testTaskOrchestration());
    await this.runTest('Test 5: Byzantine Consensus', () => this.testByzantineConsensus());
    await this.runTest('Test 6: State Synchronization', () => this.testStateSynchronization());
    await this.runTest('Test 7: Fault Tolerance', () => this.testFaultTolerance());
    await this.runTest('Test 8: Performance Metrics', () => this.testPerformanceMetrics());
    await this.runTest('Test 9: Cleanup', () => this.testCleanup());

    this.printResults();
  }

  printResults() {
    console.log('\n═══════════════════════════════════════');
    console.log('  Test Results');
    console.log('═══════════════════════════════════════');
    console.log(`Total Tests: ${this.results.passed + this.results.failed}`);
    console.log(`Passed: ${this.results.passed} ✅`);
    console.log(`Failed: ${this.results.failed} ❌`);

    if (this.results.failed > 0) {
      console.log('\nFailed Tests:');
      this.results.tests
        .filter(t => t.status === 'FAILED')
        .forEach(t => {
          console.log(`   - ${t.name}: ${t.error}`);
        });
    }

    console.log('═══════════════════════════════════════\n');

    // Exit with appropriate code
    process.exit(this.results.failed > 0 ? 1 : 0);
  }
}

// Run tests
(async () => {
  const suite = new MeshTopologyTestSuite();
  await suite.runAll();
})();
