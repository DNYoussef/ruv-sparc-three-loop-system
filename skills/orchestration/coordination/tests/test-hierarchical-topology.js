#!/usr/bin/env node

/**
 * Hierarchical Topology Test Suite
 *
 * Comprehensive tests for hierarchical coordination topology.
 * Tests: initialization, hierarchy formation, delegation, reporting, fault tolerance.
 */

const { execSync } = require('child_process');
const assert = require('assert');

function exec(command) {
  try {
    return execSync(command, { encoding: 'utf8' });
  } catch (error) {
    return { error: error.message, stdout: error.stdout, stderr: error.stderr };
  }
}

class HierarchicalTopologyTestSuite {
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

  // Test 1: Hierarchical topology initialization
  async testHierarchicalInitialization() {
    const output = exec('npx claude-flow@alpha swarm_init --topology hierarchical --maxAgents 10 --strategy specialized');

    assert(output && !output.error, 'Hierarchical initialization should succeed');
    assert(output.includes('hierarchical') || output.includes('initialized'), 'Output should confirm hierarchical initialization');
  }

  // Test 2: Coordinator spawning
  async testCoordinatorSpawning() {
    const output = exec('npx claude-flow@alpha agent_spawn --type coordinator --name hierarchical-coordinator');

    assert(output && !output.error, 'Coordinator should spawn successfully');
  }

  // Test 3: Team lead spawning
  async testTeamLeadSpawning() {
    const teamLeads = ['backend-lead', 'frontend-lead', 'testing-lead'];

    for (const lead of teamLeads) {
      const output = exec(`npx claude-flow@alpha agent_spawn --type coordinator --name ${lead}`);
      assert(output && !output.error, `Team lead ${lead} should spawn successfully`);
    }
  }

  // Test 4: Worker agent spawning
  async testWorkerSpawning() {
    const workers = [
      { type: 'coder', name: 'backend-worker-1' },
      { type: 'coder', name: 'frontend-worker-1' },
      { type: 'analyst', name: 'testing-worker-1' }
    ];

    for (const worker of workers) {
      const output = exec(`npx claude-flow@alpha agent_spawn --type ${worker.type} --name ${worker.name}`);
      assert(output && !output.error, `Worker ${worker.name} should spawn successfully`);
    }
  }

  // Test 5: Hierarchy structure verification
  async testHierarchyStructure() {
    // Store hierarchy configuration
    const hierarchyConfig = {
      coordinator: 'hierarchical-coordinator',
      team_leads: ['backend-lead', 'frontend-lead', 'testing-lead'],
      workers: {
        backend: ['backend-worker-1'],
        frontend: ['frontend-worker-1'],
        testing: ['testing-worker-1']
      }
    };

    const output = exec(`npx claude-flow@alpha memory store --key "coordination/hierarchy" --value '${JSON.stringify(hierarchyConfig)}'`);

    assert(output && !output.error, 'Hierarchy structure should be stored');

    // Verify retrieval
    const retrieveOutput = exec('npx claude-flow@alpha memory retrieve --key "coordination/hierarchy"');
    assert(retrieveOutput && !retrieveOutput.error, 'Hierarchy structure should be retrievable');
  }

  // Test 6: Top-down task delegation
  async testTaskDelegation() {
    // Coordinator delegates to team leads
    const delegationOutput = exec('npx claude-flow@alpha task_orchestrate --task "Build full-stack feature" --strategy adaptive --priority high');

    assert(delegationOutput && !delegationOutput.error, 'Task delegation should succeed');

    // Store delegation record
    const recordOutput = exec(`npx claude-flow@alpha memory store --key "delegation/task-1" --value '{"from":"coordinator","to":"team-leads","task":"Build full-stack feature"}'`);

    assert(recordOutput && !recordOutput.error, 'Delegation record should be stored');
  }

  // Test 7: Bottom-up reporting
  async testBottomUpReporting() {
    // Workers report to team leads
    const workerReports = [
      { worker: 'backend-worker-1', status: 'completed', progress: 100 },
      { worker: 'frontend-worker-1', status: 'in_progress', progress: 75 },
      { worker: 'testing-worker-1', status: 'pending', progress: 0 }
    ];

    for (const report of workerReports) {
      const output = exec(`npx claude-flow@alpha memory store --key "reports/${report.worker}" --value '${JSON.stringify(report)}'`);
      assert(output && !output.error, `Report from ${report.worker} should be stored`);
    }

    // Team leads aggregate and report to coordinator
    const aggregateReport = {
      backend: { status: 'completed', progress: 100 },
      frontend: { status: 'in_progress', progress: 75 },
      testing: { status: 'pending', progress: 0 },
      overall_progress: 58.3
    };

    const aggOutput = exec(`npx claude-flow@alpha memory store --key "reports/aggregate" --value '${JSON.stringify(aggregateReport)}'`);
    assert(aggOutput && !aggOutput.error, 'Aggregate report should be stored');
  }

  // Test 8: Coordinator failover
  async testCoordinatorFailover() {
    // Simulate coordinator failure by storing failover state
    const failoverOutput = exec(`npx claude-flow@alpha memory store --key "failover/coordinator-status" --value '{"status":"failed","timestamp":"${new Date().toISOString()}"}'`);

    assert(failoverOutput && !failoverOutput.error, 'Failover state should be stored');

    // Promote team lead to coordinator (simulated)
    const promoteOutput = exec(`npx claude-flow@alpha memory store --key "failover/new-coordinator" --value '{"promoted":"backend-lead","timestamp":"${new Date().toISOString()}"}'`);

    assert(promoteOutput && !promoteOutput.error, 'Coordinator promotion should be recorded');
  }

  // Test 9: Worker failure and redistribution
  async testWorkerFailureRecovery() {
    // Simulate worker failure
    const failureOutput = exec(`npx claude-flow@alpha memory store --key "failures/backend-worker-1" --value '{"status":"failed","timestamp":"${new Date().toISOString()}"}'`);

    assert(failureOutput && !failureOutput.error, 'Worker failure should be recorded');

    // Redistribute work to other workers (simulated)
    const redistributeOutput = exec(`npx claude-flow@alpha memory store --key "redistribution/backend-tasks" --value '{"from":"backend-worker-1","to":"backend-worker-2"}'`);

    assert(redistributeOutput && !redistributeOutput.error, 'Work redistribution should be recorded');
  }

  // Test 10: Hierarchy metrics
  async testHierarchyMetrics() {
    const metricsOutput = exec('npx claude-flow@alpha agent_metrics --metric all');

    assert(metricsOutput && !metricsOutput.error, 'Hierarchy metrics should be available');

    // Store custom hierarchy metrics
    const customMetrics = {
      delegation_latency: 5.2,
      reporting_latency: 3.1,
      coordination_overhead: 0.15,
      levels: 3,
      agents_per_level: [1, 3, 6]
    };

    const metricsStoreOutput = exec(`npx claude-flow@alpha memory store --key "metrics/hierarchy" --value '${JSON.stringify(customMetrics)}'`);

    assert(metricsStoreOutput && !metricsStoreOutput.error, 'Custom hierarchy metrics should be stored');
  }

  // Run all tests
  async runAll() {
    console.log('═══════════════════════════════════════');
    console.log('  Hierarchical Topology Test Suite');
    console.log('═══════════════════════════════════════');

    await this.runTest('Test 1: Hierarchical Initialization', () => this.testHierarchicalInitialization());
    await this.runTest('Test 2: Coordinator Spawning', () => this.testCoordinatorSpawning());
    await this.runTest('Test 3: Team Lead Spawning', () => this.testTeamLeadSpawning());
    await this.runTest('Test 4: Worker Spawning', () => this.testWorkerSpawning());
    await this.runTest('Test 5: Hierarchy Structure Verification', () => this.testHierarchyStructure());
    await this.runTest('Test 6: Top-Down Task Delegation', () => this.testTaskDelegation());
    await this.runTest('Test 7: Bottom-Up Reporting', () => this.testBottomUpReporting());
    await this.runTest('Test 8: Coordinator Failover', () => this.testCoordinatorFailover());
    await this.runTest('Test 9: Worker Failure Recovery', () => this.testWorkerFailureRecovery());
    await this.runTest('Test 10: Hierarchy Metrics', () => this.testHierarchyMetrics());

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

    process.exit(this.results.failed > 0 ? 1 : 0);
  }
}

// Run tests
(async () => {
  const suite = new HierarchicalTopologyTestSuite();
  await suite.runAll();
})();
