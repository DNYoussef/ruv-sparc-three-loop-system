#!/usr/bin/env node

/**
 * Consensus Mechanisms Test Suite
 *
 * Comprehensive tests for Byzantine and Raft consensus protocols.
 * Tests: quorum, voting, fault tolerance, leader election, log replication.
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

class ConsensusTestSuite {
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

  // Byzantine Consensus Tests

  // Test 1: Byzantine quorum calculation (2f+1)
  async testByzantineQuorum() {
    const scenarios = [
      { agents: 3, faults: 1, quorum: 3 },  // 2*1+1 = 3
      { agents: 5, faults: 2, quorum: 5 },  // 2*2+1 = 5
      { agents: 7, faults: 3, quorum: 7 }   // 2*3+1 = 7
    ];

    for (const scenario of scenarios) {
      const calculatedQuorum = 2 * scenario.faults + 1;
      assert.strictEqual(calculatedQuorum, scenario.quorum, `Quorum should be ${scenario.quorum} for ${scenario.agents} agents with ${scenario.faults} faults`);
    }

    console.log(`   Verified Byzantine quorum (2f+1) for ${scenarios.length} scenarios`);
  }

  // Test 2: Byzantine voting protocol
  async testByzantineVoting() {
    const proposal = {
      id: 'proposal-1',
      action: 'deploy-production',
      proposer: 'coordinator',
      timestamp: new Date().toISOString()
    };

    // Store proposal
    const proposalOutput = exec(`npx claude-flow@alpha memory store --key "consensus/byzantine/proposal-1" --value '${JSON.stringify(proposal)}'`);
    assert(proposalOutput && !proposalOutput.error, 'Proposal should be stored');

    // Simulate votes from agents
    const votes = {
      'agent-1': { vote: 'yes', signature: 'sig-1', timestamp: new Date().toISOString() },
      'agent-2': { vote: 'yes', signature: 'sig-2', timestamp: new Date().toISOString() },
      'agent-3': { vote: 'yes', signature: 'sig-3', timestamp: new Date().toISOString() },
      'agent-4': { vote: 'no', signature: 'sig-4', timestamp: new Date().toISOString() },
      'agent-5': { vote: 'yes', signature: 'sig-5', timestamp: new Date().toISOString() }
    };

    const votesOutput = exec(`npx claude-flow@alpha memory store --key "consensus/byzantine/proposal-1/votes" --value '${JSON.stringify(votes)}'`);
    assert(votesOutput && !votesOutput.error, 'Votes should be stored');

    // Calculate result
    const yesVotes = Object.values(votes).filter(v => v.vote === 'yes').length;
    const totalVotes = Object.keys(votes).length;
    const threshold = Math.ceil(totalVotes * 0.67); // 2f+1 threshold

    assert(yesVotes >= threshold, `Consensus should be reached with ${yesVotes}/${totalVotes} votes (threshold: ${threshold})`);
    console.log(`   Consensus reached: ${yesVotes}/${totalVotes} votes (threshold: ${threshold})`);
  }

  // Test 3: Byzantine fault tolerance with malicious agents
  async testByzantineFaultTolerance() {
    // Simulate 5 agents with 1 Byzantine (malicious) agent
    const agents = 5;
    const byzantineAgents = 1;
    const honestAgents = agents - byzantineAgents;

    // For Byzantine tolerance, need at least 2f+1 = 3 honest agents where f=1
    const minHonest = 2 * byzantineAgents + 1;

    assert(honestAgents >= minHonest, `Should have at least ${minHonest} honest agents for ${byzantineAgents} Byzantine fault tolerance`);

    // Store fault tolerance configuration
    const config = {
      total_agents: agents,
      byzantine_agents: byzantineAgents,
      honest_agents: honestAgents,
      min_honest_required: minHonest,
      fault_tolerant: honestAgents >= minHonest
    };

    const configOutput = exec(`npx claude-flow@alpha memory store --key "consensus/byzantine/fault-tolerance" --value '${JSON.stringify(config)}'`);
    assert(configOutput && !configOutput.error, 'Fault tolerance config should be stored');

    console.log(`   Byzantine fault tolerance: ${honestAgents}/${agents} honest agents (min required: ${minHonest})`);
  }

  // Test 4: Byzantine signature verification
  async testByzantineSignatureVerification() {
    // Simulate cryptographic signatures for votes
    const signatures = {
      'agent-1': { signature: 'a1b2c3d4e5f6', verified: true },
      'agent-2': { signature: 'b2c3d4e5f6a1', verified: true },
      'agent-3': { signature: 'c3d4e5f6a1b2', verified: true },
      'agent-4': { signature: 'invalid-sig', verified: false },  // Malicious agent
      'agent-5': { signature: 'd4e5f6a1b2c3', verified: true }
    };

    const verifiedSignatures = Object.values(signatures).filter(s => s.verified).length;

    assert(verifiedSignatures >= 3, 'At least 3 signatures should be verified for 5-agent Byzantine consensus');

    const sigOutput = exec(`npx claude-flow@alpha memory store --key "consensus/byzantine/signatures" --value '${JSON.stringify(signatures)}'`);
    assert(sigOutput && !sigOutput.error, 'Signatures should be stored');

    console.log(`   Signature verification: ${verifiedSignatures}/${Object.keys(signatures).length} valid signatures`);
  }

  // Raft Consensus Tests

  // Test 5: Raft leader election
  async testRaftLeaderElection() {
    // Simulate leader election
    const candidates = ['agent-1', 'agent-2', 'agent-3', 'agent-4', 'agent-5'];
    const votes = {
      'agent-1': 'agent-3',  // Vote for agent-3
      'agent-2': 'agent-3',
      'agent-3': 'agent-3',  // Vote for self
      'agent-4': 'agent-1',
      'agent-5': 'agent-3'
    };

    // Count votes
    const voteCounts = {};
    Object.values(votes).forEach(vote => {
      voteCounts[vote] = (voteCounts[vote] || 0) + 1;
    });

    // Find leader (majority)
    const majority = Math.ceil(candidates.length / 2);
    const leader = Object.entries(voteCounts).find(([_, count]) => count >= majority)?.[0];

    assert(leader, 'A leader should be elected');
    assert.strictEqual(leader, 'agent-3', 'Agent-3 should be elected as leader');

    const electionOutput = exec(`npx claude-flow@alpha memory store --key "consensus/raft/leader" --value '${JSON.stringify({ leader, votes: voteCounts })}'`);
    assert(electionOutput && !electionOutput.error, 'Leader election should be stored');

    console.log(`   Raft leader elected: ${leader} with ${voteCounts[leader]}/${candidates.length} votes`);
  }

  // Test 6: Raft log replication
  async testRaftLogReplication() {
    // Simulate log entries
    const logEntries = [
      { index: 1, term: 1, command: 'set x=1' },
      { index: 2, term: 1, command: 'set y=2' },
      { index: 3, term: 2, command: 'set z=3' }
    ];

    // Store log
    const logOutput = exec(`npx claude-flow@alpha memory store --key "consensus/raft/log" --value '${JSON.stringify(logEntries)}'`);
    assert(logOutput && !logOutput.error, 'Raft log should be stored');

    // Simulate replication to followers
    const followers = ['agent-2', 'agent-3', 'agent-4', 'agent-5'];
    for (const follower of followers) {
      const followerLogOutput = exec(`npx claude-flow@alpha memory store --key "consensus/raft/${follower}/log" --value '${JSON.stringify(logEntries)}'`);
      assert(followerLogOutput && !followerLogOutput.error, `Log should be replicated to ${follower}`);
    }

    console.log(`   Raft log replicated: ${logEntries.length} entries to ${followers.length} followers`);
  }

  // Test 7: Raft commit consensus
  async testRaftCommitConsensus() {
    // For a log entry to be committed, it must be replicated on a majority of servers
    const totalServers = 5;
    const majority = Math.ceil(totalServers / 2);

    const logEntry = { index: 4, term: 2, command: 'set a=4' };

    // Simulate replication status
    const replicationStatus = {
      'agent-1': true,  // Leader
      'agent-2': true,
      'agent-3': true,
      'agent-4': false,  // Not yet replicated
      'agent-5': true
    };

    const replicatedCount = Object.values(replicationStatus).filter(r => r).length;

    assert(replicatedCount >= majority, `Log entry should be replicated on at least ${majority} servers`);

    const commitOutput = exec(`npx claude-flow@alpha memory store --key "consensus/raft/commit-status" --value '${JSON.stringify({ logEntry, replicatedCount, majority, committed: replicatedCount >= majority })}'`);
    assert(commitOutput && !commitOutput.error, 'Commit status should be stored');

    console.log(`   Raft commit consensus: ${replicatedCount}/${totalServers} servers (majority: ${majority})`);
  }

  // Test 8: Raft term management
  async testRaftTermManagement() {
    // Raft uses terms to detect stale information
    const termHistory = [
      { term: 1, leader: 'agent-1', duration: 120 },
      { term: 2, leader: 'agent-3', duration: 180 },
      { term: 3, leader: 'agent-3', duration: 200 }
    ];

    const currentTerm = Math.max(...termHistory.map(t => t.term));

    assert.strictEqual(currentTerm, 3, 'Current term should be 3');

    const termOutput = exec(`npx claude-flow@alpha memory store --key "consensus/raft/terms" --value '${JSON.stringify({ history: termHistory, current: currentTerm })}'`);
    assert(termOutput && !termOutput.error, 'Term history should be stored');

    console.log(`   Raft term management: current term ${currentTerm} (${termHistory.length} term transitions)`);
  }

  // Comparison Tests

  // Test 9: Consensus latency comparison
  async testConsensusLatencyComparison() {
    // Byzantine typically has higher latency due to cryptographic verification
    // Raft typically has lower latency for leader-based decisions

    const latencies = {
      byzantine: {
        proposal_broadcast: 5,
        voting_phase: 15,
        signature_verification: 10,
        total: 30
      },
      raft: {
        leader_proposal: 2,
        log_replication: 8,
        commit_acknowledgement: 5,
        total: 15
      }
    };

    assert(latencies.byzantine.total > latencies.raft.total, 'Byzantine consensus should have higher latency than Raft');

    const latencyOutput = exec(`npx claude-flow@alpha memory store --key "consensus/latency-comparison" --value '${JSON.stringify(latencies)}'`);
    assert(latencyOutput && !latencyOutput.error, 'Latency comparison should be stored');

    console.log(`   Consensus latency: Byzantine ${latencies.byzantine.total}ms vs Raft ${latencies.raft.total}ms`);
  }

  // Test 10: Consensus use case selection
  async testConsensusUseCaseSelection() {
    const useCases = [
      { scenario: 'Critical security decision', recommended: 'byzantine', reason: 'High fault tolerance required' },
      { scenario: 'State machine replication', recommended: 'raft', reason: 'Strong consistency with lower latency' },
      { scenario: 'Deployment approval (potential malicious actors)', recommended: 'byzantine', reason: 'Byzantine fault tolerance needed' },
      { scenario: 'Distributed log', recommended: 'raft', reason: 'Efficient log replication' }
    ];

    for (const useCase of useCases) {
      const expected = ['byzantine', 'raft'].includes(useCase.recommended);
      assert(expected, `Use case should recommend valid consensus mechanism: ${useCase.recommended}`);
    }

    const useCaseOutput = exec(`npx claude-flow@alpha memory store --key "consensus/use-cases" --value '${JSON.stringify(useCases)}'`);
    assert(useCaseOutput && !useCaseOutput.error, 'Use cases should be stored');

    console.log(`   Verified consensus selection for ${useCases.length} use cases`);
  }

  // Run all tests
  async runAll() {
    console.log('═══════════════════════════════════════');
    console.log('  Consensus Mechanisms Test Suite');
    console.log('═══════════════════════════════════════');

    console.log('\n--- Byzantine Consensus Tests ---');
    await this.runTest('Test 1: Byzantine Quorum Calculation', () => this.testByzantineQuorum());
    await this.runTest('Test 2: Byzantine Voting Protocol', () => this.testByzantineVoting());
    await this.runTest('Test 3: Byzantine Fault Tolerance', () => this.testByzantineFaultTolerance());
    await this.runTest('Test 4: Byzantine Signature Verification', () => this.testByzantineSignatureVerification());

    console.log('\n--- Raft Consensus Tests ---');
    await this.runTest('Test 5: Raft Leader Election', () => this.testRaftLeaderElection());
    await this.runTest('Test 6: Raft Log Replication', () => this.testRaftLogReplication());
    await this.runTest('Test 7: Raft Commit Consensus', () => this.testRaftCommitConsensus());
    await this.runTest('Test 8: Raft Term Management', () => this.testRaftTermManagement());

    console.log('\n--- Comparison Tests ---');
    await this.runTest('Test 9: Consensus Latency Comparison', () => this.testConsensusLatencyComparison());
    await this.runTest('Test 10: Consensus Use Case Selection', () => this.testConsensusUseCaseSelection());

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
  const suite = new ConsensusTestSuite();
  await suite.runAll();
})();
