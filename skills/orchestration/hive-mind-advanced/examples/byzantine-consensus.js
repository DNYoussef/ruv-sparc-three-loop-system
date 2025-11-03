#!/usr/bin/env node
/**
 * Example: Byzantine Fault Tolerant Consensus
 * Demonstrates robust decision-making with faulty agents
 *
 * Scenario: Production deployment decision with potentially faulty agents
 * - Up to 1/3 of agents can be faulty/malicious
 * - 2/3 supermajority required for critical decisions
 * - Cryptographic vote signatures
 * - Audit trail for all decisions
 */

const crypto = require('crypto');

// ============================================================================
// Cryptographic Vote Implementation
// ============================================================================

class CryptographicVote {
  constructor(agentId, agentType, decision, privateKey = null) {
    this.agentId = agentId;
    this.agentType = agentType;
    this.decision = decision;
    this.timestamp = new Date().toISOString();
    this.nonce = crypto.randomBytes(16).toString('hex');

    // Generate signature
    this.signature = this._sign(privateKey);
  }

  _sign(privateKey) {
    const data = JSON.stringify({
      agentId: this.agentId,
      decision: this.decision,
      timestamp: this.timestamp,
      nonce: this.nonce
    });

    // In production, use actual private key cryptography
    // For demo, use HMAC-SHA256
    const key = privateKey || 'demo-secret-key';
    return crypto.createHmac('sha256', key).update(data).digest('hex');
  }

  verify(publicKey = null) {
    const expectedSig = this._sign(publicKey);
    return this.signature === expectedSig;
  }

  toJSON() {
    return {
      agentId: this.agentId,
      agentType: this.agentType,
      decision: this.decision,
      timestamp: this.timestamp,
      nonce: this.nonce,
      signature: this.signature
    };
  }
}

// ============================================================================
// Byzantine Agent (can be honest or faulty)
// ============================================================================

class ByzantineAgent {
  constructor(id, type, isFaulty = false) {
    this.id = id;
    this.type = type;
    this.isFaulty = isFaulty;
    this.reputation = 1.0;
    this.votingHistory = [];
  }

  vote(scenario, options) {
    if (this.isFaulty) {
      return this._faultyVote(scenario, options);
    } else {
      return this._honestVote(scenario, options);
    }
  }

  _honestVote(scenario, options) {
    // Honest agent analyzes scenario and votes rationally
    const analysis = this._analyzeScenario(scenario);
    const decision = this._makeRationalDecision(analysis, options);

    const vote = new CryptographicVote(
      this.id,
      this.type,
      decision
    );

    this.votingHistory.push({
      scenario: scenario.description,
      decision,
      honest: true,
      timestamp: new Date().toISOString()
    });

    return vote;
  }

  _faultyVote(scenario, options) {
    // Faulty agent votes randomly or maliciously
    const faultyBehaviors = [
      () => options[Math.floor(Math.random() * options.length)], // Random
      () => options[options.length - 1], // Always worst option
      () => 'INVALID_OPTION', // Invalid vote
      () => null // No vote
    ];

    const behavior = faultyBehaviors[Math.floor(Math.random() * faultyBehaviors.length)];
    const decision = behavior();

    if (decision === null) {
      return null;
    }

    const vote = new CryptographicVote(
      this.id,
      this.type,
      decision
    );

    this.votingHistory.push({
      scenario: scenario.description,
      decision,
      honest: false,
      faulty: true,
      timestamp: new Date().toISOString()
    });

    return vote;
  }

  _analyzeScenario(scenario) {
    return {
      risk: scenario.risk || 'medium',
      complexity: scenario.complexity || 'medium',
      impact: scenario.impact || 'medium',
      urgency: scenario.urgency || 'medium'
    };
  }

  _makeRationalDecision(analysis, options) {
    // Strategic decision based on analysis
    if (analysis.risk === 'high') {
      return options.includes('defer') ? 'defer' : options[0];
    }

    if (analysis.urgency === 'critical') {
      return options.includes('proceed') ? 'proceed' : options[0];
    }

    // Default to first option
    return options[0];
  }

  updateReputation(delta) {
    this.reputation = Math.max(0, Math.min(1, this.reputation + delta));
  }
}

// ============================================================================
// Byzantine Consensus Manager
// ============================================================================

class ByzantineConsensusManager {
  constructor(options = {}) {
    this.threshold = options.threshold || 2/3;
    this.minVotes = options.minVotes || 3;
    this.agents = new Map();
    this.decisions = [];
    this.auditLog = [];
  }

  registerAgent(agent) {
    this.agents.set(agent.id, agent);
    this._log('agent_registered', { agentId: agent.id, type: agent.type });
  }

  async buildConsensus(scenario, options) {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`Building Byzantine Consensus: ${scenario.description}`);
    console.log(`${'='.repeat(80)}`);
    console.log(`Options: ${options.join(', ')}`);
    console.log(`Threshold: ${(this.threshold * 100).toFixed(0)}% (${Math.ceil(this.agents.size * this.threshold)}/${this.agents.size} votes)`);

    this._log('consensus_start', { scenario: scenario.description, options });

    // Collect votes from all agents
    const votes = [];
    const faultyAgents = [];

    for (const agent of this.agents.values()) {
      const vote = agent.vote(scenario, options);

      if (vote === null) {
        console.log(`  [${agent.id}] No vote (faulty behavior)`);
        faultyAgents.push(agent.id);
        continue;
      }

      // Verify vote signature
      if (!vote.verify()) {
        console.log(`  [${agent.id}] Invalid signature (faulty agent)`);
        faultyAgents.push(agent.id);
        agent.updateReputation(-0.1);
        continue;
      }

      // Check if decision is valid
      if (!options.includes(vote.decision)) {
        console.log(`  [${agent.id}] Invalid decision: ${vote.decision} (faulty agent)`);
        faultyAgents.push(agent.id);
        agent.updateReputation(-0.1);
        continue;
      }

      votes.push(vote);
      console.log(`  [${agent.id}] ${agent.isFaulty ? '❌' : '✓'} Voted: ${vote.decision}`);
    }

    console.log(`\nValid votes: ${votes.length}/${this.agents.size}`);
    console.log(`Faulty agents detected: ${faultyAgents.length}`);

    // Check if we have minimum votes
    if (votes.length < this.minVotes) {
      const result = {
        success: false,
        reason: 'Insufficient valid votes',
        validVotes: votes.length,
        required: this.minVotes
      };

      this._log('consensus_failed', result);
      return result;
    }

    // Count votes
    const counts = {};
    options.forEach(opt => counts[opt] = 0);

    votes.forEach(vote => {
      counts[vote.decision]++;
    });

    console.log(`\nVote counts:`);
    Object.entries(counts).forEach(([option, count]) => {
      const percentage = (count / votes.length * 100).toFixed(1);
      const bar = '█'.repeat(Math.floor(count / votes.length * 40));
      console.log(`  ${option}: ${count} votes (${percentage}%) ${bar}`);
    });

    // Calculate Byzantine consensus (2/3 supermajority)
    const requiredVotes = Math.ceil(votes.length * this.threshold);

    let decision = null;
    let supermajority = false;

    for (const [option, count] of Object.entries(counts)) {
      if (count >= requiredVotes) {
        decision = option;
        supermajority = true;
        break;
      }
    }

    // If no supermajority, use simple majority
    if (!decision) {
      console.log(`\n⚠️  No ${(this.threshold * 100).toFixed(0)}% supermajority reached`);
      console.log(`   Falling back to simple majority...`);

      let maxVotes = 0;
      for (const [option, count] of Object.entries(counts)) {
        if (count > maxVotes) {
          maxVotes = count;
          decision = option;
        }
      }
    }

    const confidence = counts[decision] / votes.length;

    const result = {
      success: true,
      decision,
      confidence,
      supermajority,
      totalVotes: votes.length,
      requiredForSupermajority: requiredVotes,
      voteCounts: counts,
      faultyAgents,
      scenario: scenario.description,
      timestamp: new Date().toISOString()
    };

    console.log(`\n${'='.repeat(80)}`);
    console.log(`CONSENSUS RESULT`);
    console.log(`${'='.repeat(80)}`);
    console.log(`Decision: ${decision}`);
    console.log(`Confidence: ${(confidence * 100).toFixed(1)}%`);
    console.log(`Supermajority: ${supermajority ? 'YES ✓' : 'NO ✗'}`);
    console.log(`Faulty Agents Tolerated: ${faultyAgents.length}/${Math.floor(this.agents.size / 3)} (max 1/3)`);

    this.decisions.push(result);
    this._log('consensus_reached', result);

    return result;
  }

  // Check if system can tolerate current faulty agents
  canTolerateFailures(faultyCount) {
    const maxFaulty = Math.floor(this.agents.size / 3);
    return faultyCount <= maxFaulty;
  }

  _log(event, data) {
    this.auditLog.push({
      event,
      data,
      timestamp: new Date().toISOString()
    });
  }

  getAuditTrail() {
    return this.auditLog;
  }

  getStatistics() {
    const totalDecisions = this.decisions.length;
    const supermajorityDecisions = this.decisions.filter(d => d.supermajority).length;
    const avgConfidence = this.decisions.reduce((sum, d) => sum + d.confidence, 0) / totalDecisions;

    return {
      totalAgents: this.agents.size,
      totalDecisions,
      supermajorityDecisions,
      supermajorityRate: (supermajorityDecisions / totalDecisions * 100).toFixed(1) + '%',
      avgConfidence: (avgConfidence * 100).toFixed(1) + '%',
      byzantineThreshold: (this.threshold * 100).toFixed(0) + '%',
      maxTolerableFaults: Math.floor(this.agents.size / 3)
    };
  }
}

// ============================================================================
// Demo: Production Deployment Decisions with Faulty Agents
// ============================================================================

async function main() {
  console.log('='.repeat(80));
  console.log('Byzantine Fault Tolerant Consensus Demo');
  console.log('Scenario: Production deployment with potentially faulty agents');
  console.log('='.repeat(80));

  const consensus = new ByzantineConsensusManager({
    threshold: 2/3,
    minVotes: 3
  });

  // Register 9 agents (3 will be faulty - exactly at 1/3 boundary)
  console.log('\nRegistering agents...');

  // 6 honest agents
  for (let i = 1; i <= 6; i++) {
    const agent = new ByzantineAgent(`honest-${i}`, 'worker', false);
    consensus.registerAgent(agent);
    console.log(`  ✓ Registered honest agent: ${agent.id}`);
  }

  // 3 faulty agents (Byzantine attackers)
  for (let i = 1; i <= 3; i++) {
    const agent = new ByzantineAgent(`faulty-${i}`, 'worker', true);
    consensus.registerAgent(agent);
    console.log(`  ❌ Registered faulty agent: ${agent.id}`);
  }

  console.log(`\nTotal agents: 9 (6 honest, 3 faulty)`);
  console.log(`Byzantine tolerance: Can tolerate ${Math.floor(9 / 3)} faulty agents`);

  // Scenario 1: Deploy to production (critical decision)
  const scenario1 = {
    description: 'Deploy v2.0 to production',
    risk: 'high',
    impact: 'critical',
    urgency: 'medium',
    complexity: 'high'
  };

  await consensus.buildConsensus(scenario1, [
    'deploy_immediately',
    'deploy_with_canary',
    'defer_deployment',
    'rollback_plan_first'
  ]);

  // Scenario 2: Handle security vulnerability
  const scenario2 = {
    description: 'Critical security patch deployment',
    risk: 'critical',
    impact: 'critical',
    urgency: 'critical',
    complexity: 'low'
  };

  await consensus.buildConsensus(scenario2, [
    'emergency_patch',
    'scheduled_maintenance',
    'defer_to_weekend'
  ]);

  // Scenario 3: Scale infrastructure
  const scenario3 = {
    description: 'Scale up infrastructure for Black Friday',
    risk: 'medium',
    impact: 'high',
    urgency: 'high',
    complexity: 'medium'
  };

  await consensus.buildConsensus(scenario3, [
    'scale_now',
    'scale_gradually',
    'wait_for_metrics'
  ]);

  // Statistics
  console.log(`\n${'='.repeat(80)}`);
  console.log('CONSENSUS STATISTICS');
  console.log('='.repeat(80));

  const stats = consensus.getStatistics();
  console.log(JSON.stringify(stats, null, 2));

  // Audit Trail
  console.log(`\n${'='.repeat(80)}`);
  console.log('AUDIT TRAIL');
  console.log('='.repeat(80));

  const auditTrail = consensus.getAuditTrail();
  console.log(`Total events: ${auditTrail.length}`);
  console.log(`\nRecent events:`);
  auditTrail.slice(-5).forEach(entry => {
    console.log(`  ${entry.timestamp}: ${entry.event}`);
  });

  console.log(`\n${'='.repeat(80)}`);
  console.log('Demo completed successfully!');
  console.log(`${'='.repeat(80)}`);
}

// Run demo
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { ByzantineConsensusManager, ByzantineAgent, CryptographicVote };
