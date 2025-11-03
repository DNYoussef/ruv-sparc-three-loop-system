#!/usr/bin/env node
/**
 * Consensus Manager - Byzantine Fault Tolerant Consensus Engine
 * Implements majority, weighted, and Byzantine consensus algorithms
 */

const crypto = require('crypto');

// Consensus algorithms
const ConsensusAlgorithm = {
  MAJORITY: 'majority',
  WEIGHTED: 'weighted',
  BYZANTINE: 'byzantine'
};

// Agent types with voting weights
const AgentWeights = {
  queen: 3,
  worker: 1,
  specialist: 2
};

/**
 * Vote representation
 */
class Vote {
  constructor(agentId, agentType, option, timestamp = null) {
    this.agentId = agentId;
    this.agentType = agentType;
    this.option = option;
    this.timestamp = timestamp || new Date().toISOString();
    this.signature = this._generateSignature();
  }

  _generateSignature() {
    const data = `${this.agentId}:${this.option}:${this.timestamp}`;
    return crypto.createHash('sha256').update(data).digest('hex');
  }

  verify() {
    const expectedSig = this._generateSignature();
    return this.signature === expectedSig;
  }
}

/**
 * Consensus decision result
 */
class ConsensusResult {
  constructor(topic, decision, confidence, votes, algorithm, metadata = {}) {
    this.topic = topic;
    this.decision = decision;
    this.confidence = confidence;
    this.votes = votes;
    this.algorithm = algorithm;
    this.metadata = {
      ...metadata,
      timestamp: new Date().toISOString(),
      totalVotes: votes.length,
      uniqueAgents: new Set(votes.map(v => v.agentId)).size
    };
  }

  toJSON() {
    return {
      topic: this.topic,
      decision: this.decision,
      confidence: this.confidence,
      algorithm: this.algorithm,
      voteBreakdown: this._getVoteBreakdown(),
      metadata: this.metadata
    };
  }

  _getVoteBreakdown() {
    const breakdown = {};
    for (const vote of this.votes) {
      if (!breakdown[vote.option]) {
        breakdown[vote.option] = 0;
      }
      breakdown[vote.option]++;
    }
    return breakdown;
  }
}

/**
 * Consensus Manager - Byzantine Fault Tolerant Decision Making
 */
class ConsensusManager {
  constructor(algorithm = ConsensusAlgorithm.BYZANTINE, config = {}) {
    this.algorithm = algorithm;
    this.config = {
      byzantineThreshold: 2/3, // Require 2/3 supermajority
      minVotes: 1,
      allowAbstain: false,
      ...config
    };
    this.decisions = [];
  }

  /**
   * Build consensus from agent votes
   */
  buildConsensus(topic, votes, options = null) {
    if (!votes || votes.length === 0) {
      throw new Error('No votes provided for consensus');
    }

    // Verify all votes
    const validVotes = votes.filter(v => v.verify());
    if (validVotes.length < this.config.minVotes) {
      throw new Error(`Insufficient valid votes (need ${this.config.minVotes}, got ${validVotes.length})`);
    }

    // Calculate decision based on algorithm
    let decision, confidence;
    switch (this.algorithm) {
      case ConsensusAlgorithm.MAJORITY:
        ({ decision, confidence } = this._calculateMajority(validVotes));
        break;
      case ConsensusAlgorithm.WEIGHTED:
        ({ decision, confidence } = this._calculateWeighted(validVotes));
        break;
      case ConsensusAlgorithm.BYZANTINE:
        ({ decision, confidence } = this._calculateByzantine(validVotes));
        break;
      default:
        throw new Error(`Unknown algorithm: ${this.algorithm}`);
    }

    const result = new ConsensusResult(
      topic,
      decision,
      confidence,
      validVotes,
      this.algorithm,
      { invalidVotes: votes.length - validVotes.length }
    );

    this.decisions.push(result);
    return result;
  }

  /**
   * Simple majority voting
   */
  _calculateMajority(votes) {
    const counts = this._countVotes(votes);
    const total = votes.length;
    const winner = this._getWinner(counts);

    return {
      decision: winner.option,
      confidence: winner.count / total
    };
  }

  /**
   * Weighted voting (queen vote counts 3x)
   */
  _calculateWeighted(votes) {
    const weightedCounts = {};
    let totalWeight = 0;

    for (const vote of votes) {
      const weight = AgentWeights[vote.agentType] || 1;
      if (!weightedCounts[vote.option]) {
        weightedCounts[vote.option] = 0;
      }
      weightedCounts[vote.option] += weight;
      totalWeight += weight;
    }

    const winner = this._getWinner(weightedCounts);

    return {
      decision: winner.option,
      confidence: winner.count / totalWeight
    };
  }

  /**
   * Byzantine fault tolerant consensus (requires 2/3 supermajority)
   */
  _calculateByzantine(votes) {
    const counts = this._countVotes(votes);
    const total = votes.length;
    const threshold = Math.ceil(total * this.config.byzantineThreshold);

    // Check for supermajority
    for (const [option, count] of Object.entries(counts)) {
      if (count >= threshold) {
        return {
          decision: option,
          confidence: count / total
        };
      }
    }

    // No supermajority reached - return majority with warning
    const winner = this._getWinner(counts);
    console.warn(`Byzantine consensus failed to reach ${this.config.byzantineThreshold * 100}% threshold. Falling back to majority.`);

    return {
      decision: winner.option,
      confidence: winner.count / total
    };
  }

  /**
   * Count votes for each option
   */
  _countVotes(votes) {
    const counts = {};
    for (const vote of votes) {
      if (!counts[vote.option]) {
        counts[vote.option] = 0;
      }
      counts[vote.option]++;
    }
    return counts;
  }

  /**
   * Get winning option from vote counts
   */
  _getWinner(counts) {
    let maxCount = 0;
    let winner = null;

    for (const [option, count] of Object.entries(counts)) {
      if (count > maxCount) {
        maxCount = count;
        winner = option;
      }
    }

    return { option: winner, count: maxCount };
  }

  /**
   * Get consensus statistics
   */
  getStatistics() {
    return {
      algorithm: this.algorithm,
      totalDecisions: this.decisions.length,
      averageConfidence: this._calculateAverageConfidence(),
      consensusRate: this._calculateConsensusRate(),
      decisionHistory: this.decisions.map(d => ({
        topic: d.topic,
        decision: d.decision,
        confidence: d.confidence,
        timestamp: d.metadata.timestamp
      }))
    };
  }

  _calculateAverageConfidence() {
    if (this.decisions.length === 0) return 0;
    const sum = this.decisions.reduce((acc, d) => acc + d.confidence, 0);
    return sum / this.decisions.length;
  }

  _calculateConsensusRate() {
    if (this.decisions.length === 0) return 0;
    const strongConsensus = this.decisions.filter(d => d.confidence >= 0.75).length;
    return strongConsensus / this.decisions.length;
  }

  /**
   * Export consensus history
   */
  exportHistory() {
    return {
      algorithm: this.algorithm,
      config: this.config,
      decisions: this.decisions.map(d => d.toJSON()),
      statistics: this.getStatistics(),
      exportedAt: new Date().toISOString()
    };
  }
}

// Demo usage
function demo() {
  console.log('=== Consensus Manager Demo ===\n');

  // Create Byzantine consensus manager
  const manager = new ConsensusManager(ConsensusAlgorithm.BYZANTINE);

  // Simulate votes on architecture decision
  const votes = [
    new Vote('queen-1', 'queen', 'GraphQL'),
    new Vote('worker-1', 'worker', 'GraphQL'),
    new Vote('worker-2', 'worker', 'REST'),
    new Vote('worker-3', 'worker', 'GraphQL'),
    new Vote('worker-4', 'worker', 'GraphQL'),
    new Vote('specialist-1', 'specialist', 'GraphQL')
  ];

  // Build consensus
  const result = manager.buildConsensus('API Architecture', votes);
  console.log('Consensus Result:');
  console.log(JSON.stringify(result.toJSON(), null, 2));

  // Test weighted consensus
  console.log('\n=== Testing Weighted Consensus ===\n');
  const weightedManager = new ConsensusManager(ConsensusAlgorithm.WEIGHTED);

  const weightedVotes = [
    new Vote('queen-1', 'queen', 'microservices'),    // 3 votes
    new Vote('worker-1', 'worker', 'monolith'),       // 1 vote
    new Vote('worker-2', 'worker', 'monolith'),       // 1 vote
    new Vote('worker-3', 'worker', 'monolith')        // 1 vote
  ];

  const weightedResult = weightedManager.buildConsensus('Architecture Pattern', weightedVotes);
  console.log('Weighted Result (Queen vote counts 3x):');
  console.log(JSON.stringify(weightedResult.toJSON(), null, 2));

  // Statistics
  console.log('\n=== Consensus Statistics ===\n');
  console.log(JSON.stringify(manager.getStatistics(), null, 2));
}

// Run demo if executed directly
if (require.main === module) {
  demo();
}

module.exports = {
  ConsensusManager,
  ConsensusAlgorithm,
  Vote,
  ConsensusResult,
  AgentWeights
};
