#!/usr/bin/env node
/**
 * Example: Collective Decision Making with Memory
 * Demonstrates intelligent decision-making using shared knowledge and learning
 *
 * Scenario: Technical architecture decisions with historical context
 * - Collective memory of past decisions
 * - Pattern learning from successes/failures
 * - Association-based recommendations
 * - Confidence scoring based on historical data
 */

// ============================================================================
// Collective Memory System
// ============================================================================

class CollectiveMemory {
  constructor() {
    this.memory = new Map();
    this.associations = new Map();
    this.decisionHistory = [];
    this.patterns = new Map();
  }

  // Store decision with metadata
  storeDecision(decision, metadata = {}) {
    const entry = {
      decision,
      metadata: {
        timestamp: new Date().toISOString(),
        confidence: metadata.confidence || 0.8,
        outcome: metadata.outcome || null,
        context: metadata.context || {},
        lessons: metadata.lessons || [],
        ...metadata
      }
    };

    const key = `decision:${decision.topic}:${Date.now()}`;
    this.memory.set(key, entry);
    this.decisionHistory.push(entry);

    // Learn patterns from decision
    this._learnPattern(decision, metadata);

    return entry;
  }

  // Learn patterns from decisions
  _learnPattern(decision, metadata) {
    const patternKey = `${decision.topic}:${decision.chosen}`;

    if (!this.patterns.has(patternKey)) {
      this.patterns.set(patternKey, {
        occurrences: 0,
        successRate: 0,
        avgConfidence: 0,
        contexts: []
      });
    }

    const pattern = this.patterns.get(patternKey);
    pattern.occurrences++;
    pattern.contexts.push(metadata.context);

    if (metadata.outcome) {
      const successCount = pattern.occurrences * pattern.successRate;
      const newSuccessCount = successCount + (metadata.outcome.success ? 1 : 0);
      pattern.successRate = newSuccessCount / pattern.occurrences;
    }

    // Update average confidence
    const confSum = pattern.avgConfidence * (pattern.occurrences - 1) + (metadata.confidence || 0.8);
    pattern.avgConfidence = confSum / pattern.occurrences;
  }

  // Associate related concepts
  associate(concept1, concept2, strength = 0.8, context = {}) {
    const createAssociation = (from, to) => {
      if (!this.associations.has(from)) {
        this.associations.set(from, new Map());
      }
      this.associations.get(from).set(to, { strength, context });
    };

    // Bidirectional association
    createAssociation(concept1, concept2);
    createAssociation(concept2, concept1);
  }

  // Get related concepts
  getRelated(concept, minStrength = 0.5) {
    const related = this.associations.get(concept);
    if (!related) return [];

    return Array.from(related.entries())
      .filter(([_, data]) => data.strength >= minStrength)
      .sort((a, b) => b[1].strength - a[1].strength)
      .map(([concept, data]) => ({ concept, ...data }));
  }

  // Find similar past decisions
  findSimilarDecisions(topic, limit = 5) {
    const similar = [];

    for (const entry of this.decisionHistory) {
      if (entry.decision.topic.toLowerCase().includes(topic.toLowerCase()) ||
          topic.toLowerCase().includes(entry.decision.topic.toLowerCase())) {
        similar.push(entry);
      }
    }

    // Sort by confidence and recency
    similar.sort((a, b) => {
      const scoreA = a.metadata.confidence * 0.7 +
                     (new Date(a.metadata.timestamp).getTime() / Date.now()) * 0.3;
      const scoreB = b.metadata.confidence * 0.7 +
                     (new Date(b.metadata.timestamp).getTime() / Date.now()) * 0.3;
      return scoreB - scoreA;
    });

    return similar.slice(0, limit);
  }

  // Get recommendation based on patterns
  getRecommendation(topic, options) {
    const recommendations = {};

    for (const option of options) {
      const patternKey = `${topic}:${option}`;
      const pattern = this.patterns.get(patternKey);

      if (pattern) {
        recommendations[option] = {
          confidence: pattern.avgConfidence * 0.5 + pattern.successRate * 0.5,
          occurrences: pattern.occurrences,
          successRate: pattern.successRate,
          source: 'pattern_learning'
        };
      } else {
        // No pattern - use neutral confidence
        recommendations[option] = {
          confidence: 0.5,
          occurrences: 0,
          successRate: 0,
          source: 'no_history'
        };
      }
    }

    // Sort by confidence
    return Object.entries(recommendations)
      .sort(([_, a], [__, b]) => b.confidence - a.confidence)
      .map(([option, data]) => ({ option, ...data }));
  }

  getStatistics() {
    return {
      totalDecisions: this.decisionHistory.length,
      patternsLearned: this.patterns.size,
      associations: this.associations.size,
      avgConfidence: this.decisionHistory.reduce((sum, d) =>
        sum + d.metadata.confidence, 0) / this.decisionHistory.length || 0
    };
  }
}

// ============================================================================
// Intelligent Agent with Learning
// ============================================================================

class IntelligentAgent {
  constructor(id, specialization, memory) {
    this.id = id;
    this.specialization = specialization;
    this.memory = memory;
    this.expertise = this._getExpertise(specialization);
  }

  _getExpertise(specialization) {
    const expertiseMap = {
      architect: {
        areas: ['system-design', 'scalability', 'patterns'],
        weight: 1.2
      },
      security: {
        areas: ['security', 'compliance', 'encryption'],
        weight: 1.3
      },
      performance: {
        areas: ['optimization', 'caching', 'databases'],
        weight: 1.1
      },
      devops: {
        areas: ['deployment', 'infrastructure', 'monitoring'],
        weight: 1.0
      }
    };

    return expertiseMap[specialization] || { areas: ['general'], weight: 1.0 };
  }

  // Make informed decision using collective memory
  makeDecision(topic, options, context = {}) {
    console.log(`\n[${this.id}] Analyzing decision: ${topic}`);

    // 1. Check for similar past decisions
    const similar = this.memory.findSimilarDecisions(topic);
    console.log(`  Found ${similar.length} similar past decisions`);

    // 2. Get pattern-based recommendations
    const recommendations = this.memory.getRecommendation(topic, options);
    console.log(`  Pattern analysis complete`);

    // 3. Check associations
    const relatedConcepts = options.flatMap(opt =>
      this.memory.getRelated(opt.toLowerCase(), 0.6)
    );
    console.log(`  Found ${relatedConcepts.length} related concepts`);

    // 4. Apply expertise weight
    let bestOption = recommendations[0].option;
    let confidence = recommendations[0].confidence;

    if (this.expertise.areas.some(area => topic.toLowerCase().includes(area))) {
      confidence *= this.expertise.weight;
      console.log(`  Expertise bonus applied: ${this.expertise.weight}x`);
    }

    // 5. Consider historical outcomes
    if (similar.length > 0) {
      const historicalSuccess = similar.filter(d =>
        d.metadata.outcome && d.metadata.outcome.success
      ).length / similar.length;

      console.log(`  Historical success rate: ${(historicalSuccess * 100).toFixed(1)}%`);

      // Adjust confidence based on historical success
      confidence = confidence * 0.7 + historicalSuccess * 0.3;
    }

    // 6. Final decision
    confidence = Math.min(confidence, 1.0);

    console.log(`  Decision: ${bestOption} (${(confidence * 100).toFixed(1)}% confidence)`);

    return {
      agent: this.id,
      specialization: this.specialization,
      decision: bestOption,
      confidence,
      reasoning: this._generateReasoning(topic, bestOption, recommendations, similar),
      relatedConcepts: relatedConcepts.slice(0, 3)
    };
  }

  _generateReasoning(topic, decision, recommendations, similar) {
    const reasons = [];

    const rec = recommendations.find(r => r.option === decision);
    if (rec.occurrences > 0) {
      reasons.push(`Pattern learned from ${rec.occurrences} similar decisions`);
      reasons.push(`${(rec.successRate * 100).toFixed(0)}% success rate in past`);
    }

    if (similar.length > 0) {
      reasons.push(`Informed by ${similar.length} historical decisions`);
    }

    if (this.expertise.areas.some(area => topic.toLowerCase().includes(area))) {
      reasons.push(`Expertise in ${this.specialization}`);
    }

    return reasons;
  }
}

// ============================================================================
// Collective Decision Maker
// ============================================================================

class CollectiveDecisionMaker {
  constructor() {
    this.memory = new CollectiveMemory();
    this.agents = [];
  }

  addAgent(agent) {
    this.agents.push(agent);
  }

  async makeCollectiveDecision(topic, options, context = {}) {
    console.log('\n' + '='.repeat(80));
    console.log(`Collective Decision: ${topic}`);
    console.log('='.repeat(80));
    console.log(`Options: ${options.join(', ')}`);

    // Gather decisions from all agents
    const agentDecisions = this.agents.map(agent =>
      agent.makeDecision(topic, options, context)
    );

    // Aggregate decisions
    const voteCounts = {};
    const confidenceScores = {};

    options.forEach(opt => {
      voteCounts[opt] = 0;
      confidenceScores[opt] = [];
    });

    agentDecisions.forEach(decision => {
      voteCounts[decision.decision]++;
      confidenceScores[decision.decision].push(decision.confidence);
    });

    // Calculate weighted decision
    let bestOption = null;
    let bestScore = -1;

    for (const option of options) {
      const votes = voteCounts[option];
      const avgConfidence = confidenceScores[option].length > 0
        ? confidenceScores[option].reduce((a, b) => a + b, 0) / confidenceScores[option].length
        : 0;

      const score = (votes / this.agents.length) * 0.6 + avgConfidence * 0.4;

      if (score > bestScore) {
        bestScore = score;
        bestOption = option;
      }
    }

    const avgConfidence = confidenceScores[bestOption].reduce((a, b) => a + b, 0) /
                         confidenceScores[bestOption].length;

    const result = {
      topic,
      chosen: bestOption,
      confidence: avgConfidence,
      votes: voteCounts,
      agentDecisions,
      context
    };

    console.log('\n' + '='.repeat(80));
    console.log('COLLECTIVE DECISION RESULT');
    console.log('='.repeat(80));
    console.log(`Chosen: ${bestOption}`);
    console.log(`Confidence: ${(avgConfidence * 100).toFixed(1)}%`);
    console.log(`Vote distribution:`);
    Object.entries(voteCounts).forEach(([opt, votes]) => {
      const pct = (votes / this.agents.length * 100).toFixed(0);
      const bar = '█'.repeat(Math.floor(votes / this.agents.length * 30));
      console.log(`  ${opt}: ${votes}/${this.agents.length} (${pct}%) ${bar}`);
    });

    // Store decision in collective memory
    this.memory.storeDecision(result, {
      confidence: avgConfidence,
      context
    });

    return result;
  }

  async recordOutcome(decision, outcome) {
    // Update decision with outcome
    const entry = this.memory.decisionHistory.find(d =>
      d.decision.topic === decision.topic && d.decision.chosen === decision.chosen
    );

    if (entry) {
      entry.metadata.outcome = outcome;
      console.log(`\nRecorded outcome for "${decision.topic}": ${outcome.success ? 'SUCCESS' : 'FAILURE'}`);

      // Re-learn pattern with outcome
      this.memory._learnPattern(decision, {
        ...entry.metadata,
        outcome
      });
    }
  }
}

// ============================================================================
// Demo: Architecture Decision Making with Learning
// ============================================================================

async function main() {
  console.log('='.repeat(80));
  console.log('Collective Decision Making with Memory & Learning Demo');
  console.log('='.repeat(80));

  const collective = new CollectiveDecisionMaker();

  // Create specialized agents
  const architect = new IntelligentAgent('architect-1', 'architect', collective.memory);
  const security = new IntelligentAgent('security-1', 'security', collective.memory);
  const performance = new IntelligentAgent('perf-1', 'performance', collective.memory);
  const devops = new IntelligentAgent('devops-1', 'devops', collective.memory);

  collective.addAgent(architect);
  collective.addAgent(security);
  collective.addAgent(performance);
  collective.addAgent(devops);

  // Seed some initial knowledge and associations
  console.log('\nSeeding collective memory with domain knowledge...');

  collective.memory.associate('microservices', 'scalability', 0.9);
  collective.memory.associate('microservices', 'complexity', 0.8);
  collective.memory.associate('monolith', 'simplicity', 0.9);
  collective.memory.associate('kubernetes', 'microservices', 0.85);
  collective.memory.associate('kubernetes', 'complexity', 0.7);

  // Decision 1: Architecture Pattern
  const decision1 = await collective.makeCollectiveDecision(
    'System architecture pattern for e-commerce platform',
    ['microservices', 'monolith', 'modular-monolith'],
    { scale: 'large', team: 'distributed', timeline: '6-months' }
  );

  // Simulate outcome (success)
  await collective.recordOutcome(decision1, {
    success: true,
    metrics: {
      scalability: 9,
      complexity: 7,
      teamProductivity: 8
    },
    lessons: ['Microservices increased complexity but improved scalability']
  });

  // Decision 2: Database Choice (now with learned patterns)
  const decision2 = await collective.makeCollectiveDecision(
    'Database for product catalog service',
    ['PostgreSQL', 'MongoDB', 'DynamoDB'],
    { dataType: 'structured', scale: 'high', consistency: 'strong' }
  );

  await collective.recordOutcome(decision2, {
    success: true,
    metrics: {
      performance: 9,
      reliability: 9,
      scalability: 8
    },
    lessons: ['PostgreSQL provided strong consistency and good performance']
  });

  // Decision 3: Caching Strategy (with even more context)
  const decision3 = await collective.makeCollectiveDecision(
    'Caching strategy for product service',
    ['Redis', 'Memcached', 'In-Memory', 'CDN-Only'],
    { readWrite: '90/10', latency: 'critical', dataSize: 'medium' }
  );

  await collective.recordOutcome(decision3, {
    success: true,
    metrics: {
      latency: 9,
      hitRate: 85,
      complexity: 6
    }
  });

  // Statistics
  console.log('\n' + '='.repeat(80));
  console.log('COLLECTIVE MEMORY STATISTICS');
  console.log('='.repeat(80));
  const stats = collective.memory.getStatistics();
  console.log(JSON.stringify(stats, null, 2));

  // Show learned patterns
  console.log('\n' + '='.repeat(80));
  console.log('LEARNED PATTERNS');
  console.log('='.repeat(80));
  for (const [pattern, data] of collective.memory.patterns) {
    console.log(`\n${pattern}:`);
    console.log(`  Occurrences: ${data.occurrences}`);
    console.log(`  Success Rate: ${(data.successRate * 100).toFixed(1)}%`);
    console.log(`  Avg Confidence: ${(data.avgConfidence * 100).toFixed(1)}%`);
  }

  console.log('\n' + '='.repeat(80));
  console.log('Demo completed successfully!');
  console.log('='.repeat(80));
}

// Run demo
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { CollectiveDecisionMaker, IntelligentAgent, CollectiveMemory };
