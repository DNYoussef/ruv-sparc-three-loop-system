#!/usr/bin/env node

/**
 * Topology Selector Script
 *
 * Analyzes coordination requirements and recommends optimal topology.
 * Considers: agent count, task type, fault tolerance, communication overhead.
 *
 * Usage:
 *   node topology-selector.js --agents 6 --task-type distributed --fault-tolerance high
 */

const fs = require('fs');
const path = require('path');

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    agents: 5,
    taskType: 'general',
    faultTolerance: 'medium',
    output: 'topology-recommendation.json'
  };

  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace('--', '');
    const value = args[i + 1];

    if (key === 'agents') config.agents = parseInt(value);
    else if (key === 'task-type') config.taskType = value;
    else if (key === 'fault-tolerance') config.faultTolerance = value;
    else if (key === 'output') config.output = value;
  }

  return config;
}

// Topology selection logic
function selectTopology(config) {
  const { agents, taskType, faultTolerance } = config;

  // Decision matrix
  const recommendations = [];

  // Agent count heuristics
  if (agents <= 3) {
    recommendations.push({
      topology: 'mesh',
      score: 90,
      reason: 'Small agent count (≤3) optimal for mesh topology'
    });
  } else if (agents <= 8) {
    recommendations.push({
      topology: 'mesh',
      score: 70,
      reason: 'Medium agent count (4-8) workable with mesh'
    });
    recommendations.push({
      topology: 'hierarchical',
      score: 75,
      reason: 'Medium agent count benefits from hierarchical structure'
    });
  } else {
    recommendations.push({
      topology: 'hierarchical',
      score: 95,
      reason: 'Large agent count (>8) requires hierarchical coordination'
    });
    recommendations.push({
      topology: 'adaptive',
      score: 80,
      reason: 'Large agent count benefits from adaptive topology'
    });
  }

  // Task type heuristics
  if (taskType === 'consensus') {
    recommendations.push({
      topology: 'mesh',
      score: 85,
      reason: 'Consensus tasks benefit from peer-to-peer mesh'
    });
  } else if (taskType === 'distributed') {
    recommendations.push({
      topology: 'hierarchical',
      score: 90,
      reason: 'Distributed tasks benefit from delegation hierarchy'
    });
  } else if (taskType === 'pipeline') {
    recommendations.push({
      topology: 'ring',
      score: 80,
      reason: 'Pipeline tasks benefit from ring topology'
    });
  }

  // Fault tolerance heuristics
  if (faultTolerance === 'high') {
    recommendations.push({
      topology: 'mesh',
      score: 95,
      reason: 'High fault tolerance requires redundant mesh connections'
    });
  } else if (faultTolerance === 'medium') {
    recommendations.push({
      topology: 'hierarchical',
      score: 75,
      reason: 'Medium fault tolerance works with hierarchical backup'
    });
  }

  // Aggregate scores by topology
  const topologyScores = {};
  recommendations.forEach(rec => {
    if (!topologyScores[rec.topology]) {
      topologyScores[rec.topology] = { score: 0, reasons: [] };
    }
    topologyScores[rec.topology].score += rec.score;
    topologyScores[rec.topology].reasons.push(rec.reason);
  });

  // Sort by score
  const ranked = Object.entries(topologyScores)
    .map(([topology, data]) => ({
      topology,
      score: data.score,
      reasons: data.reasons
    }))
    .sort((a, b) => b.score - a.score);

  return ranked[0]; // Return top recommendation
}

// Generate configuration
function generateConfig(topology, config) {
  const baseConfig = {
    topology: topology.topology,
    recommendationScore: topology.score,
    reasons: topology.reasons,
    agents: config.agents,
    parameters: {}
  };

  // Topology-specific parameters
  if (topology.topology === 'mesh') {
    baseConfig.parameters = {
      strategy: 'balanced',
      maxAgents: config.agents,
      communicationPattern: 'all-to-all',
      consensusMechanism: config.faultTolerance === 'high' ? 'byzantine' : 'raft'
    };
  } else if (topology.topology === 'hierarchical') {
    baseConfig.parameters = {
      strategy: 'specialized',
      maxAgents: config.agents,
      levels: Math.ceil(Math.log2(config.agents)),
      coordinatorType: 'hierarchical-coordinator',
      delegationStrategy: 'task-based'
    };
  } else if (topology.topology === 'adaptive') {
    baseConfig.parameters = {
      strategy: 'adaptive',
      maxAgents: config.agents,
      reconfigurationInterval: 30,
      performanceThreshold: 0.7
    };
  }

  return baseConfig;
}

// Main execution
function main() {
  const config = parseArgs();

  console.log('🔍 Analyzing coordination requirements...');
  console.log(`   Agents: ${config.agents}`);
  console.log(`   Task Type: ${config.taskType}`);
  console.log(`   Fault Tolerance: ${config.faultTolerance}`);
  console.log('');

  const recommendation = selectTopology(config);
  const fullConfig = generateConfig(recommendation, config);

  console.log('✅ Topology Recommendation:');
  console.log(`   Topology: ${recommendation.topology.toUpperCase()}`);
  console.log(`   Score: ${recommendation.score}`);
  console.log('   Reasons:');
  recommendation.reasons.forEach(r => console.log(`   - ${r}`));
  console.log('');

  // Write to file
  fs.writeFileSync(config.output, JSON.stringify(fullConfig, null, 2));
  console.log(`📁 Configuration written to: ${config.output}`);
  console.log('');
  console.log('Next steps:');
  console.log(`   1. Review configuration in ${config.output}`);
  console.log(`   2. Initialize topology: node init-${recommendation.topology}-topology.js --config ${config.output}`);
  console.log(`   3. Spawn agents with roles`);
}

main();
