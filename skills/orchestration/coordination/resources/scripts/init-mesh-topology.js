#!/usr/bin/env node

/**
 * Initialize Mesh Topology
 *
 * Creates peer-to-peer mesh coordination topology.
 * All agents communicate directly with each other.
 *
 * Usage:
 *   node init-mesh-topology.js --max-agents 6 --strategy balanced
 */

const { execSync } = require('child_process');
const fs = require('fs');

function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    maxAgents: 5,
    strategy: 'balanced',
    configFile: null
  };

  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace('--', '');
    const value = args[i + 1];

    if (key === 'max-agents') config.maxAgents = parseInt(value);
    else if (key === 'strategy') config.strategy = value;
    else if (key === 'config') config.configFile = value;
  }

  // Load from config file if provided
  if (config.configFile && fs.existsSync(config.configFile)) {
    const fileConfig = JSON.parse(fs.readFileSync(config.configFile, 'utf8'));
    config.maxAgents = fileConfig.agents || config.maxAgents;
    config.strategy = fileConfig.parameters?.strategy || config.strategy;
  }

  return config;
}

function initMeshTopology(config) {
  console.log('🕸️  Initializing Mesh Topology...');
  console.log(`   Max Agents: ${config.maxAgents}`);
  console.log(`   Strategy: ${config.strategy}`);
  console.log('');

  try {
    // Initialize swarm with mesh topology
    const initCmd = `npx claude-flow@alpha swarm_init --topology mesh --maxAgents ${config.maxAgents} --strategy ${config.strategy}`;
    console.log('📡 Executing:', initCmd);
    const output = execSync(initCmd, { encoding: 'utf8' });
    console.log(output);

    // Store mesh configuration in memory
    const meshConfig = {
      topology: 'mesh',
      maxAgents: config.maxAgents,
      strategy: config.strategy,
      communicationPattern: 'all-to-all',
      timestamp: new Date().toISOString()
    };

    const storeCmd = `npx claude-flow@alpha memory store --key "coordination/mesh-config" --value '${JSON.stringify(meshConfig)}'`;
    execSync(storeCmd, { encoding: 'utf8' });

    console.log('');
    console.log('✅ Mesh topology initialized successfully!');
    console.log('');
    console.log('Configuration:');
    console.log('   - Communication: All-to-all peer connections');
    console.log('   - Redundancy: High (no single point of failure)');
    console.log('   - Consensus: Byzantine fault tolerance available');
    console.log('');
    console.log('Next steps:');
    console.log('   1. Spawn agents: node spawn-coordinated-agents.js --topology mesh');
    console.log('   2. Assign tasks: node orchestrate-distributed-task.js');
    console.log('   3. Monitor: npx claude-flow@alpha swarm_monitor');

  } catch (error) {
    console.error('❌ Error initializing mesh topology:', error.message);
    process.exit(1);
  }
}

const config = parseArgs();
initMeshTopology(config);
