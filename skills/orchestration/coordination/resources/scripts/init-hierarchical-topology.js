#!/usr/bin/env node

/**
 * Initialize Hierarchical Topology
 *
 * Creates tree-like hierarchical coordination topology.
 * Parent-child relationships with clear delegation paths.
 *
 * Usage:
 *   node init-hierarchical-topology.js --max-agents 10 --levels 3
 */

const { execSync } = require('child_process');
const fs = require('fs');

function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    maxAgents: 10,
    strategy: 'specialized',
    levels: 3,
    configFile: null
  };

  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace('--', '');
    const value = args[i + 1];

    if (key === 'max-agents') config.maxAgents = parseInt(value);
    else if (key === 'strategy') config.strategy = value;
    else if (key === 'levels') config.levels = parseInt(value);
    else if (key === 'config') config.configFile = value;
  }

  if (config.configFile && fs.existsSync(config.configFile)) {
    const fileConfig = JSON.parse(fs.readFileSync(config.configFile, 'utf8'));
    config.maxAgents = fileConfig.agents || config.maxAgents;
    config.levels = fileConfig.parameters?.levels || config.levels;
  }

  return config;
}

function calculateHierarchy(maxAgents, levels) {
  // Calculate agents per level (geometric distribution)
  const agentsPerLevel = [];
  let remaining = maxAgents - 1; // -1 for coordinator

  for (let i = 0; i < levels - 1; i++) {
    const count = Math.ceil(remaining / (levels - i));
    agentsPerLevel.push(count);
    remaining -= count;
  }

  return {
    coordinator: 1,
    levels: agentsPerLevel,
    total: maxAgents
  };
}

function initHierarchicalTopology(config) {
  console.log('🌳 Initializing Hierarchical Topology...');
  console.log(`   Max Agents: ${config.maxAgents}`);
  console.log(`   Levels: ${config.levels}`);
  console.log(`   Strategy: ${config.strategy}`);
  console.log('');

  const hierarchy = calculateHierarchy(config.maxAgents, config.levels);

  console.log('📊 Hierarchy Structure:');
  console.log(`   Level 0 (Coordinator): ${hierarchy.coordinator} agent`);
  hierarchy.levels.forEach((count, idx) => {
    console.log(`   Level ${idx + 1}: ${count} agents`);
  });
  console.log('');

  try {
    // Initialize swarm with hierarchical topology
    const initCmd = `npx claude-flow@alpha swarm_init --topology hierarchical --maxAgents ${config.maxAgents} --strategy ${config.strategy}`;
    console.log('📡 Executing:', initCmd);
    const output = execSync(initCmd, { encoding: 'utf8' });
    console.log(output);

    // Spawn coordinator first
    console.log('👑 Spawning hierarchical coordinator...');
    const coordCmd = `npx claude-flow@alpha agent_spawn --type coordinator --name hierarchical-coordinator`;
    execSync(coordCmd, { encoding: 'utf8' });

    // Store hierarchical configuration
    const hierConfig = {
      topology: 'hierarchical',
      maxAgents: config.maxAgents,
      strategy: config.strategy,
      levels: config.levels,
      hierarchy: hierarchy,
      timestamp: new Date().toISOString()
    };

    const storeCmd = `npx claude-flow@alpha memory store --key "coordination/hierarchical-config" --value '${JSON.stringify(hierConfig)}'`;
    execSync(storeCmd, { encoding: 'utf8' });

    console.log('');
    console.log('✅ Hierarchical topology initialized successfully!');
    console.log('');
    console.log('Configuration:');
    console.log('   - Structure: Tree-like with parent-child relationships');
    console.log('   - Delegation: Top-down task assignment');
    console.log('   - Coordination: Centralized through coordinator');
    console.log('');
    console.log('Next steps:');
    console.log('   1. Spawn worker agents for each level');
    console.log('   2. Assign tasks through coordinator');
    console.log('   3. Monitor hierarchy: npx claude-flow@alpha swarm_status --verbose');

  } catch (error) {
    console.error('❌ Error initializing hierarchical topology:', error.message);
    process.exit(1);
  }
}

const config = parseArgs();
initHierarchicalTopology(config);
