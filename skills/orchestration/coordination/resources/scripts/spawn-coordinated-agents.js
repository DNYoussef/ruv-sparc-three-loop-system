#!/usr/bin/env node

/**
 * Spawn Coordinated Agents
 *
 * Spawns multiple agents with role assignments for coordinated execution.
 * Supports mesh, hierarchical, and adaptive topologies.
 *
 * Usage:
 *   node spawn-coordinated-agents.js --topology mesh --roles researcher,coder,analyst
 */

const { execSync } = require('child_process');
const fs = require('fs');

function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    topology: 'mesh',
    roles: ['researcher', 'coder', 'analyst'],
    configFile: null
  };

  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace('--', '');
    const value = args[i + 1];

    if (key === 'topology') config.topology = value;
    else if (key === 'roles') config.roles = value.split(',');
    else if (key === 'config') config.configFile = value;
  }

  // Load role definitions from config file
  if (config.configFile && fs.existsSync(config.configFile)) {
    const fileConfig = JSON.parse(fs.readFileSync(config.configFile, 'utf8'));
    config.roles = fileConfig.roles || config.roles;
  }

  return config;
}

// Agent type mapping for claude-flow
const agentTypeMap = {
  'researcher': 'researcher',
  'coder': 'coder',
  'analyst': 'analyst',
  'optimizer': 'optimizer',
  'coordinator': 'coordinator',
  'tester': 'analyst',
  'reviewer': 'analyst',
  'architect': 'researcher'
};

function spawnAgents(config) {
  console.log('🤖 Spawning Coordinated Agents...');
  console.log(`   Topology: ${config.topology}`);
  console.log(`   Roles: ${config.roles.join(', ')}`);
  console.log('');

  const spawnedAgents = [];

  try {
    config.roles.forEach((role, idx) => {
      const agentType = agentTypeMap[role] || 'researcher';
      const agentName = `${role}-${idx + 1}`;

      console.log(`📡 Spawning ${role} agent (${agentName})...`);

      // Spawn agent via MCP
      const spawnCmd = `npx claude-flow@alpha agent_spawn --type ${agentType} --name ${agentName}`;

      try {
        const output = execSync(spawnCmd, { encoding: 'utf8' });
        console.log(`   ✅ ${agentName} spawned successfully`);

        spawnedAgents.push({
          name: agentName,
          role: role,
          type: agentType,
          status: 'active'
        });
      } catch (error) {
        console.warn(`   ⚠️  Warning: Could not spawn ${agentName}: ${error.message}`);
      }
    });

    console.log('');
    console.log(`✅ Spawned ${spawnedAgents.length} agents successfully!`);
    console.log('');

    // Store agent roster in memory
    const rosterConfig = {
      topology: config.topology,
      agents: spawnedAgents,
      timestamp: new Date().toISOString()
    };

    const storeCmd = `npx claude-flow@alpha memory store --key "coordination/agent-roster" --value '${JSON.stringify(rosterConfig)}'`;
    execSync(storeCmd, { encoding: 'utf8' });

    console.log('Agent Roster:');
    spawnedAgents.forEach(agent => {
      console.log(`   - ${agent.name} (${agent.role}) - ${agent.status}`);
    });

    console.log('');
    console.log('Next steps:');
    console.log('   1. Verify agents: npx claude-flow@alpha agent_list');
    console.log('   2. Orchestrate task: node orchestrate-distributed-task.js');
    console.log('   3. Monitor execution: npx claude-flow@alpha swarm_monitor');

  } catch (error) {
    console.error('❌ Error spawning agents:', error.message);
    process.exit(1);
  }
}

const config = parseArgs();
spawnAgents(config);
