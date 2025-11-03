#!/usr/bin/env node

/**
 * Orchestrate Distributed Task
 *
 * Distributes a complex task across coordinated agents.
 * Handles task decomposition, assignment, and monitoring.
 *
 * Usage:
 *   node orchestrate-distributed-task.js --task "Build REST API" --strategy adaptive
 */

const { execSync } = require('child_process');

function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    task: 'Default distributed task',
    strategy: 'adaptive',
    priority: 'medium',
    agents: null
  };

  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace('--', '');
    const value = args[i + 1];

    if (key === 'task') config.task = value;
    else if (key === 'strategy') config.strategy = value;
    else if (key === 'priority') config.priority = value;
    else if (key === 'agents') config.agents = value.split(',');
  }

  return config;
}

function orchestrateTask(config) {
  console.log('🎯 Orchestrating Distributed Task...');
  console.log(`   Task: ${config.task}`);
  console.log(`   Strategy: ${config.strategy}`);
  console.log(`   Priority: ${config.priority}`);
  console.log('');

  try {
    // Retrieve agent roster from memory
    let agentRoster = [];
    try {
      const retrieveCmd = `npx claude-flow@alpha memory retrieve --key "coordination/agent-roster"`;
      const rosterData = execSync(retrieveCmd, { encoding: 'utf8' });
      const parsed = JSON.parse(rosterData);
      agentRoster = parsed.agents || [];
    } catch (error) {
      console.warn('⚠️  No agent roster found in memory. Using default agents.');
      agentRoster = [
        { name: 'researcher-1', role: 'researcher', type: 'researcher' },
        { name: 'coder-1', role: 'coder', type: 'coder' },
        { name: 'analyst-1', role: 'analyst', type: 'analyst' }
      ];
    }

    console.log(`📋 Available Agents: ${agentRoster.length}`);
    agentRoster.forEach(agent => {
      console.log(`   - ${agent.name} (${agent.role})`);
    });
    console.log('');

    // Task orchestration via MCP
    console.log('📡 Orchestrating task across swarm...');
    const orchestrateCmd = `npx claude-flow@alpha task_orchestrate --task "${config.task}" --strategy ${config.strategy} --priority ${config.priority}`;

    const output = execSync(orchestrateCmd, { encoding: 'utf8' });
    console.log(output);

    // Store task metadata
    const taskMetadata = {
      task: config.task,
      strategy: config.strategy,
      priority: config.priority,
      agents: agentRoster.map(a => a.name),
      timestamp: new Date().toISOString(),
      status: 'in_progress'
    };

    const storeCmd = `npx claude-flow@alpha memory store --key "coordination/current-task" --value '${JSON.stringify(taskMetadata)}'`;
    execSync(storeCmd, { encoding: 'utf8' });

    console.log('');
    console.log('✅ Task orchestration initiated successfully!');
    console.log('');
    console.log('Monitoring:');
    console.log('   - Track status: npx claude-flow@alpha task_status');
    console.log('   - View results: npx claude-flow@alpha task_results --taskId <id>');
    console.log('   - Monitor swarm: npx claude-flow@alpha swarm_monitor --interval 10');

  } catch (error) {
    console.error('❌ Error orchestrating task:', error.message);
    process.exit(1);
  }
}

const config = parseArgs();
orchestrateTask(config);
