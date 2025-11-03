#!/usr/bin/env node
/**
 * GitHub PR Review Swarm Coordinator
 *
 * Orchestrates multi-agent code review swarms for GitHub pull requests.
 * Spawns specialized review agents based on PR characteristics.
 *
 * Usage:
 *   node swarm-coordinator.js <pr-number> [options]
 *   node swarm-coordinator.js 123 --agents security,performance,style
 *   node swarm-coordinator.js 123 --topology hierarchical --max-agents 6
 */

const { execSync } = require('child_process');
const path = require('path');

// Parse command line arguments
const args = process.argv.slice(2);
const prNumber = args[0];
const topology = getArg('--topology') || 'auto';
const maxAgents = parseInt(getArg('--max-agents') || '5');
const agentsOverride = getArg('--agents');
const dryRun = args.includes('--dry-run');
const verbose = args.includes('--verbose');

if (!prNumber || prNumber === '--help') {
  console.log('Usage: node swarm-coordinator.js <pr-number> [options]');
  console.log('');
  console.log('Options:');
  console.log('  --topology <type>       Swarm topology (mesh|hierarchical|ring|auto)');
  console.log('  --max-agents <num>      Maximum number of agents (default: 5)');
  console.log('  --agents <list>         Override agent selection (comma-separated)');
  console.log('  --dry-run               Show plan without executing');
  console.log('  --verbose               Show detailed execution logs');
  process.exit(prNumber ? 0 : 1);
}

/**
 * Get command line argument value
 */
function getArg(flag) {
  const idx = args.indexOf(flag);
  return idx !== -1 && idx + 1 < args.length ? args[idx + 1] : null;
}

/**
 * Execute command and handle errors
 */
function exec(command, options = {}) {
  try {
    if (verbose) console.log(`[EXEC] ${command}`);
    return execSync(command, { encoding: 'utf-8', ...options });
  } catch (error) {
    console.error(`Error executing command: ${error.message}`);
    if (!options.ignoreErrors) process.exit(1);
    return null;
  }
}

/**
 * Analyze PR using pr-analysis.js script
 */
function analyzePR(prNumber) {
  const scriptPath = path.join(__dirname, 'pr-analysis.js');
  const output = exec(`node "${scriptPath}" ${prNumber} --json`);
  return JSON.parse(output);
}

/**
 * Determine optimal topology based on PR complexity
 */
function determineTopology(analysis) {
  if (topology !== 'auto') return topology;

  if (analysis.complexity === 'simple') return 'ring';
  if (analysis.complexity === 'moderate') return 'mesh';
  return 'hierarchical';
}

/**
 * Select agents based on PR analysis
 */
function selectAgents(analysis) {
  if (agentsOverride) {
    return agentsOverride.split(',').map(a => a.trim());
  }

  let agents = analysis.recommendedAgents || [];

  // Limit to maxAgents, prioritizing critical agents
  const priority = ['security', 'performance', 'architecture', 'style', 'accessibility'];
  const prioritized = [];

  // Add priority agents first
  priority.forEach(agent => {
    if (agents.includes(agent)) {
      prioritized.push(agent);
    }
  });

  // Add remaining agents
  agents.forEach(agent => {
    if (!prioritized.includes(agent)) {
      prioritized.push(agent);
    }
  });

  return prioritized.slice(0, maxAgents);
}

/**
 * Initialize swarm with specified topology
 */
function initializeSwarm(swarmTopology, numAgents) {
  console.log(`\n🚀 Initializing ${swarmTopology} swarm with ${numAgents} agents...\n`);

  if (!dryRun) {
    exec(`npx ruv-swarm swarm init --topology ${swarmTopology} --max-agents ${numAgents}`);
  }
}

/**
 * Spawn review agent with specific role
 */
function spawnAgent(agentType, prNumber) {
  console.log(`  🤖 Spawning ${agentType} agent...`);

  if (!dryRun) {
    const agentConfig = getAgentConfig(agentType);
    exec(`npx ruv-swarm agent spawn --type ${agentConfig.type} --name "${agentConfig.name}" --capabilities "${agentConfig.capabilities}"`);
  }
}

/**
 * Get agent configuration based on review type
 */
function getAgentConfig(agentType) {
  const configs = {
    security: {
      type: 'analyst',
      name: 'Security Reviewer',
      capabilities: 'security-audit,vulnerability-scanning,owasp-compliance'
    },
    performance: {
      type: 'optimizer',
      name: 'Performance Analyst',
      capabilities: 'performance-profiling,bottleneck-analysis,optimization'
    },
    architecture: {
      type: 'analyst',
      name: 'Architecture Reviewer',
      capabilities: 'design-patterns,solid-principles,coupling-analysis'
    },
    style: {
      type: 'analyst',
      name: 'Style Reviewer',
      capabilities: 'code-style,linting,formatting,conventions'
    },
    accessibility: {
      type: 'analyst',
      name: 'Accessibility Specialist',
      capabilities: 'wcag-compliance,a11y-testing,aria-validation'
    },
    database: {
      type: 'optimizer',
      name: 'Database Specialist',
      capabilities: 'query-optimization,schema-design,indexing'
    },
    tester: {
      type: 'analyst',
      name: 'Test Reviewer',
      capabilities: 'test-coverage,test-quality,edge-cases'
    },
    docs: {
      type: 'researcher',
      name: 'Documentation Reviewer',
      capabilities: 'documentation-quality,api-docs,readme'
    }
  };

  return configs[agentType] || {
    type: 'analyst',
    name: `${agentType} Reviewer`,
    capabilities: agentType
  };
}

/**
 * Orchestrate review tasks across agents
 */
function orchestrateReview(prNumber, agents) {
  console.log(`\n📋 Orchestrating review tasks for PR #${prNumber}...\n`);

  const tasks = agents.map(agent => ({
    agent,
    task: `Review PR #${prNumber} from ${agent} perspective`
  }));

  tasks.forEach(({ agent, task }) => {
    console.log(`  📝 Assigning ${agent} review task...`);

    if (!dryRun) {
      exec(`npx ruv-swarm task assign --agent "${getAgentConfig(agent).name}" --task "${task}"`);
    }
  });
}

/**
 * Post initial status to PR
 */
function postStatusToPR(prNumber, analysis, swarmTopology, agents) {
  const statusComment = `
🔍 **Multi-Agent Code Review Initiated**

**PR Analysis:**
- Complexity: ${analysis.complexity}
- Risk Level: ${analysis.riskLevel}
- Total Changes: ${analysis.fileStats.totalChanges} lines

**Review Configuration:**
- Topology: ${swarmTopology}
- Review Agents: ${agents.join(', ')}
- Estimated Time: ${analysis.estimatedReviewTime} minutes

**Status:** In Progress 🚧

---
*Automated review powered by RUV Swarm*
`;

  console.log(`\n💬 Posting status to PR #${prNumber}...\n`);

  if (!dryRun) {
    // Escape quotes for shell
    const escapedComment = statusComment.replace(/"/g, '\\"').replace(/\n/g, '\\n');
    exec(`gh pr comment ${prNumber} --body "${escapedComment}"`);
  } else {
    console.log(statusComment);
  }
}

/**
 * Monitor swarm progress
 */
function monitorProgress(prNumber) {
  console.log(`\n📊 Monitoring review progress...\n`);

  if (!dryRun) {
    // Run swarm monitor in background
    exec(`npx ruv-swarm swarm monitor --duration 300 --interval 30 &`, { ignoreErrors: true });
  }

  console.log('  ✓ Progress monitoring started (background)');
}

// Main execution
try {
  console.log(`\n╔════════════════════════════════════════════════════════╗`);
  console.log(`║  GitHub PR Review Swarm Coordinator                    ║`);
  console.log(`╚════════════════════════════════════════════════════════╝\n`);

  if (dryRun) {
    console.log('🔍 DRY RUN MODE - No changes will be made\n');
  }

  // Step 1: Analyze PR
  console.log(`📊 Analyzing PR #${prNumber}...\n`);
  const analysis = analyzePR(prNumber);

  console.log(`  Complexity: ${analysis.complexity}`);
  console.log(`  Risk Level: ${analysis.riskLevel}`);
  console.log(`  Files Changed: ${analysis.fileStats.total}`);
  console.log(`  Total Changes: ${analysis.fileStats.totalChanges} lines`);

  // Step 2: Determine topology
  const swarmTopology = determineTopology(analysis);
  console.log(`\n🏗️  Selected Topology: ${swarmTopology}`);

  // Step 3: Select agents
  const selectedAgents = selectAgents(analysis);
  console.log(`\n👥 Selected Agents (${selectedAgents.length}):`);
  selectedAgents.forEach(agent => console.log(`  - ${agent}`));

  // Step 4: Initialize swarm
  initializeSwarm(swarmTopology, selectedAgents.length);

  // Step 5: Spawn agents
  console.log(`\n🤖 Spawning Review Agents:\n`);
  selectedAgents.forEach(agent => spawnAgent(agent, prNumber));

  // Step 6: Orchestrate review
  orchestrateReview(prNumber, selectedAgents);

  // Step 7: Post status to PR
  postStatusToPR(prNumber, analysis, swarmTopology, selectedAgents);

  // Step 8: Monitor progress
  monitorProgress(prNumber);

  console.log(`\n✅ Review swarm successfully initiated for PR #${prNumber}\n`);
  console.log(`📈 Use 'npx ruv-swarm swarm-status' to check progress\n`);

} catch (error) {
  console.error(`\n❌ Error coordinating swarm: ${error.message}\n`);
  process.exit(1);
}
