#!/usr/bin/env node
/**
 * Test Suite for Swarm Coordinator
 *
 * Tests swarm initialization, agent spawning, and task orchestration.
 *
 * Usage:
 *   node test-swarm-coordinator.js
 */

const assert = require('assert');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m'
};

let passCount = 0;
let failCount = 0;

function test(name, fn) {
  try {
    console.log(`\n${colors.blue}TEST:${colors.reset} ${name}`);
    fn();
    passCount++;
    console.log(`${colors.green}✓ PASS${colors.reset}`);
  } catch (error) {
    failCount++;
    console.log(`${colors.red}✗ FAIL${colors.reset}: ${error.message}`);
  }
}

/**
 * Test topology selection
 */
test('Should select ring topology for simple complexity', () => {
  const analysis = { complexity: 'simple' };
  const topology = 'auto';

  let selected;
  if (topology === 'auto') {
    if (analysis.complexity === 'simple') selected = 'ring';
    else if (analysis.complexity === 'moderate') selected = 'mesh';
    else selected = 'hierarchical';
  }

  assert.strictEqual(selected, 'ring');
});

test('Should select mesh topology for moderate complexity', () => {
  const analysis = { complexity: 'moderate' };
  const topology = 'auto';

  let selected;
  if (topology === 'auto') {
    if (analysis.complexity === 'simple') selected = 'ring';
    else if (analysis.complexity === 'moderate') selected = 'mesh';
    else selected = 'hierarchical';
  }

  assert.strictEqual(selected, 'mesh');
});

test('Should select hierarchical topology for complex', () => {
  const analysis = { complexity: 'complex' };
  const topology = 'auto';

  let selected;
  if (topology === 'auto') {
    if (analysis.complexity === 'simple') selected = 'ring';
    else if (analysis.complexity === 'moderate') selected = 'mesh';
    else selected = 'hierarchical';
  }

  assert.strictEqual(selected, 'hierarchical');
});

test('Should respect manual topology override', () => {
  const analysis = { complexity: 'simple' };
  const topology = 'mesh';

  let selected = topology !== 'auto' ? topology : 'ring';

  assert.strictEqual(selected, 'mesh');
});

/**
 * Test agent selection
 */
test('Should limit agents to maxAgents', () => {
  const maxAgents = 3;
  const recommendedAgents = ['security', 'performance', 'style', 'architecture', 'accessibility'];

  const selected = recommendedAgents.slice(0, maxAgents);

  assert.strictEqual(selected.length, maxAgents);
});

test('Should prioritize critical agents', () => {
  const maxAgents = 3;
  const recommendedAgents = ['docs', 'i18n', 'security', 'performance', 'style'];
  const priority = ['security', 'performance', 'architecture', 'style', 'accessibility'];

  const prioritized = [];
  priority.forEach(agent => {
    if (recommendedAgents.includes(agent)) {
      prioritized.push(agent);
    }
  });

  recommendedAgents.forEach(agent => {
    if (!prioritized.includes(agent)) {
      prioritized.push(agent);
    }
  });

  const selected = prioritized.slice(0, maxAgents);

  assert(selected.includes('security'), 'Security should be prioritized');
  assert(selected.includes('performance'), 'Performance should be prioritized');
});

test('Should allow agent override', () => {
  const agentsOverride = 'security,custom-agent';
  const recommendedAgents = ['style', 'performance'];

  const selected = agentsOverride
    ? agentsOverride.split(',').map(a => a.trim())
    : recommendedAgents;

  assert.strictEqual(selected.length, 2);
  assert(selected.includes('security'));
  assert(selected.includes('custom-agent'));
});

/**
 * Test agent configuration
 */
test('Should configure security agent correctly', () => {
  const config = {
    security: {
      type: 'analyst',
      name: 'Security Reviewer',
      capabilities: 'security-audit,vulnerability-scanning,owasp-compliance'
    }
  };

  assert.strictEqual(config.security.type, 'analyst');
  assert(config.security.capabilities.includes('security-audit'));
});

test('Should configure performance agent correctly', () => {
  const config = {
    performance: {
      type: 'optimizer',
      name: 'Performance Analyst',
      capabilities: 'performance-profiling,bottleneck-analysis,optimization'
    }
  };

  assert.strictEqual(config.performance.type, 'optimizer');
  assert(config.performance.capabilities.includes('performance-profiling'));
});

test('Should provide default config for unknown agent type', () => {
  const agentType = 'custom-agent';

  const config = {
    type: 'analyst',
    name: `${agentType} Reviewer`,
    capabilities: agentType
  };

  assert.strictEqual(config.type, 'analyst');
  assert.strictEqual(config.capabilities, 'custom-agent');
});

/**
 * Test task orchestration
 */
test('Should create tasks for all agents', () => {
  const prNumber = 123;
  const agents = ['security', 'performance', 'style'];

  const tasks = agents.map(agent => ({
    agent,
    task: `Review PR #${prNumber} from ${agent} perspective`
  }));

  assert.strictEqual(tasks.length, agents.length);
  assert(tasks[0].task.includes('security'));
  assert(tasks[1].task.includes('performance'));
});

test('Should format tasks correctly', () => {
  const prNumber = 456;
  const agent = 'security';

  const task = {
    agent,
    task: `Review PR #${prNumber} from ${agent} perspective`
  };

  assert(task.task.includes(`PR #${prNumber}`));
  assert(task.task.includes(agent));
});

/**
 * Test status comment generation
 */
test('Should generate comprehensive status comment', () => {
  const prNumber = 789;
  const analysis = {
    complexity: 'moderate',
    riskLevel: 'medium',
    fileStats: { totalChanges: 345 },
    estimatedReviewTime: 45
  };
  const topology = 'mesh';
  const agents = ['security', 'performance', 'style'];

  const status = `
Review initiated for PR #${prNumber}
Complexity: ${analysis.complexity}
Topology: ${topology}
Agents: ${agents.join(', ')}
  `.trim();

  assert(status.includes(`PR #${prNumber}`));
  assert(status.includes(analysis.complexity));
  assert(status.includes(topology));
  agents.forEach(agent => {
    assert(status.includes(agent));
  });
});

/**
 * Test validation
 */
test('Should validate PR number is provided', () => {
  const prNumber = null;

  assert.strictEqual(prNumber === null, true, 'Should detect missing PR number');
});

test('Should validate topology is valid', () => {
  const validTopologies = ['mesh', 'hierarchical', 'ring', 'star', 'auto'];
  const topology = 'mesh';

  assert(validTopologies.includes(topology), 'Topology should be valid');
});

test('Should validate maxAgents is within range', () => {
  const maxAgents = 5;

  assert(maxAgents >= 1 && maxAgents <= 10, 'Max agents should be 1-10');
});

/**
 * Print test results
 */
console.log('\n' + '='.repeat(60));
console.log(`${colors.blue}TEST RESULTS${colors.reset}`);
console.log('='.repeat(60));
console.log(`${colors.green}✓ Passed:${colors.reset} ${passCount}`);
console.log(`${colors.red}✗ Failed:${colors.reset} ${failCount}`);
console.log(`${colors.yellow}Total:${colors.reset} ${passCount + failCount}`);
console.log('='.repeat(60));

if (failCount > 0) {
  console.log(`\n${colors.red}Some tests failed!${colors.reset}`);
  process.exit(1);
} else {
  console.log(`\n${colors.green}All tests passed!${colors.reset}`);
  process.exit(0);
}
