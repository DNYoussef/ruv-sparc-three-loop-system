#!/usr/bin/env node
/**
 * Update agent registry to use ONLY actually installed MCP servers
 * Removes all references to non-existing servers
 */

const fs = require('fs');
const path = require('path');

// ONLY actually installed and working MCP servers
const INSTALLED_SERVERS = {
  'memory-mcp': {
    description: 'Persistent cross-session memory with triple-layer retention (24h/7d/30d+)',
    tools: ['vector_search', 'memory_store'],
    usage: 'Store and retrieve information with WHO/WHEN/PROJECT/WHY tagging'
  },
  'connascence-analyzer': {
    description: 'Code quality analysis (9 connascence types, 7+ violations, NASA compliance)',
    tools: ['analyze_file', 'analyze_workspace', 'health_check'],
    usage: 'Run before committing code to detect God Objects, Parameter Bombs, complexity issues'
  },
  'focused-changes': {
    description: 'Track file changes, ensure focused scope, build error trees',
    tools: ['start_tracking', 'analyze_changes', 'root_cause_analysis'],
    usage: 'Track changes to stay on task, analyze test failures for root causes'
  },
  'ToC': {
    description: 'Generate table of contents for documentation',
    tools: ['generate_toc', 'get_file_description'],
    usage: 'Generate hierarchical TOC for Python, Markdown, JSON, YAML files'
  }
};

// Agent-specific MCP server assignments with detailed usage instructions
const AGENT_MCP_ASSIGNMENTS = {
  // Code Quality Agents - Get all tools
  'coder': {
    required: ['memory-mcp', 'connascence-analyzer', 'focused-changes'],
    usage: 'WORKFLOW:\n1. Start tracking: focused-changes.start_tracking(file)\n2. Write code\n3. Analyze quality: connascence-analyzer.analyze_file(file)\n4. Fix violations before committing\n5. Store implementation decisions: memory-mcp.memory_store(decision, {intent: "implementation"})\n6. Verify changes stayed focused: focused-changes.analyze_changes(newContent)'
  },
  'reviewer': {
    required: ['memory-mcp', 'connascence-analyzer', 'focused-changes'],
    usage: 'WORKFLOW:\n1. Run quality check: connascence-analyzer.analyze_workspace(files)\n2. Check change scope: focused-changes.analyze_changes(changes)\n3. Store review findings: memory-mcp.memory_store(findings, {intent: "analysis"})\n4. Verify 0 violations before approval'
  },
  'tester': {
    required: ['memory-mcp', 'connascence-analyzer', 'focused-changes'],
    usage: 'WORKFLOW:\n1. If tests fail: focused-changes.root_cause_analysis(testResults)\n2. Analyze test code quality: connascence-analyzer.analyze_file(test_file)\n3. Store test patterns: memory-mcp.memory_store(pattern, {intent: "testing"})\n4. Track test coverage trends in memory'
  },
  'sparc-coder': {
    required: ['memory-mcp', 'connascence-analyzer', 'focused-changes'],
    usage: 'SPARC WORKFLOW:\n1. Retrieve spec from memory: memory-mcp.vector_search("specification for...")\n2. Start tracking: focused-changes.start_tracking(file)\n3. Implement code\n4. Validate quality: connascence-analyzer.analyze_file(file)\n5. Store implementation: memory-mcp.memory_store(code, {sparc_phase: "code"})'
  },

  // All code quality agents get same setup
  'functionality-audit': {
    required: ['memory-mcp', 'connascence-analyzer', 'focused-changes'],
    usage: 'Validate functionality, run connascence checks, store audit results in memory'
  },
  'theater-detection-audit': {
    required: ['memory-mcp', 'connascence-analyzer'],
    usage: 'Detect theater code with connascence-analyzer, log findings in memory'
  },
  'production-validator': {
    required: ['memory-mcp', 'connascence-analyzer'],
    usage: 'Comprehensive validation with all quality tools, certification in memory'
  },
  'code-analyzer': {
    required: ['memory-mcp', 'connascence-analyzer'],
    usage: 'Analyze code patterns, store metrics in memory for trend analysis'
  },
  'analyst': {
    required: ['memory-mcp', 'connascence-analyzer'],
    usage: 'Deep code analysis, pattern detection, store insights in memory'
  },
  'backend-dev': {
    required: ['memory-mcp', 'connascence-analyzer'],
    usage: 'Store API schemas, database designs in memory (long-term layer)'
  },
  'mobile-dev': {
    required: ['memory-mcp', 'connascence-analyzer'],
    usage: 'Store platform-specific patterns, UI components in memory'
  },
  'ml-developer': {
    required: ['memory-mcp', 'connascence-analyzer'],
    usage: 'Store model architectures, hyperparameters in memory'
  },
  'base-template-generator': {
    required: ['memory-mcp', 'connascence-analyzer'],
    usage: 'Validate generated templates with connascence, store patterns in memory'
  },
  'code-review-swarm': {
    required: ['memory-mcp', 'connascence-analyzer'],
    usage: 'Coordinate review findings, learn patterns from past reviews'
  },

  // Research & Planning - Memory only (no code quality tools)
  'researcher': {
    required: ['memory-mcp'],
    usage: 'RESEARCH WORKFLOW:\n1. Search prior research: memory-mcp.vector_search(topic)\n2. Conduct new research\n3. Store findings: memory-mcp.memory_store(findings, {intent: "research", layer: "long_term"})\n4. Tag with keywords for future retrieval'
  },
  'planner': {
    required: ['memory-mcp'],
    usage: 'PLANNING WORKFLOW:\n1. Retrieve prior plans: memory-mcp.vector_search("plan for...")\n2. Create new plan\n3. Store plan: memory-mcp.memory_store(plan, {intent: "planning", layer: "mid_term"})\n4. Link to related specifications'
  },

  // SPARC Methodology
  'specification': {
    required: ['memory-mcp'],
    usage: 'Store specifications in memory long-term layer with sparc_phase: "specification"'
  },
  'pseudocode': {
    required: ['memory-mcp'],
    usage: 'Store pseudocode in memory mid-term layer with sparc_phase: "pseudocode"'
  },
  'architecture': {
    required: ['memory-mcp'],
    usage: 'Store architecture decisions in memory long-term layer with sparc_phase: "architecture"'
  },
  'refinement': {
    required: ['memory-mcp', 'connascence-analyzer'],
    usage: 'Use connascence-analyzer for refinement validation, store in memory with sparc_phase: "refinement"'
  },

  // Documentation
  'api-docs': {
    required: ['memory-mcp', 'ToC'],
    usage: 'Generate API docs, use ToC for structure, store OpenAPI schemas in memory'
  },

  // DEFAULT for unlisted agents
  'default': {
    required: ['memory-mcp'],
    usage: 'All agents have memory-mcp access with WHO/WHEN/PROJECT/WHY tagging'
  }
};

function getAgentMCPConfig(agentName) {
  const config = AGENT_MCP_ASSIGNMENTS[agentName] || AGENT_MCP_ASSIGNMENTS['default'];

  return {
    mcp_servers: {
      required: config.required,
      recommended: [], // No recommended - only installed servers
      usage: config.usage,
      installed_servers: INSTALLED_SERVERS
    }
  };
}

function main() {
  const registryPath = path.join(__dirname, 'registry.json');
  const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));

  console.log('Updating agent registry to use ONLY installed MCP servers...\n');
  console.log(`Installed servers: ${Object.keys(INSTALLED_SERVERS).join(', ')}\n`);

  const updatedAgents = {};
  for (const [agentName, agentData] of Object.entries(registry.agents)) {
    const mcpConfig = getAgentMCPConfig(agentName);

    updatedAgents[agentName] = {
      ...agentData,
      ...mcpConfig
    };

    console.log(`Updated agent: ${agentName}`);
    console.log(`  Required servers: ${mcpConfig.mcp_servers.required.join(', ')}`);
  }

  const updatedRegistry = {
    ...registry,
    agents: updatedAgents,
    mcp_integration: {
      version: '3.0.4',
      last_updated: '2025-11-01',
      description: 'All agents use ONLY installed MCP servers (4 total)',
      installed_servers_only: true,
      installed_servers: Object.keys(INSTALLED_SERVERS),
      removed_all_non_existing: true,
      marketplace_guide: 'docs/MCP-MARKETPLACE-GUIDE.md',
      installed_guide: 'docs/INSTALLED-MCP-SERVERS.md'
    }
  };

  fs.writeFileSync(registryPath, JSON.stringify(updatedRegistry, null, 2), 'utf8');

  console.log('\n=== UPDATE COMPLETE ===');
  console.log(`Total agents updated: ${Object.keys(updatedAgents).length}`);
  console.log(`Installed servers: ${Object.keys(INSTALLED_SERVERS).length}`);
  console.log('All references to non-existing servers removed');
  console.log('All agents now have specific usage instructions for their MCP servers');
}

main();
