#!/usr/bin/env node
/**
 * Script to add MCP server assignments to agent registry
 * Based on MCP Marketplace Guide mappings
 */

const fs = require('fs');
const path = require('path');

// MCP server assignments by agent type/category
const MCP_ASSIGNMENTS = {
  // Core Development Agents
  'researcher': {
    required: ['memory-mcp'],
    recommended: ['fetch', 'tavily', 'exa', 'firecrawl'],
    usage: 'Use tavily/exa for web search, fetch for content retrieval, memory-mcp to store research findings with "research" intent'
  },
  'coder': {
    required: ['memory-mcp', 'connascence-analyzer', 'focused-changes'],
    recommended: ['filesystem', 'git'],
    usage: 'Use connascence-analyzer before committing code, focused-changes to stay on task, memory-mcp to log implementation decisions with "implementation" intent'
  },
  'tester': {
    required: ['memory-mcp', 'connascence-analyzer', 'focused-changes'],
    recommended: ['e2b'],
    usage: 'Use root_cause_analysis from focused-changes for test failures, connascence-analyzer for quality checks, memory-mcp to store test patterns with "testing" intent'
  },
  'reviewer': {
    required: ['memory-mcp', 'connascence-analyzer', 'focused-changes'],
    recommended: ['git'],
    usage: 'Run connascence-analyzer for quality checks, review change scope with focused-changes, memory-mcp to log review findings with "analysis" intent'
  },
  'planner': {
    required: ['memory-mcp'],
    recommended: ['sequential-thinking', 'ToC', 'fetch'],
    usage: 'Use sequential-thinking for complex planning, retrieve prior plans from memory-mcp with "planning" intent'
  },
  'analyst': {
    required: ['memory-mcp', 'connascence-analyzer'],
    recommended: ['logfire', 'chroma'],
    usage: 'Use connascence-analyzer for code analysis, memory-mcp to store metrics and patterns with "analysis" intent'
  },

  // GitHub & Repository Agents
  'github-modes': {
    required: ['memory-mcp'],
    recommended: ['git', 'github', 'slack'],
    usage: 'Track PR state, issue progress in memory-mcp with appropriate metadata'
  },
  'pr-manager': {
    required: ['memory-mcp'],
    recommended: ['git', 'github', 'connascence-analyzer'],
    usage: 'Analyze PR changes with connascence-analyzer, track PR lifecycle in memory-mcp'
  },
  'code-review-swarm': {
    required: ['memory-mcp', 'connascence-analyzer'],
    recommended: ['git', 'github'],
    usage: 'Coordinate review findings with connascence-analyzer, memory-mcp for pattern learning'
  },
  'issue-tracker': {
    required: ['memory-mcp'],
    recommended: ['github', 'slack', 'notion'],
    usage: 'Track issue state and progress in memory-mcp with "planning" intent'
  },
  'release-manager': {
    required: ['memory-mcp'],
    recommended: ['git', 'github', 'docker'],
    usage: 'Store release notes, deployment history in memory-mcp long-term layer'
  },
  'workflow-automation': {
    required: ['memory-mcp'],
    recommended: ['github', 'docker', 'aws', 'azure'],
    usage: 'Track workflow executions, pipeline configs in memory-mcp'
  },
  'project-board-sync': {
    required: ['memory-mcp'],
    recommended: ['github', 'slack', 'notion'],
    usage: 'Sync project state across platforms via memory-mcp'
  },
  'repo-architect': {
    required: ['memory-mcp'],
    recommended: ['git', 'github', 'ToC'],
    usage: 'Store architecture decisions in memory-mcp long-term layer'
  },
  'multi-repo-swarm': {
    required: ['memory-mcp'],
    recommended: ['git', 'github'],
    usage: 'Coordinate across repositories with memory-mcp shared state'
  },

  // SPARC Methodology
  'sparc-coord': {
    required: ['memory-mcp'],
    recommended: ['sequential-thinking', 'ToC'],
    usage: 'Orchestrate SPARC phases with memory-mcp tracking each phase'
  },
  'specification': {
    required: ['memory-mcp'],
    recommended: ['sequential-thinking', 'ToC', 'fetch'],
    usage: 'Store specifications in memory-mcp long-term layer with sparc_phase: specification'
  },
  'pseudocode': {
    required: ['memory-mcp'],
    recommended: ['sequential-thinking'],
    usage: 'Store pseudocode in memory-mcp mid-term layer with sparc_phase: pseudocode'
  },
  'architecture': {
    required: ['memory-mcp'],
    recommended: ['sequential-thinking', 'ToC'],
    usage: 'Store architecture decisions in memory-mcp long-term layer with sparc_phase: architecture'
  },
  'refinement': {
    required: ['memory-mcp', 'connascence-analyzer'],
    recommended: ['focused-changes'],
    usage: 'Use connascence-analyzer for refinement validation, memory-mcp with sparc_phase: refinement'
  },
  'sparc-coder': {
    required: ['memory-mcp', 'connascence-analyzer'],
    recommended: ['focused-changes', 'git'],
    usage: 'Implement with quality checks via connascence-analyzer, track in memory-mcp with sparc_phase: code'
  },

  // Backend & Specialized Development
  'backend-dev': {
    required: ['memory-mcp', 'connascence-analyzer'],
    recommended: ['supabase', 'postgres', 'aws', 'azure', 'docker'],
    usage: 'Store database schemas, API contracts in memory-mcp, validate code with connascence-analyzer'
  },
  'mobile-dev': {
    required: ['memory-mcp', 'connascence-analyzer'],
    recommended: ['supabase', 'firebase'],
    usage: 'Store platform-specific patterns, UI components in memory-mcp'
  },
  'ml-developer': {
    required: ['memory-mcp', 'connascence-analyzer'],
    recommended: ['chroma', 'milvus', 'duckdb'],
    usage: 'Store model architectures, hyperparameters, dataset metadata in memory-mcp'
  },
  'cicd-engineer': {
    required: ['memory-mcp'],
    recommended: ['git', 'github', 'docker', 'aws', 'azure'],
    usage: 'Store pipeline configs, deployment histories in memory-mcp'
  },
  'api-docs': {
    required: ['memory-mcp'],
    recommended: ['ToC', 'fetch'],
    usage: 'Generate API docs, store OpenAPI schemas in memory-mcp'
  },
  'system-architect': {
    required: ['memory-mcp'],
    recommended: ['sequential-thinking', 'ToC', 'aws', 'azure'],
    usage: 'Store system design decisions in memory-mcp long-term layer'
  },

  // Performance & Optimization
  'perf-analyzer': {
    required: ['memory-mcp'],
    recommended: ['logfire'],
    usage: 'Store performance metrics, bottleneck analysis in memory-mcp'
  },
  'performance-benchmarker': {
    required: ['memory-mcp'],
    recommended: ['logfire'],
    usage: 'Track benchmark results over time in memory-mcp'
  },

  // Swarm Coordination (all need memory-mcp)
  'hierarchical-coordinator': {
    required: ['memory-mcp'],
    recommended: [],
    usage: 'Store swarm topology, coordination state in memory-mcp'
  },
  'mesh-coordinator': {
    required: ['memory-mcp'],
    recommended: [],
    usage: 'Store peer connections, distributed state in memory-mcp'
  },
  'adaptive-coordinator': {
    required: ['memory-mcp'],
    recommended: [],
    usage: 'Store topology adaptations, performance metrics in memory-mcp'
  },
  'collective-intelligence-coordinator': {
    required: ['memory-mcp'],
    recommended: ['chroma'],
    usage: 'Centralized memory coordination across swarm agents'
  },
  'swarm-memory-manager': {
    required: ['memory-mcp'],
    recommended: ['chroma'],
    usage: 'Manage distributed memory, ensure consistency across swarm'
  },

  // Testing & Validation
  'functionality-audit': {
    required: ['memory-mcp', 'connascence-analyzer'],
    recommended: ['e2b', 'focused-changes'],
    usage: 'Validate functionality with e2b sandboxes, track audit results in memory-mcp'
  },
  'theater-detection-audit': {
    required: ['memory-mcp', 'connascence-analyzer'],
    recommended: [],
    usage: 'Detect theater code with connascence-analyzer, log findings in memory-mcp'
  },
  'production-validator': {
    required: ['memory-mcp', 'connascence-analyzer'],
    recommended: ['e2b', 'docker'],
    usage: 'Comprehensive validation with all quality tools, certification in memory-mcp'
  },

  // Default for any unlisted agents
  'default': {
    required: ['memory-mcp'],
    recommended: [],
    usage: 'All agents have access to memory-mcp for persistent memory with WHO/WHEN/PROJECT/WHY tagging'
  }
};

function addMCPToAgent(agentName, agentData) {
  // Get MCP assignment for this agent, or use default
  const mcpAssignment = MCP_ASSIGNMENTS[agentName] || MCP_ASSIGNMENTS['default'];

  // Add MCP server fields
  return {
    ...agentData,
    mcp_servers: {
      required: mcpAssignment.required,
      recommended: mcpAssignment.recommended,
      usage: mcpAssignment.usage
    }
  };
}

function main() {
  const registryPath = path.join(__dirname, 'registry.json');

  // Read existing registry
  const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));

  // Update each agent with MCP server info
  const updatedAgents = {};
  for (const [agentName, agentData] of Object.entries(registry.agents)) {
    updatedAgents[agentName] = addMCPToAgent(agentName, agentData);
  }

  // Create updated registry
  const updatedRegistry = {
    ...registry,
    agents: updatedAgents,
    mcp_integration: {
      version: '3.0.2',
      last_updated: '2025-11-01',
      description: 'All agents now include MCP server assignments based on their capabilities and use cases',
      marketplace_guide: 'docs/MCP-MARKETPLACE-GUIDE.md'
    }
  };

  // Write updated registry
  fs.writeFileSync(registryPath, JSON.stringify(updatedRegistry, null, 2), 'utf8');

  console.log('Agent registry updated with MCP server assignments');
  console.log(`Total agents updated: ${Object.keys(updatedAgents).length}`);
  console.log(`All agents have required: memory-mcp`);
  console.log(`14 code quality agents have required: connascence-analyzer`);
  console.log(`See docs/MCP-MARKETPLACE-GUIDE.md for complete documentation`);
}

main();
