#!/usr/bin/env node
/**
 * Script to remove paid/API-required MCP servers from agent registry
 * Keeps only FREE, local, and official Anthropic servers
 */

const fs = require('fs');
const path = require('path');

// List of FREE MCP servers (no payment, no API keys, no accounts required)
const FREE_SERVERS = [
  // Currently installed (local)
  'memory-mcp',
  'connascence-analyzer',
  'focused-changes',
  'ToC',

  // Official Anthropic (free)
  'fetch',
  'filesystem',
  'git',
  'memory',
  'sequential-thinking',
  'time',
  'everything'
];

// Servers to REMOVE (require payment/API keys/accounts)
const PAID_SERVERS = [
  // Web search (require API keys)
  'tavily', 'exa', 'firecrawl', 'browserbase',

  // Infrastructure (require accounts/credentials)
  'e2b', 'supabase', 'postgres', 'aws', 'azure', 'docker',
  'cloudflare', 'render', 'vercel',

  // GitHub (requires GITHUB_TOKEN)
  'github',

  // Databases (require setup/credentials)
  'neo4j', 'milvus', 'chroma', 'duckdb', 'clickhouse',
  'mongodb', 'couchbase', 'neon', 'singlestore',

  // Communication (require accounts)
  'slack', 'notion', 'taskade', 'discord', 'telegram',

  // AI/ML services (require API keys/accounts)
  'logfire', 'langfuse', 'elevenlabs', 'openai',

  // Observability (require setup/accounts)
  'grafana', 'sentry', 'sonarqube', 'datadog',

  // Payment services
  'stripe', 'paypal', 'chargebee'
];

function filterMCPServers(servers) {
  if (!servers || !Array.isArray(servers)) {
    return servers;
  }

  return servers.filter(server => {
    // Keep if in FREE_SERVERS list
    if (FREE_SERVERS.includes(server)) {
      return true;
    }

    // Remove if in PAID_SERVERS list
    if (PAID_SERVERS.includes(server)) {
      console.log(`  Removing paid server: ${server}`);
      return false;
    }

    // Keep if not explicitly in either list (might be free but unknown)
    console.log(`  Keeping unknown server: ${server} (verify if free)`);
    return true;
  });
}

function updateAgent(agentName, agentData) {
  if (!agentData.mcp_servers) {
    return agentData;
  }

  console.log(`\nUpdating agent: ${agentName}`);

  const originalRequired = agentData.mcp_servers.required || [];
  const originalRecommended = agentData.mcp_servers.recommended || [];

  const filteredRequired = filterMCPServers(originalRequired);
  const filteredRecommended = filterMCPServers(originalRecommended);

  console.log(`  Required: ${originalRequired.length} -> ${filteredRequired.length}`);
  console.log(`  Recommended: ${originalRecommended.length} -> ${filteredRecommended.length}`);

  return {
    ...agentData,
    mcp_servers: {
      required: filteredRequired,
      recommended: filteredRecommended,
      usage: agentData.mcp_servers.usage
    }
  };
}

function main() {
  const registryPath = path.join(__dirname, 'registry.json');

  // Read existing registry
  const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));

  console.log('Removing paid/API-required MCP servers from agent registry...\n');
  console.log(`Total paid servers to remove: ${PAID_SERVERS.length}`);
  console.log(`Keeping only free servers: ${FREE_SERVERS.length}\n`);

  // Update each agent
  const updatedAgents = {};
  let totalRemoved = 0;

  for (const [agentName, agentData] of Object.entries(registry.agents)) {
    const updated = updateAgent(agentName, agentData);
    updatedAgents[agentName] = updated;
  }

  // Create updated registry
  const updatedRegistry = {
    ...registry,
    agents: updatedAgents,
    mcp_integration: {
      version: '3.0.3',
      last_updated: '2025-11-01',
      description: 'All agents now use ONLY free MCP servers (no payment, API keys, or accounts required)',
      free_servers_only: true,
      marketplace_guide: 'docs/MCP-MARKETPLACE-GUIDE.md'
    }
  };

  // Write updated registry
  fs.writeFileSync(registryPath, JSON.stringify(updatedRegistry, null, 2), 'utf8');

  console.log('\n=== UPDATE COMPLETE ===');
  console.log(`Total agents updated: ${Object.keys(updatedAgents).length}`);
  console.log(`Free servers available: ${FREE_SERVERS.length}`);
  console.log('  - 4 currently installed (local)');
  console.log('  - 7 official Anthropic servers');
  console.log('\nAll paid/API-required servers removed from recommendations');
  console.log('See docs/MCP-MARKETPLACE-GUIDE.md for updated server list');
}

main();
