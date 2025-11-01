const fs = require('fs');
const registry = JSON.parse(fs.readFileSync('agents/registry.json', 'utf8'));
registry.agents['mobile-dev'].mcp_servers.recommended = registry.agents['mobile-dev'].mcp_servers.recommended.filter(s => s !== 'firebase');
fs.writeFileSync('agents/registry.json', JSON.stringify(registry, null, 2), 'utf8');
console.log('Removed firebase from mobile-dev');
