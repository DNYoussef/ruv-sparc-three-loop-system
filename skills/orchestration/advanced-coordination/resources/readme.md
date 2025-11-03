# Resources - Advanced Coordination

Supporting scripts, templates, and assets for multi-agent swarm coordination systems.

## Directory Structure

### scripts/
Automation and validation scripts for swarm topology deployment and management.

**Available Scripts:**
- `validate_topology.py` - Validates swarm topology configurations against schema
- `deploy_swarm.sh` - Deploys coordination swarm with specified topology and consensus

### templates/
Boilerplate configuration files for common swarm topologies.

**Available Templates:**
- `mesh-topology.yaml` - Mesh topology (peer-to-peer, all-to-all connections)
- `hierarchical-topology.yaml` - Hierarchical topology (tree structure, parent-child)
- `ring-topology.yaml` - Ring topology (circular connections)
- `star-topology.yaml` - Star topology (central coordinator hub)

### assets/
Reference diagrams and configuration examples for coordination patterns.

## Usage

### Validate Topology Configuration
```bash
python resources/scripts/validate_topology.py resources/templates/mesh-topology.yaml
```

### Deploy Swarm
```bash
bash resources/scripts/deploy_swarm.sh --topology mesh --agents 5 --strategy balanced
```

## Integration with Advanced Coordination Skill

These resources support all coordination patterns:
- **Phase 1**: Topology selection (uses templates)
- **Phase 2**: Swarm initialization (uses deployment script)
- **Phase 3**: Validation testing (uses validation script)

## Requirements

**Python Scripts:**
- Python 3.8+
- PyYAML library (`pip install pyyaml`)
- JSON schema validator (`pip install jsonschema`)

**Bash Scripts:**
- Bash 4.0+
- Claude Flow installed (`npm install -g claude-flow@alpha`)
- MCP servers configured

## Troubleshooting

**"Module not found" errors:**
```bash
pip install pyyaml jsonschema
```

**"Command not found: npx" errors:**
```bash
npm install -g npx
```

**Permission denied:**
```bash
chmod +x resources/scripts/*.sh
```
