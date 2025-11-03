# Flow Nexus Swarm - Gold Tier Skill

**Status**: Gold Tier (12+ files)
**Version**: 2.0.0
**Category**: Cloud Orchestration

## Overview

Cloud-based AI swarm deployment and event-driven workflow automation with Flow Nexus platform. This Gold tier skill provides comprehensive tooling, templates, and tests for production-ready swarm orchestration.

## Skill Structure (Gold Tier)

```
flow-nexus-swarm/
├── SKILL.md                          # Core skill documentation
├── README.md                         # This file
├── flow-nexus-swarm-process.dot     # GraphViz process visualization
├── resources/
│   ├── scripts/                     # Functional deployment scripts (4)
│   │   ├── deploy-cloud-swarm.js
│   │   ├── event-driven-workflow.js
│   │   ├── swarm-monitor.js
│   │   └── template-manager.js
│   └── templates/                   # Swarm configuration templates (3)
│       ├── full-stack-dev-swarm.json
│       ├── research-analysis-swarm.json
│       └── ci-cd-pipeline-swarm.json
└── tests/                           # Comprehensive test suites (3)
    ├── test-swarm-deployment.js
    ├── test-event-workflows.js
    └── test-integration.js
```

**Total Files**: 13 (exceeds Gold tier requirement of 12)

## Quick Start

### 1. Prerequisites

```bash
# Install Flow Nexus
npm install -g flow-nexus@latest

# Register/login
npx flow-nexus@latest register
npx flow-nexus@latest login

# Add MCP server to Claude Code
claude mcp add flow-nexus npx flow-nexus@latest mcp start
```

### 2. Deploy Your First Swarm

```bash
# Using script with default settings
node resources/scripts/deploy-cloud-swarm.js

# Or with custom configuration
node resources/scripts/deploy-cloud-swarm.js \
  --topology hierarchical \
  --agents 8 \
  --workflow ci-cd-pipeline.json
```

### 3. Use Pre-built Templates

```bash
# List available templates
node resources/scripts/template-manager.js list --category quickstart

# Deploy from template
node resources/scripts/template-manager.js deploy --template full-stack-dev
```

## Scripts

### `deploy-cloud-swarm.js`
Full-featured swarm deployment with:
- Multi-topology support (hierarchical, mesh, ring, star)
- Custom agent spawning
- Workflow integration
- Colored terminal output
- Error handling

**Usage:**
```bash
node deploy-cloud-swarm.js --config swarm-config.json
node deploy-cloud-swarm.js --topology mesh --agents 5 --verbose
```

### `event-driven-workflow.js`
Event-driven workflow automation with:
- Multiple trigger types (GitHub, schedule, manual)
- Async/sync execution modes
- Message queue processing
- Retry mechanisms

**Usage:**
```bash
node event-driven-workflow.js create --name "CI/CD" --trigger github_push
node event-driven-workflow.js execute --workflow-id <id> --async
node event-driven-workflow.js monitor --workflow-id <id>
```

### `swarm-monitor.js`
Real-time swarm monitoring with:
- Live health metrics
- Agent status tracking
- Workload visualization
- Performance analytics
- Export to JSON

**Usage:**
```bash
node swarm-monitor.js --swarm-id <id>
node swarm-monitor.js --all --watch --interval 10
node swarm-monitor.js --export metrics.json
```

### `template-manager.js`
Template management with:
- List/view templates
- Deploy from templates
- Create custom templates
- Export configurations

**Usage:**
```bash
node template-manager.js list --category quickstart
node template-manager.js deploy --template code-review --agents 4
node template-manager.js create --from-swarm <id> --name my-template
```

## Templates

### Full-Stack Development (`full-stack-dev-swarm.json`)
Complete web app development with 8 specialized agents:
- Project Manager (coordinator)
- Technical Architect (researcher)
- Backend/Frontend/Database Developers (coders)
- QA Engineer (analyst)
- Performance Engineer (optimizer)
- DevOps Specialist (analyst)

**Topology**: Hierarchical
**Use Cases**: Web apps, SaaS, e-commerce

### Research & Analysis (`research-analysis-swarm.json`)
Collaborative research with 5 agents:
- Primary/Secondary Researchers
- Data Analyst
- Insights Specialist
- Research Coordinator

**Topology**: Mesh (peer-to-peer)
**Use Cases**: Market research, competitive analysis, literature review

### CI/CD Pipeline (`ci-cd-pipeline-swarm.json`)
Automated deployment with 6 agents:
- Pipeline Orchestrator
- Code Quality Checker
- Test Runner
- Build Engineer
- Security Scanner
- Deployment Specialist

**Topology**: Star (centralized)
**Use Cases**: Continuous deployment, multi-env testing, security-first deployments

## Tests

### `test-swarm-deployment.js`
Tests swarm initialization, agent spawning, and error handling.

**Run:**
```bash
node tests/test-swarm-deployment.js
VERBOSE=true node tests/test-swarm-deployment.js
```

### `test-event-workflows.js`
Tests event-driven workflows, triggers, and retry mechanisms.

**Run:**
```bash
node tests/test-event-workflows.js
SKIP_CLEANUP=true node tests/test-event-workflows.js
```

### `test-integration.js`
End-to-end integration tests for complete workflows.

**Run:**
```bash
node tests/test-integration.js
```

## GraphViz Process Diagram

View the complete swarm deployment process:

```bash
# Generate PNG (requires Graphviz installed)
dot -Tpng flow-nexus-swarm-process.dot -o process.png

# Or view in VS Code with Graphviz extension
```

The diagram shows:
- 5 phases (Planning → Initialization → Agents → Workflows → Monitoring)
- Topology decision branching
- Workflow deployment decision
- Quality gates and error handling
- External skill integration points
- Best practices guidelines

## Architecture

### Swarm Topologies

| Topology | Structure | Best For |
|----------|-----------|----------|
| **Hierarchical** | Tree with coordinator nodes | Complex projects, management layers |
| **Mesh** | Peer-to-peer collaboration | Research, analysis, brainstorming |
| **Ring** | Circular coordination | Sequential workflows, pipelines |
| **Star** | Centralized hub | Simple delegation, command-control |

### Agent Types

- **Coordinator**: Task delegation, progress tracking
- **Researcher**: Information gathering, analysis
- **Coder**: Implementation, refactoring
- **Analyst**: Testing, quality assurance
- **Optimizer**: Performance tuning, optimization

### Execution Strategies

- **Balanced**: Equal workload distribution
- **Specialized**: Focus on expertise areas
- **Adaptive**: Dynamic adjustment based on complexity

## Integration with SPARC

```bash
# Use with SPARC workflow
npx claude-flow sparc run architect "Design swarm topology"

# Coordinate with hooks
npx claude-flow@alpha hooks pre-task --description "Deploy swarm"
node resources/scripts/deploy-cloud-swarm.js
npx claude-flow@alpha hooks post-task --task-id "swarm-deploy"
```

## Best Practices

1. **Choose Right Topology**: Match topology to project structure
2. **Use Templates**: Start with proven configurations
3. **Monitor Actively**: Enable real-time health monitoring
4. **Implement Retries**: Configure exponential backoff for resilience
5. **Leverage Async**: Use message queues for long-running workflows
6. **Clean Up Resources**: Always destroy swarms when done

## Troubleshooting

### Authentication Issues
```bash
npx flow-nexus@latest login
```

### Swarm Initialization Failures
- Verify MCP server is running: `claude mcp list`
- Check network connectivity
- Ensure valid topology/strategy values

### Workflow Execution Stalls
- Check queue status: `node event-driven-workflow.js queue`
- Verify agent availability
- Review workflow dependencies

## Documentation

- **Flow Nexus Platform**: https://flow-nexus.ruv.io
- **GitHub**: https://github.com/ruvnet/flow-nexus
- **Claude Flow**: https://github.com/ruvnet/claude-flow

## Version History

- **2.0.0** (Gold Tier): Added scripts, templates, tests, GraphViz diagram
- **1.0.0**: Initial Silver tier skill with SKILL.md

---

**Skill Tier**: Gold (13 files)
**Last Updated**: 2025-11-02
**Maintainer**: Claude Code Enhancement Team
