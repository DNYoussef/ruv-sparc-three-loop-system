# Plugins Directory Documentation

## Purpose

The `plugins/` directory contains modular sub-plugins that organize the ruv-sparc-three-loop-system into focused, domain-specific packages following the **12-Factor Agents** methodology. This architectural pattern allows users to:

1. **Selective Installation**: Install only the sub-plugins they need rather than the entire system
2. **Modular Organization**: Keep related skills, agents, and commands logically grouped
3. **Independent Versioning**: Each sub-plugin can be versioned and updated independently
4. **Reduced Complexity**: Break down the 122+ skills into manageable, focused packages

## Architecture Pattern

This is a **meta-plugin** architecture where the main plugin acts as an orchestrator and container for specialized sub-plugins. Each sub-plugin is a complete, self-contained Claude Code plugin with its own:

- `plugin.json` - Plugin metadata and configuration
- `README.md` - Documentation and feature list
- Skills, agents, commands specific to that domain

## Sub-Plugins

The system is organized into 5 focused sub-plugins:

### 1. 12fa-core
**Essential foundation for all 12-Factor Agent development**

**Location**: `plugins/12fa-core/`

**Purpose**: Core SPARC methodology, fundamental agents, and quality assurance tools

**Contents**:
- **Skills (10)**: agent-creator, sparc-methodology, functionality-audit, theater-detection, production-readiness, audit-pipeline, code-review-assistant, quick-quality-check, smart-bug-fix, prompt-architect
- **Agents (12)**: coder, reviewer, tester, planner, researcher, sparc-coord, sparc-coder, specification, pseudocode, architecture, refinement, technical-debt-manager

**Use When**: Starting any new project, need systematic development workflow, require quality validation

### 2. 12fa-three-loop
**The Three-Loop Integrated Development System**

**Location**: `plugins/12fa-three-loop/`

**Purpose**: Research-driven planning, parallel swarm implementation, and CI/CD intelligent recovery

**Contents**:
- **Skills (3)**: research-driven-planning (Loop 1), parallel-swarm-implementation (Loop 2), cicd-intelligent-recovery (Loop 3)
- **Agents (6)**: Specialized loop coordinators and validators

**Use When**: Building complex features, need comprehensive planning with risk mitigation, require automated failure recovery

**Integration**: Forms the backbone of the three-loop system:
- Loop 1: Research-driven planning with 5x pre-mortem
- Loop 2: Parallel swarm implementation with theater detection
- Loop 3: CI/CD intelligent recovery with root cause analysis

### 3. 12fa-swarm
**Multi-agent swarm orchestration and coordination**

**Location**: `plugins/12fa-swarm/`

**Purpose**: Advanced swarm topologies, consensus mechanisms, and distributed coordination

**Contents**:
- **Skills (15)**: swarm-orchestration, hive-mind-advanced, cascade-orchestrator, parallel-swarm, coordination patterns
- **Agents (20+)**: hierarchical-coordinator, mesh-coordinator, byzantine-coordinator, raft-manager, consensus-validator, etc.

**Use When**: Need multi-agent coordination, require Byzantine consensus, building distributed systems, orchestrating complex workflows

### 4. 12fa-security
**Security, compliance, and reverse engineering**

**Location**: `plugins/12fa-security/`

**Purpose**: Network security, compliance validation, malware analysis, and penetration testing

**Contents**:
- **Skills (8)**: network-security-setup, sandbox-configurator, reverse-engineering (quick/deep/firmware), wcag-accessibility, compliance validation
- **Agents (8)**: Security specialists for various domains

**Use When**: Hardening applications, analyzing binaries, ensuring WCAG compliance, conducting security audits

### 5. 12fa-visual-docs
**Documentation, diagrams, and visual communication**

**Location**: `plugins/12fa-visual-docs/`

**Purpose**: Graphviz workflow diagrams, technical documentation, and visual architecture

**Contents**:
- **Skills (6)**: Graphviz generation, architecture diagrams, documentation automation
- **Resources**: 271+ pre-built Graphviz workflow templates

**Use When**: Creating system diagrams, documenting architecture, generating visual workflows, building technical documentation

## Installation Patterns

### Full Installation (Recommended for Most Users)
Install the main plugin which includes all sub-plugins:
```bash
claude plugin install ruv-sparc-three-loop-system
```

### Selective Installation (Advanced Users)
Install only specific sub-plugins you need:
```bash
# Core foundation (always recommended)
claude plugin install ruv-sparc-three-loop-system/plugins/12fa-core

# Add three-loop system for complex features
claude plugin install ruv-sparc-three-loop-system/plugins/12fa-three-loop

# Add swarm coordination for multi-agent workflows
claude plugin install ruv-sparc-three-loop-system/plugins/12fa-swarm
```

### Minimal Installation (Learning/Testing)
Start with just the core:
```bash
claude plugin install ruv-sparc-three-loop-system/plugins/12fa-core
```

Then add other modules as needed.

## Technical Details

### Directory Structure
```
plugins/
├── 12fa-core/
│   ├── plugin.json
│   ├── README.md
│   ├── skills/
│   ├── agents/
│   └── commands/
├── 12fa-three-loop/
│   ├── plugin.json
│   ├── README.md
│   └── skills/
├── 12fa-swarm/
│   ├── plugin.json
│   ├── README.md
│   ├── skills/
│   └── agents/
├── 12fa-security/
│   ├── plugin.json
│   ├── README.md
│   └── skills/
└── 12fa-visual-docs/
    ├── plugin.json
    ├── README.md
    ├── skills/
    └── templates/
```

### Versioning Strategy
- **Main Plugin**: Semantic versioning for the entire system (v2.0.0)
- **Sub-Plugins**: Independent versioning allowing different update cycles
- **Compatibility**: Sub-plugins maintain compatibility with main plugin API

### Dependencies
- **12fa-core**: No dependencies (standalone)
- **12fa-three-loop**: Depends on 12fa-core
- **12fa-swarm**: Optional, enhances 12fa-three-loop
- **12fa-security**: Standalone
- **12fa-visual-docs**: Standalone

## Design Rationale

### Why Not Standard Directories?

The `plugins/` directory is **non-standard** per official Claude Code documentation, which specifies:
- `commands/` - Custom slash commands
- `agents/` - Custom agents
- `skills/` - Agent Skills
- `hooks/` - Event handlers
- `mcp/` - MCP servers

However, this plugin uses `plugins/` for **meta-plugin architecture** because:

1. **Scale**: 122 skills across 8 domains is too large for a single flat structure
2. **Modularity**: Users should be able to install just what they need
3. **Maintenance**: Easier to update and version independent modules
4. **Organization**: Clear separation of concerns by domain
5. **Reusability**: Sub-plugins can be shared/forked independently

### Compatibility with Claude Code

This architecture is **fully compliant** because:
- The main plugin.json is in the correct location (`.claude-plugin/plugin.json`)
- Standard directories (`skills/`, `agents/`, `commands/`) exist at plugin root
- Sub-plugins are self-contained with their own `plugin.json` files
- Claude Code plugin system supports nested plugin structures
- No conflicts with official specifications

## Usage Examples

### Example 1: New Project with SPARC Methodology
```bash
# Install core for SPARC workflow
claude plugin install ruv-sparc-three-loop-system/plugins/12fa-core

# Use SPARC methodology
claude> "Build user authentication using SPARC methodology"
# Auto-triggers: sparc-methodology skill → 5-phase workflow
```

### Example 2: Complex Feature with Three-Loop System
```bash
# Install core + three-loop
claude plugin install ruv-sparc-three-loop-system/plugins/12fa-core
claude plugin install ruv-sparc-three-loop-system/plugins/12fa-three-loop

# Use three-loop system
claude> "Build payment processing with Stripe integration"
# Auto-triggers:
#   Loop 1: research-driven-planning (research + pre-mortem)
#   Loop 2: parallel-swarm-implementation (6-agent parallel build)
#   Loop 3: cicd-intelligent-recovery (automated testing + fixes)
```

### Example 3: Multi-Agent Swarm Orchestration
```bash
# Full installation for advanced swarm features
claude plugin install ruv-sparc-three-loop-system

# Use swarm coordination
claude> "Refactor legacy codebase using swarm coordination"
# Auto-triggers: swarm-orchestration → hierarchical-coordinator
```

## Maintenance

### Adding a New Sub-Plugin

1. Create directory: `plugins/12fa-{domain}/`
2. Add `plugin.json` with metadata
3. Add `README.md` with feature list
4. Organize skills/agents/commands
5. Update main README.md to list new sub-plugin
6. Update this PLUGINS-DIRECTORY.md

### Updating Sub-Plugins

Each sub-plugin can be versioned independently:
```json
{
  "name": "12fa-core",
  "version": "1.2.0",
  "parent": {
    "name": "ruv-sparc-three-loop-system",
    "version": "2.0.0"
  }
}
```

## Future Expansion

Potential future sub-plugins:
- **12fa-data**: Data engineering, ML pipelines, ETL workflows
- **12fa-frontend**: React, Vue, CSS, accessibility specialists
- **12fa-backend**: API development, database design, caching strategies
- **12fa-research**: Deep research SOP, literature synthesis, reproducibility

## References

- **Main Plugin**: [README.md](../README.md)
- **Claude Code Docs**: https://code.claude.com/docs/en/plugins
- **12-Factor Methodology**: https://12factor.net/
- **Validation Report**: [PLUGIN-VALIDATION-REPORT.md](PLUGIN-VALIDATION-REPORT.md)
