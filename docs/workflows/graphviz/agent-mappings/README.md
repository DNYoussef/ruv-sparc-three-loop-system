# Agent-Command Relationship GraphViz Diagrams

## Overview

This directory contains 5 comprehensive GraphViz diagrams visualizing agent-to-command relationships across the SPARC development ecosystem. Each diagram provides a different perspective on how agents interact with commands, highlighting patterns, clusters, and specializations.

## Diagrams

### 1. Complete Agent-Command Network (`agent-command-network.dot`)

**Purpose**: Bipartite graph showing all agents and all commands with complete relationship mapping.

**Features**:
- 54+ agents organized into 9 MECE domain clusters
- 45+ commands grouped by category (File, Git, Communication, Memory, Testing, Utilities)
- Color-coded edges by agent domain
- Comprehensive mapping of all agent-command relationships

**Key Insights**:
- Universal commands (used by 10+ agents): `/file-read`, `/memory-store`, `/memory-retrieve`
- Core development agents have broadest command usage (12+ commands)
- Coordination agents focus heavily on memory and communication commands

**Best For**: Understanding the complete system topology and identifying universal infrastructure commands.

---

### 2. Core Agent Command Hub (`core-agent-hub.dot`)

**Purpose**: Focus on 15-20 most important "core" agents and their command usage patterns.

**Features**:
- 15 core agents across 3 tiers: Development, Coordination, Specialized
- Commands categorized by usage frequency (★★★★★ = 10+ agents, ★ = 1 agent)
- Weighted edges showing usage intensity (line thickness)
- Usage statistics and overlap analysis

**Key Insights**:
- `file-read` is universal (used by all 15 core agents)
- `memory-store` and `memory-retrieve` are critical coordination infrastructure
- Development workflow: file operations → git operations → testing → communication
- Command overlap indicates strong infrastructure dependency

**Best For**: Identifying the most critical agents and commands for system stability.

---

### 3. Command Distribution Heat Map (`command-distribution.dot`)

**Purpose**: Visualize command usage distribution across domains with node sizes indicating popularity.

**Features**:
- Commands organized into 10 MECE domains
- Node size proportional to number of agents using each command
- Domain-level usage statistics (avg agents per command)
- Cross-domain workflow relationships (dashed lines)

**Key Insights**:
- **Memory & State** domain has highest avg usage (6.5 agents/command)
- **File Operations** most distributed (8 commands, 6.1 avg)
- **Utilities** underutilized (0.5 agents/command avg)
- 70% of commands used by 3+ agents (indicates good reuse)
- 5 major cross-domain workflows identified

**Best For**: Understanding command popularity, identifying underutilized commands, and detecting usage patterns.

---

### 4. Agent Capability Clusters (`agent-capability-clusters.dot`)

**Purpose**: Group agents by primary capability and show command overlap between clusters.

**Features**:
- 9 capability clusters: Core Dev, QA, Coordination, VCS, Memory Mgmt, Research, Consensus, Documentation, DevOps
- Shared commands within each cluster (high cohesion)
- Universal commands spanning 5+ clusters (octagons)
- Cross-cluster workflows showing inter-cluster dependencies
- Cluster cohesion metrics (% command overlap)

**Key Insights**:
- **Memory Management** cluster has highest cohesion (83%)
- `/file-read` spans 6/9 clusters (universal)
- `/memory-store` spans 7/9 clusters (universal)
- 5 cross-cluster workflows: Development→QA, QA→Documentation, Coordination→Development, VCS→DevOps, Research→Coordination
- Most common workflow: Development→QA (commit then test)

**Best For**: Understanding agent groupings, identifying cluster boundaries, and optimizing cross-cluster coordination.

---

### 5. Specialist Agent Focus (`specialist-agents.dot`)

**Purpose**: Highlight highly specialized agents and their unique/exclusive command sets.

**Features**:
- 8 specialist categories: ML/Data Science, Security, Consensus, Performance, DevOps, Repo Architecture, SPARC, Reverse Engineering
- Existing commands (solid) vs. proposed/conceptual commands (dashed)
- Exclusivity analysis (% unique commands per specialist)
- Recommendations for new specialist-specific commands

**Key Insights**:
- **ML & Performance** specialists have highest exclusivity (75% unique commands)
- **Consensus** specialists have lowest exclusivity (33%) - heavily reliant on shared coordination
- 18+ exclusive/proposed commands identified
- 8 specialist capabilities with unique value propositions
- Recommendations: Implement `/ml-train`, `/security-scan`, `/consensus-vote`, `/perf-profile`

**Best For**: Identifying gaps in specialist tooling and planning new command development.

---

## Rendering the Diagrams

### Prerequisites

Install Graphviz:

**macOS**:
```bash
brew install graphviz
```

**Ubuntu/Debian**:
```bash
sudo apt-get install graphviz
```

**Windows**:
```bash
choco install graphviz
# Or download from: https://graphviz.org/download/
```

**Verify installation**:
```bash
dot -V
```

### Rendering Commands

**Generate PNG images (recommended for viewing)**:
```bash
cd docs/workflows/graphviz/agent-mappings/

# Render all diagrams
dot -Tpng agent-command-network.dot -o agent-command-network.png
dot -Tpng core-agent-hub.dot -o core-agent-hub.png
dot -Tpng command-distribution.dot -o command-distribution.png
dot -Tpng agent-capability-clusters.dot -o agent-capability-clusters.png
dot -Tpng specialist-agents.dot -o specialist-agents.png
```

**Generate SVG (vector graphics, scalable)**:
```bash
dot -Tsvg agent-command-network.dot -o agent-command-network.svg
dot -Tsvg core-agent-hub.dot -o core-agent-hub.svg
dot -Tsvg command-distribution.dot -o command-distribution.svg
dot -Tsvg agent-capability-clusters.dot -o agent-capability-clusters.svg
dot -Tsvg specialist-agents.dot -o specialist-agents.svg
```

**Generate PDF (printable)**:
```bash
dot -Tpdf agent-command-network.dot -o agent-command-network.pdf
dot -Tpdf core-agent-hub.dot -o core-agent-hub.pdf
dot -Tpdf command-distribution.dot -o command-distribution.pdf
dot -Tpdf agent-capability-clusters.dot -o agent-capability-clusters.pdf
dot -Tpdf specialist-agents.dot -o specialist-agents.pdf
```

**Batch render all formats**:
```bash
#!/bin/bash
for diagram in *.dot; do
    base="${diagram%.dot}"
    dot -Tpng "$diagram" -o "$base.png"
    dot -Tsvg "$diagram" -o "$base.svg"
    dot -Tpdf "$diagram" -o "$base.pdf"
    echo "Rendered: $base"
done
```

### Layout Engines

Graphviz provides different layout engines for different graph types:

- **`dot`** (default): Hierarchical/directed graphs - **USED FOR ALL DIAGRAMS**
- **`neato`**: Undirected graphs with spring model
- **`fdp`**: Undirected graphs with force-directed placement
- **`sfdp`**: Large undirected graphs (scalable force-directed)
- **`circo`**: Circular layout
- **`twopi`**: Radial layout

**Try alternative layouts** (if default doesn't render well):
```bash
neato -Tpng agent-command-network.dot -o agent-command-network-neato.png
fdp -Tpng command-distribution.dot -o command-distribution-fdp.png
```

### Troubleshooting

**Issue: Diagram too large to render**
```bash
# Increase memory limit and timeout
dot -Tpng -Gmaxiter=10000 -Gmargin=0 agent-command-network.dot -o output.png
```

**Issue: Overlapping nodes**
```bash
# Increase spacing
dot -Tpng -Gnodesep=2.0 -Granksep=3.0 diagram.dot -o output.png
```

**Issue: Text too small**
```bash
# Increase DPI
dot -Tpng -Gdpi=300 diagram.dot -o output.png
```

## Diagram Statistics

| Diagram | Nodes | Edges | Clusters | Complexity |
|---------|-------|-------|----------|------------|
| Complete Network | 99+ | 250+ | 16 | Very High |
| Core Hub | 45 | 80+ | 6 | High |
| Distribution | 47 | 65+ | 11 | Medium |
| Capability Clusters | 65+ | 120+ | 11 | High |
| Specialist Focus | 55+ | 75+ | 16 | Medium |

## Use Cases

### For Developers
- **Network**: Understand all available commands for your agent type
- **Core Hub**: Identify most frequently used commands for optimization
- **Distribution**: Find underutilized commands that might be useful

### For Architects
- **Capability Clusters**: Design agent groupings and cluster boundaries
- **Specialist Focus**: Identify gaps in tooling and plan new specialist agents
- **Network**: Validate MECE domain taxonomy completeness

### For Coordinators
- **Core Hub**: Understand command usage patterns for swarm optimization
- **Distribution**: Identify cross-domain workflows for coordination
- **Capability Clusters**: Optimize inter-cluster communication

### For Product Managers
- **Specialist Focus**: Prioritize specialist command development
- **Distribution**: Understand feature adoption and usage
- **Capability Clusters**: Plan feature rollout by cluster

## Color Coding Reference

### Agent Domains (Network & Core Hub)
- **Blue (#4A90E2)**: Core Development
- **Orange (#F5A623)**: Swarm Coordination
- **Purple (#7B68EE)**: Consensus & Distributed
- **Green (#50C878)**: Performance & Optimization
- **Red (#E85D75)**: GitHub & Repository
- **Orange (#FF9F40)**: SPARC Methodology
- **Light Blue (#5DADE2)**: Specialized Development
- **Dark Green (#27AE60)**: Testing & Validation
- **Dark Red (#E74C3C)**: Migration & Planning

### Command Categories (Distribution)
- **Blue (#2196F3)**: File Operations
- **Orange (#FF9800)**: Git Operations
- **Purple (#9C27B0)**: Communication & Coordination
- **Green (#4CAF50)**: Memory & State
- **Yellow (#FBC02D)**: Testing & Validation
- **Pink (#E91E63)**: Code Quality
- **Teal (#00897B)**: Documentation
- **Brown (#D84315)**: Utilities

### Capability Clusters
- **Blue (#2196F3)**: Core Development
- **Purple (#9C27B0)**: Quality Assurance
- **Orange (#FF9800)**: Coordination
- **Green (#4CAF50)**: Version Control
- **Pink (#E91E63)**: Memory Management
- **Teal (#009688)**: Research & Analysis
- **Yellow (#FBC02D)**: Consensus & Distributed
- **Indigo (#3F51B5)**: Documentation & Specification
- **Brown (#795548)**: DevOps & Automation

### Specialist Categories
- **Indigo (#3F51B5)**: Reverse Engineering
- **Green (#4CAF50)**: ML & Data Science
- **Pink (#E91E63)**: Security & Cryptography
- **Orange (#FF9800)**: Consensus & Distributed
- **Teal (#009688)**: Performance & Optimization
- **Yellow (#FBC02D)**: DevOps & Infrastructure
- **Purple (#9C27B0)**: Repository & Architecture
- **Light Blue (#03A9F4)**: SPARC Methodology

## Version History

- **v1.0** (2025-11-01): Initial creation of 5 comprehensive diagrams
  - Complete Agent-Command Network
  - Core Agent Command Hub
  - Command Distribution Heat Map
  - Agent Capability Clusters
  - Specialist Agent Focus

## Contributing

To update these diagrams:

1. Edit the `.dot` source files
2. Maintain consistent color schemes (see Color Coding Reference)
3. Update statistics in this README
4. Re-render all formats (PNG, SVG, PDF)
5. Commit both source and rendered files

## Related Documentation

- **Agent Taxonomy**: `docs/agent-taxonomy/`
- **Command Reference**: `docs/commands/`
- **SPARC Methodology**: `docs/sparc/`
- **Integration Patterns**: `docs/integration-plans/`

## License

These diagrams are part of the SPARC Development Environment documentation.

---

**Last Updated**: 2025-11-01
**Maintainer**: Code Quality Analyzer Agent
**Status**: ✅ Production Ready
