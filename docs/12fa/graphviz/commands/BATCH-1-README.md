# Command Graphviz Diagrams - Batch 1

This directory contains **28 Graphviz workflow diagrams** for slash commands, visualizing their execution flows and decision logic.

---

## Quick Reference

### Rendering Diagrams

**Single diagram to SVG:**
```bash
dot -Tsvg claude-flow-swarm-process.dot -o claude-flow-swarm-process.svg
```

**All diagrams to SVG:**
```bash
for f in *-process.dot; do dot -Tsvg "$f" -o "${f%.dot}.svg"; done
```

**Single diagram to PNG:**
```bash
dot -Tpng -Gdpi=300 claude-flow-swarm-process.dot -o claude-flow-swarm-process.png
```

**All diagrams to PNG:**
```bash
for f in *-process.dot; do dot -Tpng -Gdpi=300 "$f" -o "${f%.dot}.png"; done
```

---

## Diagram Categories

### 1. Core Orchestration (3 diagrams)
| Command | File | Purpose |
|---------|------|---------|
| `/claude-flow-swarm` | `claude-flow-swarm-process.dot` | Multi-agent swarm coordination |
| `/claude-flow-memory` | `claude-flow-memory-process.dot` | CRDT-based memory system |
| `/sparc` | `sparc-process.dot` | 5-phase SPARC methodology |

### 2. SPARC Methodology (11 diagrams)
| Command | File | Purpose |
|---------|------|---------|
| `/sparc:code` | `sparc-code-process.dot` | Auto-Coder implementation |
| `/sparc:integration` | `sparc-integration-process.dot` | System integration |
| `/sparc:security-review` | `sparc-security-review-process.dot` | Security audit |
| `/sparc:spec-pseudocode` | `sparc-spec-pseudocode-process.dot` | Requirements & pseudocode |
| `/sparc:debug` | `sparc-debug-process.dot` | Systematic debugging |
| `/sparc:sparc` | `sparc-sparc-process.dot` | Full SPARC orchestration |
| `/sparc:refinement-optimization-mode` | `sparc-refinement-optimization-mode-process.dot` | Performance optimization |
| `/sparc:devops` | `sparc-devops-process.dot` | Deployment automation |
| `/sparc:ask` | `sparc-ask-process.dot` | Task formulation |
| `/sparc:mcp` | `sparc-mcp-process.dot` | MCP integration |
| `/sparc:docs-writer` | `sparc-docs-writer-process.dot` | Documentation generation |

### 3. Audit & Quality (3 diagrams)
| Command | File | Purpose |
|---------|------|---------|
| `/audit-pipeline` | `audit-pipeline-process.dot` | Complete audit pipeline |
| `/theater-detect` | `theater-detect-process.dot` | Theater detection |
| `/functionality-audit` | `functionality-audit-process.dot` | Sandbox testing |

### 4. Multi-Model (4 diagrams)
| Command | File | Purpose |
|---------|------|---------|
| `/gemini-megacontext` | `gemini-megacontext-process.dot` | 2M context analysis |
| `/codex-auto` | `codex-auto-process.dot` | Extended reasoning |
| `/gemini-search` | `gemini-search-process.dot` | Real-time web search |
| `/gemini-media` | `gemini-media-process.dot` | Multimodal processing |

### 5. Workflow & Agent (4 diagrams)
| Command | File | Purpose |
|---------|------|---------|
| `/agent-rca` | `agent-rca-process.dot` | Root cause analysis |
| `/create-micro-skill` | `create-micro-skill-process.dot` | Atomic skill creation |
| `/create-cascade` | `create-cascade-process.dot` | Workflow orchestration |
| `/sparc:post-deployment-monitoring-mode` | `sparc-post-deployment-monitoring-mode-process.dot` | Post-deployment monitoring |

### 6. Essential Commands (3 diagrams)
| Command | File | Purpose |
|---------|------|---------|
| `/quick-check` | `quick-check-process.dot` | Fast quality check |
| `/fix-bug` | `fix-bug-process.dot` | Smart bug fixing |
| `/build-feature` | `build-feature-process.dot` | Feature development |

---

## Color Scheme

Each diagram uses a consistent color scheme:

| Color | Hex | Usage |
|-------|-----|-------|
| Light Blue | `#E3F2FD` | Standard nodes |
| Green | `#4CAF50` | Start/Success nodes |
| Red | `#F44336` | Error/Critical nodes |
| Yellow | `#FFF9C4` | Validation nodes |
| Orange | `#FFE0B2` | Processing nodes |
| Blue | `#BBDEFB` | Execution nodes |
| Purple | `#C5CAE9` | Decision nodes |
| Teal | `#B2DFDB` | Storage/Memory nodes |

---

## Validation

Run the validation script to check all diagrams:

```bash
bash validate-diagrams.sh
```

---

## Statistics

- **Total diagrams**: 28
- **Average file size**: ~3 KB
- **Total directory size**: ~85 KB
- **Categories**: 6
- **Format**: Graphviz DOT language

---

**Created**: 2025-11-01
**Version**: 1.0.0
**Status**: Complete
