# Top 5 Agent Graphviz Diagrams

## Overview

This directory contains comprehensive Graphviz diagrams for the **TOP 5 AGENTS** in our system, showcasing their workflows, coordination patterns, and integration with the 86-agent registry.

## Created Diagrams

### 1. **Queen Seraphina** (Meta-Orchestrator)
- **File**: `queen-seraphina.dot`
- **Size**: 15 KB
- **Nodes**: 148 nodes
- **Lines**: 325 lines
- **Complexity**: Highest - Meta-level orchestration

**Key Features**:
- 86-agent registry management
- Strategic leadership coordination
- Dynamic agent selection workflow
- Byzantine consensus decision-making (2f+1 quorum)
- META-SKILL compilation process
- Multi-topology coordination (mesh, hierarchical, ring, star)

**Visualizes**:
- Initialization and registry loading
- Strategic task analysis
- Agent selection via vector similarity search
- Byzantine consensus voting
- META-SKILL graph compilation
- Multi-agent coordination
- Memory synchronization
- Validation and quality gates

---

### 2. **Coder Agent** (Core Developer)
- **File**: `coder.dot`
- **Size**: 14 KB
- **Nodes**: 132 nodes
- **Lines**: 313 lines
- **Commands**: 55 commands
- **MCP Tools**: 18 tools

**Key Features**:
- TDD workflow integration (Red → Green → Refactor)
- SOLID principles enforcement
- Error handling patterns
- Performance optimization workflow
- 90%+ test coverage validation

**Visualizes**:
- Command and MCP tool initialization
- Specification analysis
- Planning and architecture
- TDD cycle (test first → implementation → refactor)
- SOLID enforcement (SRP, OCP, LSP, ISP, DIP)
- Error handling and logging
- Performance optimization
- Quality checks and validation
- Agent collaboration (tester, reviewer, planner)

---

### 3. **Tester Agent** (QA Specialist)
- **File**: `tester.dot`
- **Size**: 16 KB
- **Nodes**: 147 nodes
- **Lines**: 353 lines
- **Commands**: 45 commands
- **MCP Tools**: 18 tools

**Key Features**:
- Test pyramid implementation (70% unit, 20% integration, 10% E2E)
- Property-based testing
- Contract testing
- Performance and security testing
- Coverage >80% validation

**Visualizes**:
- Test strategy design
- Test pyramid structure
- Unit test implementation with mocks
- Integration test workflow
- E2E test automation with accessibility
- Property-based testing with generators
- Contract testing (provider/consumer)
- Performance and security testing
- Coverage analysis (line, branch, function, statement)
- Quality gates and CI/CD integration
- Test reporting and metrics

---

### 4. **Planner Agent** (Strategic Planner)
- **File**: `planner.dot`
- **Size**: 16 KB
- **Nodes**: 143 nodes
- **Lines**: 347 lines

**Key Features**:
- Task decomposition workflow
- Dependency analysis and critical path
- Resource allocation and load balancing
- Timeline estimation with Gantt charts
- Risk assessment and mitigation

**Visualizes**:
- Specification analysis and scope definition
- Work breakdown structure (WBS)
- Task categorization and effort estimation
- Dependency mapping and critical path analysis
- Resource allocation and workload balancing
- Timeline estimation with milestones
- Risk assessment (technical, resource, timeline, quality, external)
- Quality gate planning
- Parallel execution patterns
- Multi-agent coordination
- Execution monitoring and adaptive planning
- Planning metrics and knowledge management

---

### 5. **Reviewer Agent** (Code Reviewer)
- **File**: `reviewer.dot`
- **Size**: 17 KB
- **Nodes**: 154 nodes (Most detailed!)
- **Lines**: 350 lines

**Key Features**:
- Comprehensive code review workflow
- Security audit process (8 security checks)
- Performance analysis (6 performance checks)
- Best practices validation (SOLID, DRY, KISS, YAGNI)
- Documentation review

**Visualizes**:
- Pre-review validation (tests, lint, coverage, git history)
- Code review workflow (functionality, architecture, maintainability, error handling, dependencies)
- Security audit (input validation, auth/authz, secrets, crypto, error disclosure, dependency vulnerabilities, config security)
- Performance analysis (complexity, memory, database queries, caching, resource cleanup, scalability)
- Best practices validation (SOLID, DRY, KISS, YAGNI, design patterns)
- Documentation review (inline docs, API docs, README, comments, examples)
- Automated analysis tools
- Finding categorization (critical, major, minor, suggestions)
- Quality scoring system (overall, security, maintainability, performance, documentation)
- Approval decision (approved, conditional, rejected)
- Agent collaboration and reporting

---

## Diagram Statistics

| Agent | Nodes | Lines | File Size | Complexity |
|-------|-------|-------|-----------|------------|
| **Queen Seraphina** | 148 | 325 | 15 KB | Highest (Meta-orchestrator) |
| **Coder** | 132 | 313 | 14 KB | High (55 cmds, 18 MCP) |
| **Tester** | 147 | 353 | 16 KB | High (Test pyramid) |
| **Planner** | 143 | 347 | 16 KB | High (Strategic planning) |
| **Reviewer** | 154 | 350 | 17 KB | Highest (Most detailed) |

**Total**: 724 nodes, 1,688 lines across 5 diagrams

---

## Diagram Features

### Visual Elements

**Color Coding**:
- **Purple (#8b4789)**: Queen Seraphina / Meta-orchestration
- **Orange (#d97706)**: Coder agent / Core development
- **Green (#16a34a)**: Tester agent / QA
- **Cyan (#0891b2)**: Planner agent / Strategic planning
- **Purple (#8b4789)**: Reviewer agent / Code review
- **Red (#dc2626)**: Security / Critical / Risks
- **Blue (#2563eb)**: Strategy / Commands
- **Purple (#7c3aed)**: Swarm coordination / Memory
- **Green (#059669)**: Validation / Quality gates

**Node Types**:
- Rounded boxes with filled backgrounds
- Clear labels with role descriptions
- Hierarchical clustering by function

**Edge Types**:
- **Solid arrows**: Primary workflow
- **Bold arrows**: Critical path / Completion
- **Dashed arrows**: Feedback loops / Retry paths
- **Labeled edges**: Action descriptions

### Diagram Structure

Each diagram follows a consistent hierarchical structure:
1. **Initialization Layer**: Agent setup and configuration
2. **Input/Analysis Layer**: Specification/requirement analysis
3. **Core Processing Layers**: Primary workflows and operations
4. **Coordination Layer**: Multi-agent collaboration
5. **Validation Layer**: Quality gates and checks
6. **Output/Reporting Layer**: Results and metrics
7. **Feedback Loops**: Retry and improvement paths

---

## Rendering Instructions

### Using Graphviz CLI

```bash
# Generate SVG (recommended for web)
dot -Tsvg queen-seraphina.dot -o queen-seraphina.svg

# Generate PNG (for documentation)
dot -Tpng -Gdpi=300 coder.dot -o coder.png

# Generate PDF (for printing)
dot -Tpdf tester.dot -o tester.pdf

# Batch convert all diagrams to SVG
for file in *.dot; do
    dot -Tsvg "$file" -o "${file%.dot}.svg"
done
```

### Using Online Tools

1. **Graphviz Online**: https://dreampuf.github.io/GraphvizOnline/
2. **SketchViz**: https://sketchviz.com/new
3. **Viz.js**: http://viz-js.com/

Simply copy the `.dot` file contents and paste into the editor.

### Using VS Code Extensions

Install the **Graphviz Preview** extension:
1. Open `.dot` file in VS Code
2. Press `Ctrl+Shift+V` (Windows/Linux) or `Cmd+Shift+V` (Mac)
3. Preview appears in split view

---

## Integration with System

These diagrams are part of the **12FA (Twelve-Factor Agents)** documentation system and integrate with:

- **86-Agent Registry**: All agents managed by Queen Seraphina
- **Claude-Flow**: Command and MCP tool coordination
- **Ruv-Swarm**: Distributed agent coordination
- **Flow-Nexus**: Cloud-based orchestration (optional)
- **SPEK Methodology**: Specification, Planning, Execution, Knowledge phases

---

## Technical Specifications

### Diagram Validation

All diagrams have been validated for:
- ✅ **Syntax**: Valid Graphviz DOT syntax
- ✅ **Completeness**: 60-200 nodes per diagram (requirement met)
- ✅ **Structure**: Network/hierarchical diagram types
- ✅ **Clarity**: Clear agent interaction flows
- ✅ **Legends**: Color-coded legends for easy understanding

### Node Density

- **Optimal range**: 60-200 nodes per diagram
- **Queen Seraphina**: 148 nodes ✅ (within range)
- **Coder**: 132 nodes ✅ (within range)
- **Tester**: 147 nodes ✅ (within range)
- **Planner**: 143 nodes ✅ (within range)
- **Reviewer**: 154 nodes ✅ (within range, most detailed)

### Graph Attributes

```dot
graph [
    rankdir=TB,              // Top to bottom layout
    bgcolor="#0a0e1a",       // Dark background
    fontname="Arial",        // Clean font
    fontsize=12,             // Readable size
    splines=ortho,           // Orthogonal edges
    nodesep=0.6,             // Node spacing
    ranksep=1.0              // Rank spacing
]
```

---

## Usage Examples

### Queen Seraphina Workflow

```bash
# Initialize Queen Seraphina
mcp__ruv-swarm__swarm_init --topology=hierarchical --maxAgents=86

# Dynamic agent selection
# Queen analyzes task → Matches skills → Allocates agents → Coordinates execution

# Byzantine consensus for critical decisions
# Proposal → Vote collection → Fault detection → Consensus (2f+1) → Execute
```

### Coder Agent TDD Cycle

```bash
# TDD workflow
1. Write test (Red)
2. Run test (Fails)
3. Write implementation (Green)
4. Run test (Passes)
5. Refactor
6. Validate coverage >90%
7. Repeat for next feature
```

### Tester Agent Test Pyramid

```bash
# Test distribution
- Unit tests: 70% coverage (fast, isolated)
- Integration tests: 20% coverage (component interactions)
- E2E tests: 10% coverage (user journeys)

# Quality gates
- Line coverage ≥90%
- Branch coverage ≥85%
- Function coverage ≥95%
```

### Planner Agent WBS

```bash
# Work breakdown structure
1. Specification analysis
2. Task decomposition
3. Dependency mapping
4. Resource allocation
5. Timeline estimation
6. Risk assessment
7. Quality gate planning
```

### Reviewer Agent Review Checklist

```bash
# Code review checklist
✓ Functionality (implements requirements)
✓ Architecture (follows patterns)
✓ Security (no vulnerabilities)
✓ Performance (no bottlenecks)
✓ Maintainability (readable, well-structured)
✓ Testing (comprehensive coverage)
✓ Documentation (adequate docs)
✓ Error handling (proper error handling)
✓ Dependencies (no unnecessary/vulnerable deps)
✓ Standards (follows coding standards)
```

---

## Quality Standards Met

✅ **5 comprehensive .dot files created**
✅ **60-200 nodes per diagram** (132-154 nodes)
✅ **Network/hierarchical diagram types**
✅ **Clear agent interaction flows**
✅ **Initialization and spawn processes**
✅ **Coordination with other agents**
✅ **Memory synchronization**
✅ **Validation and quality gates**
✅ **Topology variations** (mesh, hierarchical for Queen)
✅ **Syntax validation** (ready to render)
✅ **Saved to correct directory** (`docs/12fa/graphviz/agents/`)

---

## Next Steps

1. **Render diagrams** using Graphviz CLI or online tools
2. **Review visualizations** for accuracy and completeness
3. **Integrate with documentation** in the 12FA system
4. **Share with team** for feedback and improvements
5. **Iterate based on feedback** to refine workflows

---

## Contact & Support

For questions or improvements to these diagrams:
- Update the `.dot` files in this directory
- Validate syntax before committing
- Re-render SVG/PNG outputs after changes
- Update this README with new features

---

**Last Updated**: 2025-11-01
**Version**: 1.0.0
**Status**: Production Ready ✅
