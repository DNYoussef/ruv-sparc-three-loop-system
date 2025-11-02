# GraphViz Workflow Diagrams

## Overview

This directory contains GraphViz `.dot` files that visualize cascade orchestration patterns. These diagrams help understand workflow structures, data flows, and execution patterns.

## Files

- **workflow.dot**: Comprehensive visualization of all 6 cascade patterns
  - Sequential Pipeline
  - Parallel Fan-Out
  - Conditional Branching
  - Iterative Refinement
  - Codex Sandbox Iteration
  - Hybrid Sequential-Parallel

## Generating Diagrams

### Prerequisites

Install GraphViz:

**Windows** (via Chocolatey):
```bash
choco install graphviz
```

**macOS** (via Homebrew):
```bash
brew install graphviz
```

**Linux** (Debian/Ubuntu):
```bash
sudo apt-get install graphviz
```

### Generate PNG Images

```bash
# From the graphviz directory
dot -Tpng workflow.dot -o workflow.png
```

### Generate SVG (Scalable)

```bash
dot -Tsvg workflow.dot -o workflow.svg
```

### Generate PDF

```bash
dot -Tpdf workflow.dot -o workflow.pdf
```

### Generate All Formats

```bash
# Batch generate all formats
dot -Tpng workflow.dot -o workflow.png
dot -Tsvg workflow.dot -o workflow.svg
dot -Tpdf workflow.dot -o workflow.pdf
```

## Viewing Diagrams

### Online Viewers

If you don't want to install GraphViz:

1. **GraphViz Online**: http://www.webgraphviz.com/
   - Copy contents of `workflow.dot`
   - Paste into editor
   - Click "Generate Graph!"

2. **Edotor**: https://edotor.net/
   - Open `workflow.dot` file
   - Live preview updates automatically

### VS Code Extension

Install "Graphviz Preview" extension:
- Open `workflow.dot`
- Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
- Type "Graphviz: Open Preview"

## Diagram Contents

### Pattern 1: Sequential Pipeline
- **Use Case**: ETL pipelines, data flows, ordered execution
- **Flow**: Input → Extract → Validate → Transform → Load → Output
- **Color**: Blue (#e3f2fd)

### Pattern 2: Parallel Fan-Out
- **Use Case**: Code quality checks, independent tasks, max throughput
- **Flow**: Input → [Lint, Security, Tests, Complexity] → Aggregate → Output
- **Color**: Purple (#f3e5f5)
- **Coordination**: Swarm with 4 agents in mesh topology

### Pattern 3: Conditional Branching
- **Use Case**: Adaptive workflows, risk-based routing, complexity-aware
- **Flow**: Input → Analyze → Decision → [Simple/Medium/Complex paths] → Output
- **Color**: Green (#e8f5e9)

### Pattern 4: Iterative Refinement
- **Use Case**: Quality loops, auto-correction, convergence
- **Flow**: Input → Test → Check → [Pass: Done | Fail: Fix → Repeat] → Output
- **Color**: Orange (#fff3e0)
- **Max Iterations**: 5

### Pattern 5: Codex Sandbox Iteration
- **Use Case**: Safe testing, auto-fixing, regression prevention
- **Flow**: Input → Sandbox → Execute → [Pass: Done | Fail: Codex Fix → Retry]
- **Color**: Pink (#fce4ec)
- **Environment**: Isolated (no network, no filesystem)

### Pattern 6: Hybrid Sequential-Parallel
- **Use Case**: CI/CD pipelines, optimal throughput, dependency handling
- **Flow**: Checkout → [Build Frontend/Backend/Workers] → Integrate → [Deploy US/EU] → Smoke Tests → Production
- **Color**: Teal (#e0f2f1)

## Customizing Diagrams

### Edit Node Colors

```dot
node_name [label="Label", fillcolor="#hexcolor"];
```

### Change Layout Direction

```dot
rankdir=TB;  // Top to Bottom (default)
rankdir=LR;  // Left to Right
rankdir=BT;  // Bottom to Top
rankdir=RL;  // Right to Left
```

### Add New Patterns

```dot
subgraph cluster_mypattern {
    label="My Custom Pattern";
    style=filled;
    color="#f0f0f0";

    node1 [label="Step 1", fillcolor="#90caf9"];
    node2 [label="Step 2", fillcolor="#64b5f6"];

    node1 -> node2 [label="data"];
}
```

### Node Shapes

Available shapes:
- `box`: Rectangle (default)
- `diamond`: Decision point
- `ellipse`: Rounded
- `cylinder`: Database/storage
- `box3d`: 3D box effect
- `note`: Document/note shape

## Best Practices

1. **Use Subgraphs**: Group related nodes with `subgraph cluster_*`
2. **Color Code**: Use consistent colors for pattern types
3. **Add Labels**: Label edges with data flow descriptions
4. **Include Notes**: Use note shapes for explanations
5. **Keep It Simple**: Don't overcrowd diagrams

## Troubleshooting

### "command not found: dot"
- GraphViz not installed. Follow installation instructions above.

### Overlapping Nodes
- Add `rankdir=LR` for left-to-right layout
- Use `{rank=same; node1; node2}` to align nodes

### Missing Arrows
- Check semicolons at end of statements
- Verify node names match exactly

### Rendering Issues
- Try different output formats (PNG, SVG, PDF)
- Simplify complex diagrams into multiple files

## Additional Resources

- **GraphViz Documentation**: https://graphviz.org/documentation/
- **DOT Language Guide**: https://graphviz.org/doc/info/lang.html
- **Node Shapes**: https://graphviz.org/doc/info/shapes.html
- **Color Names**: https://graphviz.org/doc/info/colors.html

## Examples Gallery

After generating diagrams, you'll see:

- **Sequential flow**: Linear data pipeline
- **Parallel execution**: Multi-agent coordination
- **Branching logic**: Conditional routing
- **Iteration loops**: Auto-correction cycles
- **Sandbox isolation**: Safe testing environments
- **Hybrid patterns**: Complex CI/CD workflows

---

**Tip**: Generate SVG format for best quality in documentation and presentations.
