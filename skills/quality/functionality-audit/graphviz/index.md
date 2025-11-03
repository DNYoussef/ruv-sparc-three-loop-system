# Functionality Audit GraphViz Diagrams - Index

**Created**: 2025-11-02
**Location**: `C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\skills\functionality-audit\graphviz\`
**Purpose**: Visual process documentation for the functionality-audit skill

---

## 📊 Diagram Catalog

### 1. Main Workflow (`workflow.dot`)

**File**: `workflow.dot`
**Type**: Process Flow Diagram
**Complexity**: Medium (20+ nodes, 3 clusters)
**Dimensions**: ~1200x1600px (recommended render)

**Description**: Complete 6-step functionality audit workflow with external integrations

**Nodes**:
- 1 start node (ellipse)
- 6 main process steps (rounded boxes)
- 1 decision diamond
- 1 completion node
- 3 integration tools (E2B, Flow-Nexus, Memory-MCP)
- 3 metrics nodes (coverage, performance, quality)

**Clusters**:
1. **External Integrations** - E2B sandboxes, Flow-Nexus orchestration, Memory-MCP storage
2. **Metrics & Monitoring** - Coverage tracking, performance metrics, quality scores
3. **Legend** - Visual guide to diagram elements

**Use Cases**:
- Onboarding new developers to the audit process
- Training agents on workflow execution
- Documentation for stakeholders
- Process optimization discussions

**Render Recommendations**:
```bash
# High-quality PNG for documentation
dot -Tpng -Gdpi=300 workflow.dot -o workflow-hq.png

# SVG for web/interactive use
dot -Tsvg workflow.dot -o workflow.svg

# PDF for presentations
dot -Tpdf workflow.dot -o workflow.pdf
```

---

### 2. Debugging Decision Tree (`debugging-decision-tree.dot`)

**File**: `debugging-decision-tree.dot`
**Type**: Decision Tree Diagram
**Complexity**: High (40+ nodes, 3 clusters)
**Dimensions**: ~1400x2000px (recommended render)

**Description**: Systematic bug resolution strategy based on reproducibility and category

**Decision Hierarchy**:

```
Bug Detected
    ├─ Reproducible?
    │   ├─ YES → Categorize
    │   │   ├─ Logic Error → Static debugging
    │   │   ├─ Integration Issue → Contract testing
    │   │   ├─ Performance Issue → Profiling
    │   │   └─ Correctness Issue → Property testing
    │   └─ NO → Non-reproducible strategies
    │       └─ Race conditions, timing, stress testing
    └─ Fix Applied → Verify
        ├─ Verified → Success ✓
        └─ Regression → Revert and retry
```

**Clusters**:
1. **Debugging Tools & Techniques** - Static analysis, dynamic analysis, testing tools, monitoring
2. **Best Practices** - 4-step systematic debugging methodology
3. **Decision Tree Legend** - Visual guide to node types

**Use Cases**:
- Guiding agent debugging strategy selection
- Training developers on systematic bug resolution
- Standardizing debugging approaches across teams
- Post-mortem analysis framework

**Render Recommendations**:
```bash
# High-quality PNG for training materials
dot -Tpng -Gdpi=300 debugging-decision-tree.dot -o debug-tree-hq.png

# SVG for interactive decision tool
dot -Tsvg debugging-decision-tree.dot -o debug-tree.svg

# PDF for printed reference guides
dot -Tpdf debugging-decision-tree.dot -o debug-tree.pdf
```

---

## 🎨 Visual Design System

### Color Palette

| Color | Hex Code | Usage |
|-------|----------|-------|
| **Blue** | #1976D2 | Start/initialization, logic errors |
| **Orange** | #F57C00 | Sandbox operations, decision points |
| **Purple** | #7B1FA2 | Test generation, integration issues |
| **Green** | #388E3C | Execution, success, correctness |
| **Yellow** | #F9A825 | Analysis, performance issues |
| **Red** | #C62828 | Debugging, errors, failures |
| **Light Blue** | #E3F2FD | Backgrounds, clusters |

### Node Styles

- **Ellipse** - Start/end states
- **Rounded Box** - Process steps
- **Diamond** - Decision points
- **Box (dashed)** - External integrations
- **Box (dotted)** - Metrics/monitoring

### Edge Styles

- **Solid (penwidth=2)** - Primary workflow
- **Solid (penwidth=3)** - Success path
- **Dashed** - Integration connections
- **Dotted** - Metrics/monitoring connections

---

## 📁 File Structure

```
graphviz/
├── INDEX.md                        (this file)
├── README.md                       (usage guide)
├── workflow.dot                    (main process flow)
├── debugging-decision-tree.dot     (bug resolution strategy)
├── workflow.png                    (rendered PNG - auto-generated)
├── workflow.svg                    (rendered SVG - auto-generated)
├── debugging-decision-tree.png     (rendered PNG - auto-generated)
└── debugging-decision-tree.svg     (rendered SVG - auto-generated)
```

**Note**: `.png` and `.svg` files are auto-generated from `.dot` files. Always edit `.dot` source files.

---

## 🔧 Quick Commands

### Render All Diagrams

```bash
# Navigate to directory
cd "C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\skills\functionality-audit\graphviz"

# Render all DOT files to PNG and SVG
for file in *.dot; do
    dot -Tpng "$file" -o "${file%.dot}.png"
    dot -Tsvg "$file" -o "${file%.dot}.svg"
done
```

### Validate DOT Syntax

```bash
# Check syntax without rendering
dot -Tplain workflow.dot > /dev/null
dot -Tplain debugging-decision-tree.dot > /dev/null
```

### Preview in Browser (using SVG)

```bash
# Render to SVG and open
dot -Tsvg workflow.dot -o workflow.svg
start workflow.svg  # Windows
# open workflow.svg  # macOS
# xdg-open workflow.svg  # Linux
```

---

## 🔗 Integration Points

### 1. Skill README

**File**: `../README.md`

```markdown
## Visual Workflow

The functionality-audit skill follows a 6-step systematic process:

![Workflow Diagram](graphviz/workflow.svg)

For detailed debugging strategies, see the [debugging decision tree](graphviz/debugging-decision-tree.svg).
```

### 2. Agent Prompts

**Usage in agent initialization**:

```javascript
const workflowContext = `
Refer to the workflow diagram at:
${__dirname}/graphviz/workflow.svg

Current step: ${currentStep}
Next actions: ${getNextActions(currentStep)}
`;
```

### 3. Documentation Sites

**Embed in Markdown**:

```markdown
<!-- For GitHub/GitLab -->
![Functionality Audit Workflow](./graphviz/workflow.png)

<!-- For documentation sites with base URL -->
![Functionality Audit Workflow](/skills/functionality-audit/graphviz/workflow.svg)
```

---

## 📊 Diagram Metrics

### Workflow Diagram Complexity

| Metric | Value |
|--------|-------|
| Total Nodes | 20 |
| Total Edges | 25 |
| Clusters | 3 |
| Decision Points | 1 |
| Max Depth | 7 levels |
| Estimated Render Time | 0.5s |

### Debugging Tree Complexity

| Metric | Value |
|--------|-------|
| Total Nodes | 42 |
| Total Edges | 48 |
| Clusters | 3 |
| Decision Points | 3 |
| Max Depth | 5 levels |
| Estimated Render Time | 0.8s |

---

## 🔄 Maintenance Schedule

### When to Update

1. **Skill Logic Changes**
   - New steps added to workflow
   - Step sequence modified
   - Integration points changed

2. **Bug Category Changes**
   - New bug types identified
   - Debugging strategies updated
   - Tool recommendations changed

3. **Visual Design Updates**
   - Color palette refinement
   - Layout improvements
   - Accessibility enhancements

### Update Checklist

- [ ] Edit `.dot` source file
- [ ] Validate syntax (`dot -Tplain file.dot`)
- [ ] Re-render to PNG/SVG
- [ ] Update README.md if new elements added
- [ ] Update INDEX.md metrics if complexity changed
- [ ] Commit with descriptive message
- [ ] Notify dependent documentation maintainers

---

## 📚 Related Documentation

1. **Skill Documentation**
   - `../README.md` - Main functionality-audit docs
   - `../skill.yaml` - Skill configuration

2. **Testing Documentation**
   - `../../docs/TESTING-COMPLETE-SUMMARY.md`
   - `../../docs/production-readiness-completion-summary.md`

3. **Other GraphViz Diagrams**
   - `../../docs/workflows/graphviz/` - System-wide workflow diagrams
   - `../../docs/workflows/graphviz/agent-mappings/` - Agent coordination diagrams

4. **External Resources**
   - [Graphviz Documentation](https://graphviz.org/documentation/)
   - [DOT Language Guide](https://graphviz.org/doc/info/lang.html)
   - [Node Shapes Reference](https://graphviz.org/doc/info/shapes.html)
   - [Color Names](https://graphviz.org/doc/info/colors.html)

---

## 🎯 Success Criteria

These diagrams are successful if they:

1. ✅ **Clarity** - Non-technical stakeholders understand the workflow
2. ✅ **Completeness** - All major steps and decision points represented
3. ✅ **Accuracy** - Match actual skill implementation
4. ✅ **Maintainability** - Easy to update when process changes
5. ✅ **Accessibility** - Readable in both color and grayscale
6. ✅ **Integration** - Embedded in documentation and agent prompts

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: "Graphviz not found"
```bash
# Windows (Chocolatey)
choco install graphviz

# macOS (Homebrew)
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz
```

**Issue**: "DOT syntax error"
```bash
# Validate syntax
dot -Tplain workflow.dot > /dev/null

# Common errors:
# - Missing semicolon after node/edge
# - Unmatched quotes
# - Invalid node names (use quotes for spaces)
```

**Issue**: "Rendering takes too long"
```bash
# Use simpler layout engine
neato -Tpng workflow.dot -o workflow.png  # Force-directed
circo -Tpng workflow.dot -o workflow.png  # Circular
```

**Issue**: "Text overlapping in rendered diagram"
```bash
# Increase node spacing
dot -Tpng -Gnodesep=1.0 -Granksep=2.0 workflow.dot -o workflow.png
```

---

**Last Updated**: 2025-11-02
**Maintained By**: Claude Code (Functionality Audit Skill Team)
**Review Schedule**: Quarterly or on major skill updates
