# Functionality Audit GraphViz Diagrams

This directory contains GraphViz DOT files that visualize the functionality-audit skill workflow and debugging decision trees.

## Diagrams

### 1. `workflow.dot` - Main Process Flow
**Purpose**: Complete visualization of the 6-step functionality audit workflow

**Key Components**:
- **Step 1: Create Sandbox** - Isolated E2B environment with resource limits
- **Step 2: Generate Tests** - Comprehensive test case generation including edge cases
- **Step 3: Execute & Monitor** - Test execution with stdout/stderr capture
- **Step 4: Analyze Results** - Compare expected vs actual, categorize failures
- **Step 5: Debug & Fix** - Systematic investigation and code changes
- **Step 6: Verify Fix** - Regression testing and validation

**Integrations Shown**:
- E2B Sandboxes (code execution environment)
- Flow-Nexus (swarm coordination)
- Memory-MCP (result persistence and pattern learning)

**Metrics**:
- Code coverage
- Performance metrics
- Quality scores

### 2. `debugging-decision-tree.dot` - Bug Resolution Strategy
**Purpose**: Decision tree for choosing optimal debugging approach based on bug characteristics

**Decision Points**:

#### Level 1: Reproducibility
- **Reproducible** → Proceed to categorization
- **Non-Reproducible** → Race conditions, timing issues, stress testing approach

#### Level 2: Bug Category (for reproducible bugs)

1. **Logic Error**
   - Incorrect algorithms, wrong conditions
   - Strategy: Breakpoints, step-through debugging, unit tests

2. **Integration Issue**
   - API contract mismatches, dependency failures
   - Strategy: Mock dependencies, test isolation, contract validation

3. **Performance Issue**
   - Timeouts, memory leaks, CPU bottlenecks
   - Strategy: Profiling, algorithmic optimization, caching

4. **Correctness Issue**
   - Wrong output, type mismatches, validation failures
   - Strategy: Input/output validation, property-based testing

#### Level 3: Verification
- **Fix Verified** → Success (all tests pass, no regression)
- **Regression Detected** → Revert, isolate side effects, retry

**Tools Referenced**:
- Static Analysis (TypeScript, ESLint, SonarQube)
- Dynamic Analysis (Debuggers, profilers, memory analyzers)
- Testing Tools (Jest/Mocha, property testing, E2E frameworks)
- Monitoring (APM, error tracking, log aggregation)

**Best Practices**:
1. Isolate the problem (minimal reproduction)
2. Form hypothesis (test systematically)
3. Fix incrementally (one change at a time)
4. Add tests first (prevent regression)

## Rendering the Diagrams

### Using Graphviz CLI

```bash
# Install Graphviz
# Windows (Chocolatey): choco install graphviz
# macOS (Homebrew): brew install graphviz
# Ubuntu/Debian: sudo apt-get install graphviz

# Render as PNG
dot -Tpng workflow.dot -o workflow.png
dot -Tpng debugging-decision-tree.dot -o debugging-decision-tree.png

# Render as SVG (scalable)
dot -Tsvg workflow.dot -o workflow.svg
dot -Tsvg debugging-decision-tree.dot -o debugging-decision-tree.svg

# Render as PDF
dot -Tpdf workflow.dot -o workflow.pdf
dot -Tpdf debugging-decision-tree.dot -o debugging-decision-tree.pdf
```

### Using Online Renderers

1. **GraphvizOnline**: https://dreampuf.github.io/GraphvizOnline/
   - Copy/paste DOT file contents
   - Instant rendering in browser
   - Export as SVG/PNG

2. **Viz.js**: https://viz-js.com/
   - WebAssembly-based Graphviz renderer
   - No server-side processing

3. **Graphviz Visual Editor**: http://magjac.com/graphviz-visual-editor/
   - Interactive editing
   - Real-time preview

## Integration with Documentation

These diagrams can be referenced in:

1. **Skill README** (`../README.md`)
   ```markdown
   ## Workflow Visualization
   ![Functionality Audit Workflow](graphviz/workflow.png)

   ## Debugging Decision Tree
   ![Bug Resolution Strategy](graphviz/debugging-decision-tree.png)
   ```

2. **Developer Documentation**
   - Include in training materials
   - Reference in debugging guides
   - Link from error messages

3. **Agent Prompts**
   - Show agents the workflow context
   - Guide debugging strategy selection
   - Visualize coordination patterns

## Color Coding

### Workflow Diagram (`workflow.dot`)
- **Blue** (#1976D2) - Start/initialization
- **Orange** (#F57C00) - Sandbox/environment operations
- **Purple** (#7B1FA2) - Test generation
- **Green** (#388E3C) - Execution/success
- **Yellow** (#F9A825) - Analysis
- **Red** (#C62828) - Debugging/errors

### Debugging Decision Tree (`debugging-decision-tree.dot`)
- **Red** (#C62828) - Errors/bugs
- **Orange** (#E65100) - Decision points
- **Blue** (#1976D2) - Logic errors
- **Purple** (#7B1FA2) - Integration issues
- **Yellow** (#F9A825) - Performance issues
- **Green** (#388E3C) - Correctness/success

## Maintenance

When updating these diagrams:

1. **Keep consistency** with the skill implementation
2. **Update colors** to match established palette
3. **Test rendering** before committing (multiple formats)
4. **Document changes** in git commit messages
5. **Regenerate images** after DOT file changes

## Advanced Usage

### Automated Rendering in CI/CD

```yaml
# .github/workflows/render-diagrams.yml
name: Render GraphViz Diagrams

on:
  push:
    paths:
      - '**/*.dot'

jobs:
  render:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Graphviz
        run: sudo apt-get install -y graphviz
      - name: Render diagrams
        run: |
          find . -name "*.dot" -exec sh -c 'dot -Tpng "$1" -o "${1%.dot}.png"' _ {} \;
          find . -name "*.dot" -exec sh -c 'dot -Tsvg "$1" -o "${1%.dot}.svg"' _ {} \;
      - name: Commit rendered images
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add **/*.png **/*.svg
          git commit -m "Auto-render GraphViz diagrams" || echo "No changes"
          git push
```

### Dynamic Diagram Generation

For runtime visualization, use Node.js libraries:

```javascript
const { graphviz } = require('node-graphviz');
const fs = require('fs');

async function renderDiagram(dotFile, outputFormat = 'svg') {
  const dotContent = fs.readFileSync(dotFile, 'utf8');
  const svg = await graphviz.dot(dotContent, outputFormat);
  return svg;
}

// Usage in agent prompts
const workflowSvg = await renderDiagram('workflow.dot', 'svg');
// Embed in HTML documentation or agent UI
```

## Related Files

- `../README.md` - Main functionality-audit skill documentation
- `../skill.yaml` - Skill configuration and metadata
- `../../docs/TESTING-COMPLETE-SUMMARY.md` - Testing methodology docs
- `../../docs/workflows/graphviz/` - Other workflow diagrams

## Questions?

For questions about these diagrams:
1. Check the skill documentation (`../README.md`)
2. Review the skill implementation
3. Consult the GraphViz documentation: https://graphviz.org/documentation/
4. Open an issue in the repository
