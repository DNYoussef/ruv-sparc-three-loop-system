# Functionality Audit GraphViz - Quick Reference Card

**🎯 Purpose**: Visual documentation for 6-step validation workflow + debugging decision tree

---

## 📊 Two Diagrams

### 1️⃣ `workflow.dot` - Main Process
**6 Steps**: Sandbox → Tests → Execute → Analyze → Debug → Verify

**Integrations**: E2B + Flow-Nexus + Memory-MCP

**Render**: `dot -Tsvg workflow.dot -o workflow.svg`

---

### 2️⃣ `debugging-decision-tree.dot` - Bug Resolution
**3 Decision Levels**:
1. Reproducible? (Yes/No)
2. Category? (Logic/Integration/Performance/Correctness)
3. Verified? (Success/Regression)

**Render**: `dot -Tsvg debugging-decision-tree.dot -o debugging-decision-tree.svg`

---

## 🚀 Quick Commands

```bash
# Render all to PNG + SVG
for f in *.dot; do dot -Tpng "$f" -o "${f%.dot}.png"; dot -Tsvg "$f" -o "${f%.dot}.svg"; done

# High-quality PNG (300 DPI)
dot -Tpng -Gdpi=300 workflow.dot -o workflow-hq.png

# Validate syntax
dot -Tplain workflow.dot > /dev/null
```

---

## 🎨 Color Key

| Color | Meaning |
|-------|---------|
| 🔵 Blue | Start/logic |
| 🟠 Orange | Sandbox/decisions |
| 🟣 Purple | Tests/integration |
| 🟢 Green | Success/execution |
| 🟡 Yellow | Analysis/performance |
| 🔴 Red | Errors/debugging |

---

## 🔗 Integration

**In Markdown**:
```markdown
![Workflow](graphviz/workflow.svg)
```

**In Agent Prompts**:
```javascript
const workflow = readFileSync('graphviz/workflow.svg', 'utf8');
```

---

## 📚 Full Docs

- **README.md** - Complete usage guide
- **INDEX.md** - Detailed catalog + metrics
- **../README.md** - Skill documentation

---

**Created**: 2025-11-02 | **Files**: 4 | **Total Size**: ~32KB
