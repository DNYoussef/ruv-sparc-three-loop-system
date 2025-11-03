# Example 2: Creating an Agent-Powered Analysis Skill

## Scenario
You need a skill that analyzes code quality and provides detailed recommendations.

## Step-by-Step Process

### Phase 1: Intent Analysis
**Request**: "Create a skill for analyzing Python code quality"

**Deep Analysis**:
- Surface Intent: Code quality analysis
- Root Cause: Need systematic quality checks
- Context: Pre-commit reviews, code optimization
- Agents Required: code-analyzer, reviewer
- Success: Actionable improvement recommendations

### Phase 2: Use Cases
```yaml
Example 1:
  Input: Python file with god object
  Output: Detected anti-patterns with fix suggestions

Example 2:
  Input: Module with complex functions
  Output: Complexity metrics + refactoring guide
```

### Phase 3: Structure Decision
- Type: Agent-powered skill
- Primary Agent: code-analyzer
- Support: reviewer
- Resources: Analysis scripts, best practices reference

### Phase 4: skill.md Content
```markdown
---
name: analyze-python-code-quality
description: Comprehensive Python code quality analysis using code-analyzer and reviewer agents. Use when reviewing Python files for production readiness.
orchestration:
  primary_agent: code-analyzer
  support_agents: [reviewer]
  coordination: sequential
---

# Python Code Quality Analyzer

Systematic analysis of Python code with actionable improvements.

## Process

### 1. Static Analysis
Run code-analyzer to detect:
- Complexity violations (cyclomatic > 10)
- God objects (>15 methods)
- Parameter bombs (>6 params)
- Deep nesting (>4 levels)

### 2. Best Practices Review
Engage reviewer for:
- PEP 8 compliance
- Type hint coverage
- Documentation quality
- Test coverage gaps

### 3. Generate Report
Create structured report with:
- Severity ratings (critical/warning/info)
- Specific line references
- Fix recommendations
- Refactoring patterns

## Output Format
```yaml
Analysis Results:
  File: {filename}
  Issues:
    - Severity: critical
      Type: god-object
      Line: 45
      Description: Class has 23 methods
      Recommendation: Split into 3 focused classes
```
```

### Phase 5: Create Resources
```python
# resources/scripts/analyze.py
import ast
import complexity

def analyze_file(filepath):
    with open(filepath) as f:
        tree = ast.parse(f.read())
    # Analysis logic
    return results
```

### Phase 6: Create Reference
```markdown
# references/best-practices.md

## Python Quality Standards

### Complexity Limits
- Cyclomatic complexity: ≤10
- Function length: ≤50 lines
- Class methods: ≤15

### NASA Standards
- Parameters: ≤6
- Nesting depth: ≤4
```

### Phase 7: Directory Structure
```
analyze-python-code-quality/
├── skill.md
├── README.md
├── examples/
│   └── example-analysis.md
├── references/
│   └── best-practices.md
├── resources/
│   └── scripts/
│       └── analyze.py
└── graphviz/
    └── workflow.dot
```

**Result**: Production-ready agent-powered skill with complete resources.
