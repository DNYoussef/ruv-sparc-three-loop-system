# 3-Part Self-Dogfooding System

**Status**: Production Ready | **Date**: 2025-11-02
**Purpose**: Continuous improvement through systematic feedback loops across Memory MCP, Connascence Analyzer, and Claude Code

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLAUDE CODE (Orchestrator)                    │
│  - Task spawning                                                 │
│  - Agent coordination                                            │
│  - File operations                                               │
│  - Git workflows                                                 │
└───────────┬─────────────────────────────────────────┬───────────┘
            │                                         │
            │ ①                                       │ ④
            ↓                                         ↓
┌───────────────────────┐                 ┌──────────────────────┐
│    CONNASCENCE         │←────── ② ──────│    MEMORY MCP        │
│    ANALYZER            │                 │                      │
│  - Code quality        │─────── ③ ──────→│  - Persistent memory │
│  - Violation detection │                 │  - Cross-session     │
│  - NASA compliance     │                 │  - Semantic search   │
└───────────────────────┘                 └──────────────────────┘
```

**Flow**:
1. **Claude Code** generates code or coordinates agents
2. **Connascence Analyzer** detects violations and quality issues
3. **Memory MCP** stores findings, fixes, and patterns
4. **Claude Code** retrieves past patterns for improvement

---

## 🔄 Feedback Loop Cycles

### Cycle 1: Quality Detection → Storage
**Trigger**: After code generation/modification
**Duration**: 30-60 seconds

```
Code Generation (Claude Code)
  → Quality Analysis (Connascence)
    → Violation Detection (7 types)
      → Store Findings (Memory MCP)
        → Tag with metadata (agent, project, intent)
```

**Automation**: `dogfood-quality-check.bat`

### Cycle 2: Pattern Retrieval → Application
**Trigger**: Before similar code generation
**Duration**: 10-30 seconds

```
Task Assignment (Claude Code)
  → Query Past Patterns (Memory MCP)
    → Retrieve Similar Fixes
      → Apply Best Practices
        → Generate Improved Code
```

**Automation**: `dogfood-memory-retrieval.bat`

### Cycle 3: Continuous Improvement
**Trigger**: Weekly/on-demand
**Duration**: 5-10 minutes

```
Analyze Trends (Memory MCP query)
  → Identify Common Violations
    → Refactor Codebase (Connascence)
      → Measure Improvement
        → Update Agent Prompts
          → Store Learnings (Memory MCP)
```

**Automation**: `dogfood-continuous-improvement.bat`

---

## 📊 Workflow Examples

### Example 1: God Object Refactoring

**Scenario**: Connascence detects God Object (26 methods > 15 threshold)

```bash
# Step 1: Detect violation
npm run connascence:analyze -- src/server.js
# Output: God Object detected, 26 methods

# Step 2: Store finding
node scripts/store-violation.js --type "God Object" --file "src/server.js" --count 26

# Step 3: Query similar fixes
node scripts/query-fixes.js --type "God Object" --project "memory-mcp"
# Output: 3 past fixes retrieved (delegation pattern, single responsibility)

# Step 4: Apply fix pattern
# Claude Code Task: Refactor using delegation pattern from memory

# Step 5: Verify improvement
npm run connascence:analyze -- src/server-refactored.js
# Output: 12 methods (within threshold)

# Step 6: Store success
node scripts/store-success.js --type "God Object Fix" --improvement "26→12 methods"
```

### Example 2: Parameter Bomb Prevention

**Scenario**: NASA violation detected (14 params > 6 limit)

```bash
# Step 1: Connascence detects CoP violation
npm run connascence:analyze -- src/api-handler.js
# Output: Parameter Bomb, 14 params

# Step 2: Memory retrieval
node scripts/query-fixes.js --type "Parameter Bomb"
# Output: "Use config object pattern" (past fix from 3 weeks ago)

# Step 3: Generate fixed code (Claude Code)
# Automatically applies config object pattern

# Step 4: Verify
npm run connascence:analyze -- src/api-handler-fixed.js
# Output: 2 params (config, context) - PASS

# Step 5: Update metrics
node scripts/update-metrics.js --violation "CoP" --status "fixed"
```

### Example 3: Cross-Project Learning

**Scenario**: Improving Memory MCP based on Connascence patterns

```bash
# Step 1: Analyze Memory MCP codebase
npm run connascence:analyze -- ../memory-mcp-triple-system/src/

# Step 2: Store violations in memory
node scripts/store-violations.js --project "memory-mcp" --scan-results violations.json

# Step 3: Query Connascence Analyzer fixes
node scripts/query-fixes.js --project "connascence-analyzer" --intent "bugfix"
# Output: 12 refactoring patterns that worked

# Step 4: Apply patterns to Memory MCP (Claude Code)
# Task: Apply successful patterns from connascence-analyzer to memory-mcp

# Step 5: Re-analyze
npm run connascence:analyze -- ../memory-mcp-triple-system/src/
# Output: 50% reduction in violations

# Step 6: Store cross-project success
node scripts/store-cross-project.js --source "connascence" --target "memory-mcp"
```

---

## 📈 Metrics to Track

### 1. Code Quality Metrics

**Tracked by**: Connascence Analyzer + Memory MCP storage

| Metric | Description | Target | Current |
|--------|-------------|--------|---------|
| God Object Count | Classes with >15 methods | 0 | TBD |
| Parameter Bomb Count | Functions with >6 params | 0 | TBD |
| Cyclomatic Complexity | Functions with >10 complexity | <5% | TBD |
| Deep Nesting Violations | Nesting >4 levels | 0 | TBD |
| Long Function Count | Functions >50 lines | <10% | TBD |
| Magic Literal Count | Hardcoded values | <20 | TBD |
| Duplicate Code Blocks | Repeated code segments | 0 | TBD |

**Storage Schema** (Memory MCP):
```json
{
  "project": "memory-mcp|connascence-analyzer|claude-code",
  "metric": "God Object Count",
  "value": 2,
  "timestamp": "2025-11-02T10:30:00Z",
  "agent": "code-analyzer",
  "intent": "quality-tracking"
}
```

### 2. Fix Success Rate

**Tracked by**: Memory MCP (before/after analysis)

| Fix Type | Attempted | Successful | Success Rate | Avg Time |
|----------|-----------|------------|--------------|----------|
| God Object Refactor | TBD | TBD | TBD | TBD |
| Parameter Bomb Fix | TBD | TBD | TBD | TBD |
| Complexity Reduction | TBD | TBD | TBD | TBD |
| Nesting Simplification | TBD | TBD | TBD | TBD |

**Query**:
```javascript
// Retrieve all fixes for a specific violation type
const fixes = await vectorSearch({
  query: "God Object refactoring successful",
  limit: 20,
  metadata: { intent: "bugfix", project: "memory-mcp" }
});

// Calculate success rate
const successRate = fixes.filter(f => f.metadata.outcome === "success").length / fixes.length;
```

### 3. Cross-System Usage

**Tracked by**: Memory MCP (cross-project references)

| Source System | Target System | Pattern Transfers | Success Rate |
|---------------|---------------|-------------------|--------------|
| Connascence → Memory MCP | TBD | TBD | TBD |
| Memory MCP → Connascence | TBD | TBD | TBD |
| Claude Code → Both | TBD | TBD | TBD |

**Storage Schema**:
```json
{
  "source_project": "connascence-analyzer",
  "target_project": "memory-mcp",
  "pattern": "delegation for God Object",
  "applied_at": "2025-11-02T11:00:00Z",
  "outcome": "success",
  "improvement": "26 methods → 12 methods",
  "agent": "coder",
  "intent": "refactor"
}
```

### 4. Improvement Velocity

**Tracked by**: Time-series analysis in Memory MCP

| Period | Violations Found | Violations Fixed | Net Change | Velocity |
|--------|------------------|------------------|------------|----------|
| Week 1 | TBD | TBD | TBD | TBD |
| Week 2 | TBD | TBD | TBD | TBD |
| Week 3 | TBD | TBD | TBD | TBD |
| Week 4 | TBD | TBD | TBD | TBD |

**Query**:
```javascript
// Weekly trend analysis
const weeklyMetrics = await vectorSearch({
  query: "violations fixed per week",
  limit: 52, // 1 year
  metadata: { intent: "quality-tracking" }
});

// Calculate velocity (fixes per week)
const velocity = weeklyMetrics.reduce((sum, week) => sum + week.metadata.fixes_count, 0) / weeklyMetrics.length;
```

### 5. Agent Learning Curve

**Tracked by**: Memory MCP (agent-specific performance)

| Agent | Tasks Completed | Violations Created | Violations Fixed | Net Quality Score |
|-------|-----------------|--------------------|--------------------|-------------------|
| coder | TBD | TBD | TBD | TBD |
| reviewer | TBD | TBD | TBD | TBD |
| tester | TBD | TBD | TBD | TBD |

**Storage Schema**:
```json
{
  "agent": "coder",
  "task_id": "AUTH-123",
  "violations_created": 2,
  "violations_fixed": 5,
  "net_score": 3,
  "timestamp": "2025-11-02T12:00:00Z",
  "project": "memory-mcp"
}
```

---

## 🤖 Integration with Claude Code Agents

### Agent Access Patterns

#### 1. Code Quality Agents (14 agents with Connascence access)

**Agents**: `coder`, `reviewer`, `tester`, `code-analyzer`, `functionality-audit`, `theater-detection-audit`, `production-validator`, `sparc-coder`, `analyst`, `backend-dev`, `mobile-dev`, `ml-developer`, `base-template-generator`, `code-review-swarm`

**Workflow**:
```javascript
// Before code generation
Task("coder", "Implement auth feature", {
  hooks: {
    pre_task: [
      "npx claude-flow hooks session-restore",
      "node scripts/query-past-patterns.js --feature auth"
    ],
    post_edit: [
      "npm run connascence:analyze -- $FILE",
      "node scripts/store-analysis.js --file $FILE"
    ],
    post_task: [
      "node scripts/update-metrics.js --agent coder --task $TASK_ID"
    ]
  }
});
```

#### 2. Memory-Only Agents (37 total - 14 code quality = 23 planning agents)

**Agents**: All other agents (coordinators, planning, documentation, etc.)

**Workflow**:
```javascript
// Planning agents use Memory MCP only
Task("hierarchical-coordinator", "Plan multi-agent task", {
  hooks: {
    pre_task: [
      "node scripts/query-past-plans.js --task-type coordination"
    ],
    post_task: [
      "node scripts/store-plan.js --plan-id $PLAN_ID"
    ]
  }
});
```

### Hook Integration

**File**: `C:\Users\17175\hooks\12fa\dogfooding-hooks.js`

```javascript
module.exports = {
  async postEdit(file, agent) {
    // Only code quality agents run Connascence
    const codeQualityAgents = ['coder', 'reviewer', 'tester', ...];

    if (codeQualityAgents.includes(agent)) {
      // Run Connascence analysis
      const violations = await runConnascence(file);

      // Store in Memory MCP
      await memoryStore({
        text: `File ${file} analyzed by ${agent}: ${violations.length} violations`,
        metadata: {
          agent,
          file,
          violations: violations.map(v => v.type),
          timestamp: new Date().toISOString(),
          project: getProject(file),
          intent: 'quality-tracking'
        }
      });
    }

    // All agents store basic metadata
    await memoryStore({
      text: `${agent} edited ${file}`,
      metadata: {
        agent,
        file,
        timestamp: new Date().toISOString(),
        intent: 'tracking'
      }
    });
  }
};
```

---

## 🔧 MCP Coordination

### Topology Selection for Dogfooding

**Recommended**: Mesh topology for maximum cross-communication

```javascript
// Initialize swarm for dogfooding tasks
mcp__claude-flow__swarm_init({
  topology: "mesh",
  maxAgents: 6,
  strategy: "adaptive"
});

// Spawn specialized agents
mcp__claude-flow__agent_spawn({ type: "code-analyzer" }); // Connascence expert
mcp__claude-flow__agent_spawn({ type: "coder" });         // Code generator
mcp__claude-flow__agent_spawn({ type: "tester" });        // Test writer
mcp__claude-flow__agent_spawn({ type: "reviewer" });      // Code reviewer
```

### Task Orchestration

```javascript
// Full dogfooding cycle
mcp__claude-flow__task_orchestrate({
  task: `
    1. Analyze Memory MCP codebase with Connascence
    2. Store violations in Memory MCP
    3. Query past fixes for similar violations
    4. Apply fixes to Memory MCP
    5. Re-analyze and verify improvement
    6. Store success patterns for future use
  `,
  strategy: "sequential", // Must run in order
  maxAgents: 4,
  priority: "high"
});
```

---

## 📊 Metrics Dashboard Specification

### Dashboard Requirements

**Tool**: Grafana + SQLite backend (Memory MCP integration)

#### Panel 1: Real-Time Violation Tracker

**Type**: Time series graph
**Data Source**: Memory MCP vector search
**Query**: `violations detected per day across all projects`

```json
{
  "title": "Violations Over Time",
  "type": "graph",
  "datasource": "Memory MCP",
  "targets": [
    {
      "query": "violations detected",
      "metadata": { "intent": "quality-tracking" },
      "groupBy": "date",
      "aggregate": "count"
    }
  ],
  "yAxis": { "label": "Violation Count" },
  "xAxis": { "label": "Date" }
}
```

#### Panel 2: Fix Success Rate by Agent

**Type**: Bar chart
**Data Source**: Memory MCP
**Query**: `successful fixes per agent`

```json
{
  "title": "Agent Performance",
  "type": "bar",
  "datasource": "Memory MCP",
  "targets": [
    {
      "query": "agent fixes successful",
      "metadata": { "intent": "bugfix" },
      "groupBy": "agent",
      "aggregate": "success_rate"
    }
  ],
  "yAxis": { "label": "Success Rate (%)" },
  "xAxis": { "label": "Agent Name" }
}
```

#### Panel 3: Cross-Project Learning Matrix

**Type**: Heatmap
**Data Source**: Memory MCP
**Query**: `pattern transfers between projects`

```json
{
  "title": "Cross-Project Knowledge Transfer",
  "type": "heatmap",
  "datasource": "Memory MCP",
  "targets": [
    {
      "query": "pattern transfer",
      "metadata": { "intent": "refactor" },
      "matrix": {
        "rows": ["connascence-analyzer", "memory-mcp", "claude-code"],
        "cols": ["connascence-analyzer", "memory-mcp", "claude-code"],
        "value": "transfer_count"
      }
    }
  ]
}
```

#### Panel 4: Code Quality Score Trend

**Type**: Gauge + Time series
**Data Source**: Connascence + Memory MCP
**Query**: `aggregate quality score over time`

```json
{
  "title": "Overall Code Quality",
  "type": "gauge",
  "datasource": "Memory MCP",
  "targets": [
    {
      "query": "code quality score",
      "metadata": { "intent": "quality-tracking" },
      "aggregate": "average",
      "period": "last_7_days"
    }
  ],
  "thresholds": [
    { "value": 0, "color": "red" },
    { "value": 70, "color": "yellow" },
    { "value": 90, "color": "green" }
  ]
}
```

#### Panel 5: Violation Type Distribution

**Type**: Pie chart
**Data Source**: Connascence
**Query**: `violation types with counts`

```json
{
  "title": "Violation Breakdown",
  "type": "pie",
  "datasource": "Connascence",
  "targets": [
    {
      "query": "all violations",
      "groupBy": "type",
      "aggregate": "count"
    }
  ],
  "legend": {
    "position": "right",
    "values": ["God Object", "Parameter Bomb", "Deep Nesting", "Cyclomatic Complexity", "Long Function", "Magic Literal", "Duplicate Code"]
  }
}
```

### Dashboard Export

**File**: `C:\Users\17175\config\grafana\dogfooding-dashboard.json`

```json
{
  "dashboard": {
    "title": "3-Part Dogfooding System",
    "panels": [
      { "id": 1, "title": "Violations Over Time", "type": "graph" },
      { "id": 2, "title": "Agent Performance", "type": "bar" },
      { "id": 3, "title": "Cross-Project Learning", "type": "heatmap" },
      { "id": 4, "title": "Code Quality Score", "type": "gauge" },
      { "id": 5, "title": "Violation Breakdown", "type": "pie" }
    ],
    "refresh": "10s",
    "time": { "from": "now-7d", "to": "now" }
  }
}
```

---

## 🚀 Quick Start

### 1. Run Quality Check (Single Project)

```bash
cd C:\Users\17175\scripts
dogfood-quality-check.bat connascence-analyzer
```

### 2. Retrieve Past Patterns

```bash
dogfood-memory-retrieval.bat "God Object refactoring"
```

### 3. Full Continuous Improvement Cycle

```bash
dogfood-continuous-improvement.bat
```

### 4. View Dashboard

```bash
npm run dashboard:start
# Open http://localhost:3000/dashboards/dogfooding
```

---

## 📝 Best Practices

### 1. Daily Dogfooding Routine

**Morning** (5 minutes):
```bash
# Check overnight violations
dogfood-quality-check.bat all
# Review top 5 violations
node scripts/top-violations.js --count 5
```

**After Major Changes** (2 minutes):
```bash
# Analyze changed files
npm run connascence:analyze -- $CHANGED_FILES
# Store results
node scripts/store-analysis.js --files $CHANGED_FILES
```

**Weekly** (10 minutes):
```bash
# Full continuous improvement cycle
dogfood-continuous-improvement.bat
# Review dashboard
npm run dashboard:open
```

### 2. Agent-Specific Workflows

**For Coder Agent**:
```javascript
// Before coding
Task("coder", "Implement feature X", {
  preWork: "node scripts/query-past-patterns.js --feature X",
  postWork: "npm run connascence:analyze && node scripts/store-analysis.js"
});
```

**For Reviewer Agent**:
```javascript
// Code review
Task("reviewer", "Review PR #123", {
  preWork: "node scripts/query-past-reviews.js --pr 123",
  postWork: "node scripts/store-review.js --pr 123"
});
```

### 3. Cross-Project Learning

**Weekly Task**:
```bash
# Analyze all 3 systems
for project in connascence-analyzer memory-mcp claude-code; do
  npm run connascence:analyze -- ../$project/src
  node scripts/store-violations.js --project $project
done

# Find cross-project patterns
node scripts/cross-project-analysis.js
# Output: "Delegation pattern reduced God Objects by 60% across all projects"
```

---

## 🎯 Success Criteria

### Short-term (1 week)
- [ ] All 3 scripts operational
- [ ] Metrics dashboard deployed
- [ ] At least 10 violation → fix → store cycles completed
- [ ] 1 cross-project pattern transfer successful

### Mid-term (1 month)
- [ ] 50% reduction in God Object violations
- [ ] 80% fix success rate for Parameter Bombs
- [ ] 20+ successful pattern transfers between projects
- [ ] Agent learning curve shows improvement (fewer violations per task)

### Long-term (3 months)
- [ ] <5 total violations across all 3 systems
- [ ] 95%+ fix success rate
- [ ] Automated weekly refactoring with 0 manual intervention
- [ ] Dashboard shows consistent quality improvement trend

---

## 📚 Related Documentation

- `C:\Users\17175\docs\integration-plans\MCP-INTEGRATION-GUIDE.md` - MCP setup
- `C:\Users\17175\hooks\12fa\memory-mcp-tagging-protocol.js` - Tagging protocol
- `C:\Users\17175\docs\integration-plans\CONNASCENCE-INTEGRATION-GUIDE.md` - Connascence setup
- `C:\Users\17175\scripts\` - Executable workflows

---

**Last Updated**: 2025-11-02
**Maintained By**: Claude Code + Memory MCP + Connascence Analyzer (self-dogfooding!)
**Status**: Production Ready - Feedback loops active
