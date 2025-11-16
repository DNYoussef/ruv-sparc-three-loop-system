# Complete Command Catalog: 65+ Slash Commands

## Overview

This catalog documents all 65+ available slash commands across 15 categories, showing how to compose them into powerful cascades.

**Total Commands**: 65+
**Categories**: 15
**Architecture**: Three-tier modular system

## Command Categories

### 1. Analysis Commands (3)

#### /bottleneck-detect
**Purpose**: Analyze performance bottlenecks in swarm operations with auto-fix
```bash
npx claude-flow bottleneck detect [--fix] [--threshold 20]
```
**Cascade Use**: Optimization workflows
**Chains with**: `/topology-optimize`, `/cache-manage`

#### /performance-report
**Purpose**: Comprehensive performance analysis and metrics
```bash
npx claude-flow analysis performance-report [--export report.json]
```
**Cascade Use**: Performance audits
**Chains with**: `/bottleneck-detect`, `/token-usage`

#### /token-usage
**Purpose**: Token optimization analysis
```bash
npx claude-flow analysis token-usage [--time-range 24h]
```
**Cascade Use**: Cost optimization
**Chains with**: `/performance-report`, `/optimization/*`

---

### 2. Automation Commands (3)

#### /auto-agent
**Purpose**: Automatically spawn agents based on task analysis
```bash
npx claude-flow auto agent --task "Build REST API" [--strategy balanced]
```
**Cascade Use**: Intelligent task distribution
**Chains with**: `/task-orchestrate`, `/swarm-init`

#### /smart-spawn
**Purpose**: Intelligent agent spawning with workload prediction
```bash
npx claude-flow automation smart-spawn [--predict-load]
```
**Cascade Use**: Resource optimization
**Chains with**: `/auto-agent`, `/topology-optimize`

#### /workflow-select
**Purpose**: Choose optimal predefined workflow for task
```bash
npx claude-flow automation workflow-select --task "description"
```
**Cascade Use**: Workflow automation
**Chains with**: `/workflow-execute`, `/auto-agent`

---

### 3. Coordination Commands (3)

#### /swarm-init
**Purpose**: Initialize swarm with specified topology
```bash
npx claude-flow coordination swarm-init --topology mesh --max-agents 6
```
**Cascade Use**: Swarm setup
**Chains with**: `/agent-spawn`, `/task-orchestrate`

#### /agent-spawn
**Purpose**: Spawn specific agent type
```bash
npx claude-flow coordination agent-spawn --type coder --capabilities [...]
```
**Cascade Use**: Manual agent creation
**Chains with**: `/swarm-init`, `/task-orchestrate`

#### /task-orchestrate
**Purpose**: Orchestrate task across swarm
```bash
npx claude-flow coordination task-orchestrate --task "description" --strategy adaptive
```
**Cascade Use**: Task distribution
**Chains with**: `/swarm-init`, `/agent-spawn`

---

### 4. GitHub Commands (5)

#### /repo-analyze
**Purpose**: Comprehensive repository analysis
```bash
npx claude-flow github repo-analyze [--repo owner/name]
```
**Cascade Use**: Codebase understanding
**Chains with**: `/code-review`, `/issue-triage`

#### /github-swarm
**Purpose**: Swarm-powered GitHub operations
```bash
npx claude-flow github github-swarm --action <action>
```
**Cascade Use**: Automated GitHub workflows
**Chains with**: `/pr-enhance`, `/code-review`

#### /pr-enhance
**Purpose**: Enhance pull requests with AI
```bash
npx claude-flow github pr-enhance --pr-number 123
```
**Cascade Use**: PR automation
**Chains with**: `/code-review`, `/repo-analyze`

#### /issue-triage
**Purpose**: Intelligent issue triage and categorization
```bash
npx claude-flow github issue-triage [--priority high]
```
**Cascade Use**: Issue management
**Chains with**: `/repo-analyze`, `/task-orchestrate`

#### /code-review
**Purpose**: Swarm-powered code review
```bash
npx claude-flow github code-review --pr-number 123 [--focus security]
```
**Cascade Use**: Quality assurance
**Chains with**: `/pr-enhance`, `/repo-analyze`

---

### 5. Hooks Commands (5)

#### /pre-task
**Purpose**: Execute before task starts
```bash
npx claude-flow hooks pre-task --description "task"
```
**Cascade Use**: Task preparation
**Chains with**: `/task-orchestrate`, `/post-task`

#### /post-task
**Purpose**: Execute after task completes
```bash
npx claude-flow hooks post-task --task-id "id"
```
**Cascade Use**: Task cleanup
**Chains with**: `/pre-task`, `/session-end`

#### /pre-edit
**Purpose**: Execute before file edit
```bash
npx claude-flow hooks pre-edit --file "path"
```
**Cascade Use**: Edit preparation
**Chains with**: `/post-edit`

#### /post-edit
**Purpose**: Execute after file edit
```bash
npx claude-flow hooks post-edit --file "path" --memory-key "key"
```
**Cascade Use**: Edit processing
**Chains with**: `/pre-edit`, `/memory-persist`

#### /session-end
**Purpose**: Clean up session and export metrics
```bash
npx claude-flow hooks session-end [--export-metrics]
```
**Cascade Use**: Session management
**Chains with**: `/memory-persist`, `/performance-report`

---

### 6. Memory Commands (3)

#### /memory-persist
**Purpose**: Persist memory across sessions
```bash
npx claude-flow memory persist [--export memory.json] [--import memory.json]
```
**Cascade Use**: Session continuity
**Chains with**: `/session-end`, `/memory-search`

#### /memory-search
**Purpose**: Search persistent memory
```bash
npx claude-flow memory search --query "query"
```
**Cascade Use**: Context retrieval
**Chains with**: `/memory-persist`, `/task-orchestrate`

#### /memory-usage
**Purpose**: Analyze memory usage and optimization
```bash
npx claude-flow memory memory-usage [--detail summary]
```
**Cascade Use**: Memory optimization
**Chains with**: `/memory-persist`, `/cache-manage`

---

### 7. Monitoring Commands (3)

#### /swarm-monitor
**Purpose**: Real-time swarm activity monitoring
```bash
npx claude-flow monitoring swarm-monitor [--duration 10] [--interval 1]
```
**Cascade Use**: Live monitoring
**Chains with**: `/agent-metrics`, `/bottleneck-detect`

#### /real-time-view
**Purpose**: Live dashboard of swarm operations
```bash
npx claude-flow monitoring real-time-view
```
**Cascade Use**: Operational visibility
**Chains with**: `/swarm-monitor`, `/agent-metrics`

#### /agent-metrics
**Purpose**: Agent performance metrics
```bash
npx claude-flow monitoring agent-metrics [--agent-id id]
```
**Cascade Use**: Agent optimization
**Chains with**: `/swarm-monitor`, `/performance-report`

---

### 8. Optimization Commands (3)

#### /cache-manage
**Purpose**: Optimize caching strategy
```bash
npx claude-flow optimization cache-manage [--clear] [--optimize]
```
**Cascade Use**: Performance tuning
**Chains with**: `/bottleneck-detect`, `/memory-usage`

#### /parallel-execute
**Purpose**: Execute tasks in parallel
```bash
npx claude-flow optimization parallel-execute --tasks [...]
```
**Cascade Use**: Concurrent processing
**Chains with**: `/task-orchestrate`, `/topology-optimize`

#### /topology-optimize
**Purpose**: Optimize swarm topology for workload
```bash
npx claude-flow optimization topology-optimize [--apply] [--target speed]
```
**Cascade Use**: Swarm optimization
**Chains with**: `/bottleneck-detect`, `/swarm-init`

---

### 9. Training Commands (3)

#### /neural-train
**Purpose**: Train neural agents with sample tasks
```bash
npx claude-flow training neural-train [--iterations 10]
```
**Cascade Use**: Agent improvement
**Chains with**: `/pattern-learn`, `/model-update`

#### /pattern-learn
**Purpose**: Learn cognitive patterns from usage
```bash
npx claude-flow training pattern-learn [--pattern convergent]
```
**Cascade Use**: Pattern recognition
**Chains with**: `/neural-train`, `/model-update`

#### /model-update
**Purpose**: Update agent models with learned patterns
```bash
npx claude-flow training model-update [--agent-id id]
```
**Cascade Use**: Continuous improvement
**Chains with**: `/neural-train`, `/pattern-learn`

---

### 10. Workflow Commands (3)

#### /workflow-execute
**Purpose**: Execute predefined workflow
```bash
npx claude-flow workflows workflow-execute --workflow-id id
```
**Cascade Use**: Workflow automation
**Chains with**: `/workflow-create`, `/task-orchestrate`

#### /workflow-create
**Purpose**: Create new workflow definition
```bash
npx claude-flow workflows workflow-create --name "name" --steps [...]
```
**Cascade Use**: Workflow design
**Chains with**: `/workflow-execute`, `/workflow-export`

#### /workflow-export
**Purpose**: Export workflow definition
```bash
npx claude-flow workflows workflow-export --workflow-id id --output file
```
**Cascade Use**: Workflow sharing
**Chains with**: `/workflow-create`, `/memory-persist`

---

### 11. SPARC Commands (20)

#### /sparc:architect
**Purpose**: System architecture design specialist
```bash
# Via slash command system
```
**Cascade Use**: Design phase
**Chains with**: `/sparc:code`, `/sparc:spec-pseudocode`

#### /sparc:code
**Purpose**: Implementation specialist
```bash
# Via slash command system
```
**Cascade Use**: Coding phase
**Chains with**: `/sparc:tdd`, `/sparc:architect`

#### /sparc:tdd
**Purpose**: Test-Driven Development specialist
```bash
# Via slash command system
```
**Cascade Use**: Testing phase
**Chains with**: `/sparc:code`, `/functionality-audit`

#### /sparc:debug
**Purpose**: Debugging specialist
```bash
# Via slash command system
```
**Cascade Use**: Bug fixing
**Chains with**: `/agent-rca`, `/codex-auto`

#### /sparc:security-review
**Purpose**: Security audit specialist
```bash
# Via slash command system
```
**Cascade Use**: Security checks
**Chains with**: `/code-review`, `/audit-pipeline`

#### /sparc:docs-writer
**Purpose**: Documentation specialist
```bash
# Via slash command system
```
**Cascade Use**: Documentation
**Chains with**: `/sparc:architect`, `/gemini-media`

#### /sparc:devops
**Purpose**: DevOps and deployment specialist
```bash
# Via slash command system
```
**Cascade Use**: Deployment
**Chains with**: `/audit-pipeline`, `/github-swarm`

#### /sparc:integration
**Purpose**: System integration specialist
```bash
# Via slash command system
```
**Cascade Use**: Integration testing
**Chains with**: `/functionality-audit`, `/sparc:tdd`

#### /sparc:spec-pseudocode
**Purpose**: Specification and pseudocode phase
```bash
# Via slash command system
```
**Cascade Use**: Planning phase
**Chains with**: `/sparc:architect`, `/sparc:code`

#### /sparc:refinement-optimization-mode
**Purpose**: Code refinement and optimization
```bash
# Via slash command system
```
**Cascade Use**: Optimization phase
**Chains with**: `/style-audit`, `/bottleneck-detect`

(Plus 10 more SPARC commands...)

---

### 12. Audit Commands (4) - Created

#### /theater-detect
**Purpose**: Find mocks, TODOs, placeholders
```bash
/theater-detect src/ [--fix]
```
**Cascade Use**: Phase 1 of audit pipeline
**Chains with**: `/functionality-audit`, `/style-audit`

#### /functionality-audit
**Purpose**: Test with Codex auto-fix iteration
```bash
/functionality-audit src/ --model codex-auto [--max-iterations 5]
```
**Cascade Use**: Phase 2 of audit pipeline (critical!)
**Chains with**: `/theater-detect`, `/style-audit`

#### /style-audit
**Purpose**: Lint and polish to standards
```bash
/style-audit src/
```
**Cascade Use**: Phase 3 of audit pipeline
**Chains with**: `/functionality-audit`, `/audit-pipeline`

#### /audit-pipeline
**Purpose**: Complete 3-phase audit
```bash
/audit-pipeline src/ [--phase all] [--model codex-auto]
```
**Cascade Use**: Complete quality workflow
**Chains with**: All audit commands

---

### 13. Multi-Model Commands (7) - Created

#### /gemini-megacontext
**Purpose**: 1M token context analysis
```bash
/gemini-megacontext "Analyze entire codebase" --context src/
```
**Cascade Use**: Large codebase analysis
**Chains with**: `/agent-architect`, `/gemini-media`

#### /gemini-search
**Purpose**: Real-time web information
```bash
/gemini-search "Latest React 19 best practices"
```
**Cascade Use**: Research phase
**Chains with**: `/codex-auto`, `/agent-architect`

#### /gemini-media
**Purpose**: Generate images/videos
```bash
/gemini-media "Architecture diagram" --output diagram.png
```
**Cascade Use**: Visualization
**Chains with**: `/agent-architect`, `/sparc:docs-writer`

#### /gemini-extensions
**Purpose**: Figma, Stripe, Postman integration
```bash
/gemini-extensions "Extract Figma design" --extension figma
```
**Cascade Use**: External tool integration
**Chains with**: `/codex-auto`, `/gemini-media`

#### /codex-auto
**Purpose**: Rapid sandboxed prototyping
```bash
/codex-auto "Implement feature" --sandbox true
```
**Cascade Use**: Fast implementation, auto-fixing
**Chains with**: `/functionality-audit`, `/gemini-search`

#### /codex-reasoning
**Purpose**: GPT-5-Codex alternative reasoning
```bash
/codex-reasoning "Alternative algorithm for sorting"
```
**Cascade Use**: Second opinion, alternatives
**Chains with**: `/agent-architect`, `/codex-auto`

#### /multi-model
**Purpose**: Intelligent AI orchestrator
```bash
/multi-model "Task description" --auto-route
```
**Cascade Use**: Smart routing to best AI
**Chains with**: All AI commands

---

### 14. Agent Commands (1) - Created

#### /agent-rca
**Purpose**: Root cause analysis specialist
```bash
/agent-rca "Bug description" --context src/ --depth deep
```
**Cascade Use**: Systematic debugging
**Chains with**: `/codex-auto`, `/functionality-audit`

---

### 15. Workflow Creation Commands (2) - Created

#### /create-micro-skill
**Purpose**: Create atomic micro-skill
```bash
/create-micro-skill "Validate JSON schemas" --technique program-of-thought
```
**Cascade Use**: Skill creation
**Chains with**: `/create-cascade`

#### /create-cascade
**Purpose**: Create workflow cascade
```bash
/create-cascade "Quality pipeline" --stages "..."
```
**Cascade Use**: Workflow creation
**Chains with**: `/workflow-execute`

---

## Cascade Composition Patterns

### Pattern 1: Complete Development Lifecycle
```bash
# Research → design → implement → test → deploy
/gemini-search "Best practices for feature X"
/sparc:architect "Design feature X with best practices"
/codex-auto "Implement designed feature"
/functionality-audit --model codex-auto
/sparc:tdd "Create comprehensive tests"
/style-audit
/sparc:docs-writer "Document feature"
/sparc:devops "Deploy to staging"
```

### Pattern 2: Performance Optimization
```bash
# Analyze → optimize → validate
/performance-report --export baseline.json
/bottleneck-detect --threshold 15
/topology-optimize --apply --target speed
/cache-manage --optimize
/parallel-execute --tasks [optimization tasks]
/performance-report --export optimized.json
# Compare baseline vs optimized
```

### Pattern 3: GitHub Workflow Automation
```bash
# Analyze → review → enhance → merge
/repo-analyze --repo owner/name
/issue-triage --priority high
/pr-enhance --pr-number 123
/code-review --pr-number 123 --focus security --suggest-fixes
/audit-pipeline src/ --output quality-report.json
/github-swarm --action auto-merge-if-passing
```

### Pattern 4: Intelligent Bug Fix
```bash
# Debug → fix → test → optimize
/agent-rca "Bug description" --depth deep
/sparc:debug "Fix identified issue"
/codex-auto "Implement fix from RCA"
/functionality-audit --model codex-auto --max-iterations 10
/sparc:integration "Test integration"
/style-audit
```

### Pattern 5: Legacy Modernization
```bash
# Analyze → detect → fix → test → optimize
/gemini-megacontext "Analyze entire legacy codebase" --context src/
/theater-detect src/ --fix
/sparc:architect "Design modernization plan"
/codex-auto "Refactor to modern patterns"
/functionality-audit --model codex-auto
/style-audit
/performance-report --export modernized.json
/gemini-media "Generate new architecture diagram"
/sparc:docs-writer "Update documentation"
```

### Pattern 6: Continuous Improvement
```bash
# Train → analyze → optimize → persist
/neural-train --iterations 20
/pattern-learn --pattern adaptive
/model-update --all-agents
/performance-report
/bottleneck-detect --fix
/memory-persist --export improved-state.json
```

### Pattern 7: Multi-Model Research & Build
```bash
# Parallel research + design → implement → validate
parallel ::: \
  "/gemini-search 'Latest framework best practices'" \
  "/gemini-megacontext 'Analyze existing patterns' --context src/" \
  "/agent-architect 'Design system architecture'"
# Results merged, then:
/codex-auto "Implement from combined research and design"
/functionality-audit --model codex-auto
/style-audit
```

## Command Combination Matrix

| From → To | audit-pipeline | codex-auto | gemini-search | agent-rca | swarm-init |
|-----------|---------------|------------|---------------|-----------|------------|
| **gemini-search** | ❌ | ✅ Prototype | N/A | ❌ | ❌ |
| **agent-rca** | ❌ | ✅ Fix | ❌ | N/A | ❌ |
| **codex-auto** | ✅ Test | N/A | ❌ | ❌ | ❌ |
| **theater-detect** | ✅ Pipeline | ✅ Fix | ❌ | ❌ | ❌ |
| **sparc:architect** | ❌ | ✅ Implement | ✅ Research | ❌ | ❌ |

✅ = High value combination
❌ = Not typically combined

## Usage Statistics

Based on typical development workflows:

**Most Used Commands** (Top 10):
1. `/functionality-audit` (with Codex iteration)
2. `/codex-auto` (rapid prototyping)
3. `/audit-pipeline` (complete quality)
4. `/gemini-search` (research)
5. `/agent-rca` (debugging)
6. `/style-audit` (polish)
7. `/theater-detect` (cleanup)
8. `/sparc:architect` (design)
9. `/code-review` (GitHub PR)
10. `/gemini-megacontext` (large context)

**Best Cascade Starters**:
- `/gemini-search` - Research-driven
- `/agent-rca` - Bug-driven
- `/theater-detect` - Cleanup-driven
- `/repo-analyze` - GitHub-driven
- `/gemini-megacontext` - Analysis-driven

**Best Cascade Finishers**:
- `/style-audit` - Final polish
- `/sparc:docs-writer` - Documentation
- `/memory-persist` - Save state
- `/performance-report` - Metrics
- `/session-end` - Cleanup

## Creating Custom Cascades

### Step 1: Identify Goal
```
What's the end goal?
- Feature development
- Bug fixing
- Performance optimization
- Code quality
- Documentation
```

### Step 2: Select Commands
```
Choose from catalog based on:
- Category relevance
- Chaining compatibility
- Multi-model opportunities
```

### Step 3: Define Sequence
```
Sequential: cmd1 → cmd2 → cmd3
Parallel: [cmd1 + cmd2 + cmd3] → merge
Conditional: cmd1 && cmd2 || cmd3
```

### Step 4: Test & Refine
```
Run cascade on test code
Measure results
Optimize sequence
Add error handling
```

## Command Discovery

### List All Commands
```bash
find ~/.claude/commands -name "*.md" ! -name "README.md" | wc -l
# Returns: 65+
```

### Commands by Category
```bash
ls ~/.claude/commands/
# Shows: analysis, automation, coordination, github, etc.
```

### Command Details
```bash
cat ~/.claude/commands/<category>/<command>.md
```

## Summary

**Total Available**: 65+ commands
**Categories**: 15
**Primary Innovation**: Cascades = command sequences
**Secret Sauce**: Codex iteration in `/functionality-audit`
**Best Practice**: Compose simple commands into powerful workflows

Start with audit-pipeline, explore multi-model commands, then build custom cascades for your specific needs!
