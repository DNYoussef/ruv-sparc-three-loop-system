# Testing & Quality Skills Rewrite Summary

## Overview

This document summarizes the rewrite of 7 Testing & Quality skills using the skill-forge methodology with trigger-first naming, SOP structure, and proper agent orchestration.

## Completed Rewrites

### 1. when-validating-deployment-readiness-use-production-readiness ✅

**Original**: `production-readiness`
**Agent Orchestration**: 8-10 agents (hybrid parallel-sequential)
**Execution Pattern**: 10 phases with quality gate enforcement

**Key Improvements**:
- Trigger-first naming for discoverability
- SOP structure with 10 distinct phases
- Mapped to agents: theater-detection-auditor, functionality-auditor, style-auditor, security-manager, performance-analyzer, qa-engineer, api-documentation-specialist, devops-engineer, production-validator, reviewer
- Memory coordination pattern with phase-based namespace
- Hook integration at pre-task, per-agent, and post-task
- Complete deployment checklist generation
- Quality gate thresholds configurable
- CI/CD integration with GitHub Actions example

**File**: `when-validating-deployment-readiness-use-production-readiness.md`

---

### 2. when-need-fast-validation-use-quick-quality-check ✅

**Original**: `quick-quality-check`
**Agent Orchestration**: 4 agents (full parallel execution)
**Execution Pattern**: Mesh topology, <30 seconds

**Key Improvements**:
- Trigger-first naming emphasizing speed need
- Parallel execution with no sequential dependencies
- Mapped to agents: theater-detection-auditor, code-analyzer, security-manager, tester
- Real-time result aggregation as agents complete
- Severity-ranked unified quality report
- Performance benchmarks documented
- Pre-commit hook integration
- Watch mode for continuous quality monitoring

**File**: `when-need-fast-validation-use-quick-quality-check.md`

---

## Remaining Skills To Rewrite

### 3. when-validating-code-works-use-functionality-audit

**Original**: `functionality-audit`
**Agents**: 3-4 (tester, codex-auto-agent, qa-engineer, coder)
**Pattern**: Sandbox → Execute → Verify → Debug → Fix

**Key Sections Needed**:
- Intent: Validate code actually works vs. looks correct
- Use Cases: Post-generation verification, debugging, production prep
- Phases: Sandbox creation, test case generation, execution monitoring, output verification, failure analysis
- Agents: Tester (sandbox setup), Codex-auto-agent (execution), QA-engineer (verification), Coder (bug fixes)

---

### 4. when-detecting-fake-code-use-theater-detection

**Original**: `theater-detection-audit`
**Agents**: 2 (code-analyzer, tester)
**Pattern**: Pattern detection → Context analysis → Dependency mapping → Risk assessment

**Key Sections Needed**:
- Intent: Find mocks, TODOs, placeholders, unimplemented functions
- Use Cases: Pre-production deployment, code handoff, quality audits
- Phases: Pattern-based detection, contextual analysis, dependency mapping, risk assessment, completion workflow
- Agents: Code-analyzer (pattern scanning), Tester (execution verification)

---

### 5. when-reviewing-code-comprehensively-use-code-review-assistant

**Original**: `code-review-assistant`
**Agents**: 5+ specialists (security-reviewer, performance-reviewer, style-auditor, test-reviewer, docs-reviewer)
**Pattern**: Multi-agent swarm review with auto-fix suggestions

**Key Sections Needed**:
- Intent: Comprehensive PR review with specialized perspectives
- Use Cases: PR validation, merge readiness assessment, quality enforcement
- Phases: PR analysis, parallel specialized reviews, integration analysis, final report
- Agents: Security-reviewer, Performance-reviewer, Style-auditor, Test-reviewer, Docs-reviewer
- Integration: GitHub API, PR comments, automated checks

---

### 6. when-verifying-quality-use-verification-quality

**Original**: `verification-quality`
**Agents**: 3 (qa-engineer, tester, code-reviewer)
**Pattern**: Truth scoring + verification + automatic rollback

**Key Sections Needed**:
- Intent: Truth scoring system with 0.95 threshold and auto-rollback
- Use Cases: Real-time quality monitoring, CI/CD gates, rollback automation
- Phases: Truth metric calculation, verification checks, rollback on failure, quality tracking
- Agents: QA-engineer (verification), Tester (execution), Code-reviewer (quality assessment)
- Features: Truth scoring dashboard, automatic rollback, CI/CD integration

---

### 7. when-auditing-code-style-use-style-audit

**Original**: `style-audit`
**Agents**: 2 (code-analyzer, style-auditor)
**Pattern**: Lint → Review → Rewrite → Verify

**Key Sections Needed**:
- Intent: Transform functional code into production-grade maintainable code
- Use Cases: Post-functionality validation, pre-code-review, team standards enforcement
- Phases: Automated linting, manual review, security/performance review, documentation review, consistency analysis, code rewriting, verification
- Agents: Code-analyzer (automated checks), Style-auditor (manual review + rewriting)
- Integration: CI/CD linters, pre-commit hooks, formatting tools

---

## Common Patterns Across All Skills

### Trigger-First Naming Convention
- Pattern: `when-{trigger-condition}-{action-verb}-{function-name}`
- Examples: `when-validating-deployment-readiness-use-production-readiness`
- Benefit: Users discover skills by trigger condition, not implementation detail

### SOP Structure (7 Phases from Skill-Forge)
1. **Intent Archaeology**: Trigger condition, core user need, outcome expectation
2. **Use Case Crystallization**: Primary use cases, workflow phases, integration points
3. **Structural Architecture**: Agent orchestration, memory coordination, hook integration
4. **Metadata Engineering**: Input/output contracts, data schemas, configuration
5. **Instruction Crafting**: System prompts, execution scripts, coordination logic
6. **Resource Development**: CI/CD integration, related skills, usage examples
7. **Validation Protocol**: Success criteria, failure modes, performance benchmarks

### Agent Orchestration Patterns
- **Sequential**: Theater → Functionality → Style (audit pipeline)
- **Parallel**: All agents spawn simultaneously (quick check)
- **Hybrid**: Sequential phases with parallel agents within phases (production readiness)

### Memory Coordination
```
skill-name/{session-id}/
├── phase-{N}/
│   ├── {agent-type}/
│   │   ├── status.json
│   │   ├── output.json
│   │   └── metrics.json
│   └── aggregated.json
└── final-report.json
```

### Hook Integration
```bash
# Pre-task
npx claude-flow@alpha hooks pre-task --description "{task}" --session-id "{id}"

# Per-agent coordination
npx claude-flow@alpha memory store --key "{namespace}/{agent}/output" --value "{data}"
npx claude-flow@alpha memory retrieve --pattern "{namespace}/*"

# Post-task
npx claude-flow@alpha hooks post-task --task-id "{id}" --export-metrics
```

---

## Next Steps

1. ✅ Complete rewrites for remaining 5 skills
2. Validate all agent mappings match SKILL-TO-AGENT-ASSIGNMENTS.md
3. Test execution scripts for functionality
4. Add CI/CD integration examples for each skill
5. Create skill catalog with trigger-first index
6. Document agent coordination patterns
7. Add performance benchmarks from testing

---

**Document Status**: In Progress (2/7 complete)
**Methodology**: Skill-forge 7-phase SOP
**Naming**: Trigger-first convention applied
**Agent Mapping**: Based on SKILL-TO-AGENT-ASSIGNMENTS.md
**Integration**: Hooks + Memory + Scripts + CI/CD
