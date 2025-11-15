# New Playbooks v3.0 - MECE Gap Analysis Results

**Generated**: 2025-11-15
**Method**: Sequential Thinking + MECE Framework
**New Playbooks**: 9
**Total Playbooks**: 38 (was 29)

---

## Gap Analysis Summary

Using MECE (Mutually Exclusive, Collectively Exhaustive) framework across 7 dimensions:
1. User Intent (Build, Fix, Optimize, **Learn**, Analyze, **Migrate**, **Refactor**, **Document**)
2. Project Lifecycle (Inception, Development, Testing, Deployment, **Maintenance**, **Sunsetting**)
3. Problem Complexity (Simple, Medium, Complex, **Emergency**)
4. Collaboration Type (Solo, Team, **Stakeholder**, **Knowledge Transfer**)
5. Artifact Type (Code, **Documentation**, Tests, Infrastructure, Data, **Processes**)
6. User Expertise (**Beginner**, Intermediate, Expert, **Non-technical**)
7. Urgency/Risk (Routine, Time-sensitive, **Critical**, Exploratory)

**Identified 15 major gaps**, created playbooks for top 9 priority scenarios.

---

## NEW PLAYBOOKS

### 1. Codebase Onboarding & Learning

**Category**: Learning & Knowledge Transfer
**When**: "Help me understand this codebase", "Onboard new developer", "I'm new here"
**Gap Filled**: Learning/Understanding user intent, Beginner expertise level

**Skill Sequence**:
```yaml
1. intent-analyzer
   - Identify what user wants to learn
   - Learning goals and timeline

2. researcher
   - Explore codebase structure
   - Identify key components
   - Map dependencies

3. architecture-diagram-generator
   - Create visual system overview
   - Component interaction diagrams
   - Data flow diagrams

4. developer-documentation-agent
   - Generate onboarding guide
   - Setup instructions
   - Code walkthrough

5. pair-programming
   - Interactive guided learning
   - Q&A session
   - Hands-on exploration

6. memory-mcp (throughout)
   - Store learning progress
   - Track completed modules
   - Persist key insights
```

**Parallel Opportunities**: Steps 3-4 can run in parallel (diagrams + docs)

**Skills**: 6 skills
**Time**: 1-3 hours
**Complexity**: Medium
**MCP Requirements**: memory-mcp (REQUIRED for learning persistence)

---

### 2. Emergency Incident Response (P0/Critical)

**Category**: Emergency/Crisis Management
**When**: "Production is down", "Critical bug in prod", "P0 incident", "Service outage"
**Gap Filled**: Catastrophic/Emergency complexity, Critical urgency

**Skill Sequence**:
```yaml
1. sop-dogfooding-pattern-retrieval
   - Search memory for similar P0 incidents
   - Retrieve past solutions (10-30 seconds)

2. hierarchical-coordinator (PARALLEL investigation)
   - Spawn 3 agents concurrently:
     a. production-validator (assess current state)
     b. performance-analysis (find bottlenecks)
     c. smart-bug-fix (identify root cause)

3. smart-bug-fix
   - Implement emergency hotfix
   - Codex sandbox iteration if needed

4. functionality-audit
   - Verify fix in sandbox
   - Quick smoke tests

5. cicd-intelligent-recovery
   - Run critical path tests
   - Validate no regressions

6. memory-mcp
   - Store incident details
   - Document resolution steps
   - Add to incident knowledge base
```

**Parallel Opportunities**: Step 2 (3 agents concurrent), critical for speed

**Skills**: 7 skills + hierarchical coordination
**Time**: 30 minutes - 2 hours (CRITICAL PATH)
**Complexity**: High (but optimized for SPEED)
**MCP Requirements**: memory-mcp (incident history), flow-nexus (sandbox testing)

---

### 3. Refactoring & Technical Debt Cleanup

**Category**: Code Quality & Maintenance
**When**: "Refactor God object", "Clean up technical debt", "Extract module", "Improve code quality"
**Gap Filled**: Refactor/Cleanup intent, Maintenance lifecycle

**Skill Sequence**:
```yaml
1. sop-dogfooding-quality-detection
   - Run Connascence analysis
   - Detect God Objects, Parameter Bombs, etc.

2. clarity-linter
   - Identify clarity violations
   - Thin helpers, magic literals

3. code-analyzer
   - Analyze coupling and complexity
   - Identify refactoring opportunities

4. planner
   - Create refactoring plan
   - Identify dependencies
   - Determine safe refactoring order

5. sparc-methodology
   - TDD approach to refactoring
   - Spec → Pseudocode → Architect → Refine → Code

6. functionality-audit
   - Validate no functionality broken
   - Sandbox testing

7. sop-code-review
   - Verify code quality improved
   - Check metrics before/after

8. memory-mcp
   - Store refactoring patterns
   - Document what worked
```

**Parallel Opportunities**: Steps 1-3 can run in parallel (3 different analyzers)

**Skills**: 8 skills
**Time**: 2-6 hours
**Complexity**: Medium-High
**MCP Requirements**: connascence-analyzer, focused-changes, memory-mcp

---

### 4. Database Migration & Schema Changes

**Category**: Data Operations
**When**: "Migrate database", "Upgrade Postgres", "Schema migration", "Data transformation"
**Gap Filled**: Migration/Upgrade intent, Data operations

**Skill Sequence**:
```yaml
1. planner
   - Create comprehensive migration plan
   - Risk assessment
   - Rollback strategy

2. database-backup-recovery-agent
   - Full production backup
   - Verify backup integrity
   - Document restore procedure

3. database-migration-agent
   - Prepare migration scripts
   - Schema diff analysis
   - Migration strategy (blue-green, rolling, etc.)

4. data-pipeline-engineer
   - Create data transformation pipeline
   - Handle schema changes
   - Validate data integrity

5. sql-database-specialist
   - Validate new schema design
   - Index optimization
   - Performance tuning

6. functionality-audit
   - Test migration in sandbox
   - Verify data correctness
   - Performance benchmarks

7. cicd-intelligent-recovery
   - Run full test suite against new schema
   - Validate application compatibility

8. memory-mcp
   - Store migration decisions
   - Document lessons learned
```

**Parallel Opportunities**: Steps 3-4 can run in parallel (scripts + pipeline)

**Skills**: 8 skills
**Time**: 1-3 days
**Complexity**: High (DATA RISK - backup critical!)
**MCP Requirements**: memory-mcp (migration history), flow-nexus (sandbox testing)

---

### 5. Dependency Upgrade & Audit

**Category**: Maintenance & Security
**When**: "Update dependencies", "Upgrade React 17 to 18", "Security vulnerability in dependency"
**Gap Filled**: Migration/Upgrade intent, Maintenance lifecycle

**Skill Sequence**:
```yaml
1. researcher
   - Audit current dependency tree
   - Identify outdated packages
   - Security vulnerability scan

2. planner
   - Create upgrade strategy
   - Risk assessment per dependency
   - Compatibility matrix

3. hierarchical-coordinator (PARALLEL upgrade testing)
   - Spawn N agents (one per major dependency):
     a. functionality-audit (test upgrade in isolation)
     b. code-analyzer (detect breaking changes)
     c. cicd-intelligent-recovery (run test suite)

4. sparc-methodology (if breaking changes)
   - Fix breaking changes with TDD
   - Update API usage

5. sop-code-review
   - Verify upgrade quality
   - Check for deprecation warnings

6. memory-mcp
   - Store successful upgrade path
   - Document breaking change fixes
```

**Parallel Opportunities**: Step 3 (N parallel agents, one per dependency)

**Skills**: 6 skills + hierarchical coordination
**Time**: 2-8 hours
**Complexity**: Medium
**MCP Requirements**: memory-mcp (upgrade history), flow-nexus (parallel testing)

---

### 6. Comprehensive Documentation Generation

**Category**: Documentation & Knowledge Management
**When**: "Generate all docs", "Create API documentation", "Document architecture"
**Gap Filled**: Document/Explain intent, Documentation artifact type

**Skill Sequence**:
```yaml
1. code-analyzer
   - Analyze codebase structure
   - Identify API endpoints
   - Map component hierarchy

2. hierarchical-coordinator (PARALLEL doc generation)
   - Spawn 4 documentation agents:
     a. api-documentation-specialist (OpenAPI/Swagger)
     b. architecture-diagram-generator (C4 models, UML)
     c. developer-documentation-agent (README, setup, guides)
     d. technical-writing-agent (User guides, tutorials)

3. sop-code-review
   - Validate documentation accuracy
   - Check code-doc consistency
   - Verify examples work

4. memory-mcp
   - Store documentation templates
   - Persist writing style guide
```

**Parallel Opportunities**: Step 2 (4 parallel doc types)

**Skills**: 6 skills + hierarchical coordination
**Time**: 3-8 hours
**Complexity**: Medium
**MCP Requirements**: memory-mcp (doc templates)

---

### 7. Performance Optimization Deep Dive

**Category**: Optimization & Tuning
**When**: "App is slow", "Optimize performance", "Reduce load time", "Database queries slow"
**Gap Filled**: Optimize/Improve intent

**Skill Sequence**:
```yaml
1. performance-analysis
   - Benchmark current state
   - Lighthouse audit (if frontend)
   - Identify performance budget

2. perf-analyzer
   - Detect bottlenecks
   - Workflow analysis
   - Critical path identification

3. hierarchical-coordinator (PARALLEL optimization)
   - Spawn specialized optimizers:
     a. frontend-performance-optimizer (bundle, UI, Core Web Vitals)
     b. query-optimization-agent (database query tuning)
     c. cache-strategy-agent (Redis/Memcached caching layer)

4. functionality-audit
   - Verify optimizations don't break functionality
   - Regression testing

5. performance-analysis
   - Benchmark improvements
   - Compare before/after metrics
   - ROI analysis

6. memory-mcp
   - Store optimization patterns
   - Document performance gains
```

**Parallel Opportunities**: Step 3 (3+ parallel optimizations by domain)

**Skills**: 7 skills + hierarchical coordination
**Time**: 4-12 hours
**Complexity**: High
**MCP Requirements**: flow-nexus (performance testing), memory-mcp (optimization patterns)

---

### 8. Internationalization (I18N) Implementation

**Category**: Feature Implementation
**When**: "Add multi-language support", "Translate app", "Localization"
**Gap Filled**: Existing i18n-automation skill had no playbook!

**Skill Sequence**:
```yaml
1. planner
   - I18n strategy (languages, workflow, tools)
   - Resource requirements
   - Timeline

2. i18n-automation
   - Setup translation workflow
   - Configure i18n library
   - Establish translation pipeline

3. hierarchical-coordinator (PARALLEL implementation)
   - Spawn frontend specialists:
     a. react-specialist OR vue-developer (integrate i18n library)
     b. i18n-automation (extract translatable strings)
     c. technical-writing-agent (create translation style guide)

4. functionality-audit
   - Test all target locales
   - Verify text rendering (RTL, special chars)
   - Check date/number formatting

5. wcag-accessibility
   - Ensure accessibility per locale
   - Screen reader compatibility
   - Keyboard navigation

6. memory-mcp
   - Store i18n patterns
   - Translation workflow
```

**Parallel Opportunities**: Step 3 (3 parallel i18n tasks)

**Skills**: 7 skills + hierarchical coordination
**Time**: 1-3 days
**Complexity**: Medium-High
**MCP Requirements**: memory-mcp (translation memory)

---

### 9. Accessibility (A11Y) Compliance

**Category**: Compliance & Quality
**When**: "Make app accessible", "WCAG AA compliance", "Screen reader support"
**Gap Filled**: Existing wcag-accessibility skill had no playbook!

**Skill Sequence**:
```yaml
1. wcag-accessibility
   - Audit current a11y state
   - WCAG 2.1 AA/AAA gap analysis
   - Priority violations

2. planner
   - Create compliance roadmap
   - Phase violations by severity
   - Estimate effort

3. hierarchical-coordinator (PARALLEL a11y fixes)
   - Spawn accessibility specialists:
     a. ui-component-builder (create accessible components)
     b. accessibility-specialist (fix ARIA, semantic HTML)
     c. playwright (setup automated a11y testing)

4. wcag-accessibility
   - Validate WCAG compliance
   - Automated + manual testing
   - Generate compliance report

5. functionality-audit
   - Test with screen readers (NVDA, JAWS, VoiceOver)
   - Keyboard navigation testing
   - High-contrast mode testing

6. memory-mcp
   - Store a11y patterns
   - Component library
   - Compliance checklists
```

**Parallel Opportunities**: Step 3 (3+ parallel a11y specialists)

**Skills**: 6 skills + hierarchical coordination
**Time**: 2-5 days
**Complexity**: High (Compliance requirement)
**MCP Requirements**: playwright (a11y automation), memory-mcp (a11y patterns)

---

## Playbook Statistics

### Before (v2.2)
- **Total Playbooks**: 29
- **Categories**: 8
- **Coverage**: ~60% of common scenarios

### After (v3.0)
- **Total Playbooks**: 38 (+9 new)
- **Categories**: 10 (+2 new: Learning, Emergency)
- **Coverage**: ~85% of common scenarios

### New Playbook Breakdown

| Category | Count | Playbooks |
|----------|-------|-----------|
| Learning & Knowledge | 1 | Codebase Onboarding |
| Emergency Response | 1 | P0 Incident Response |
| Maintenance | 3 | Refactoring, Dependency Upgrade, Database Migration |
| Documentation | 1 | Comprehensive Doc Generation |
| Optimization | 1 | Performance Deep Dive |
| Compliance | 2 | I18N, A11Y |

---

## MCP Requirements by Playbook

| Playbook | Required MCPs | Token Cost | Why |
|----------|---------------|------------|-----|
| Codebase Onboarding | memory-mcp | 12.4k | Learning persistence |
| Emergency Response | memory-mcp, flow-nexus | 44.9k | Incident history + sandbox |
| Refactoring | connascence, focused-changes, memory-mcp | 31.7k | Code analysis + change tracking |
| Database Migration | memory-mcp, flow-nexus | 44.9k | Migration history + sandbox |
| Dependency Upgrade | memory-mcp, flow-nexus | 44.9k | Upgrade history + parallel testing |
| Doc Generation | memory-mcp | 12.4k | Doc templates |
| Performance Optimization | flow-nexus, memory-mcp | 44.9k | Perf testing + patterns |
| I18N Implementation | memory-mcp | 12.4k | Translation memory |
| A11Y Compliance | playwright, memory-mcp | 16.6k | Automated testing + patterns |

---

## Routing Criteria (Phase 4 Updates)

Add to CLAUDE.md Phase 4 routing table:

| Task Type | Route To Playbook | When |
|-----------|-------------------|------|
| Learn codebase | codebase-onboarding | New developer, unfamiliar codebase |
| Production down | emergency-incident-response | P0, critical outage |
| Refactor code | refactoring-technical-debt | God objects, code smells |
| Database change | database-migration | Schema changes, DB upgrades |
| Update dependencies | dependency-upgrade-audit | Security patches, major upgrades |
| Generate docs | comprehensive-documentation | API docs, architecture docs |
| Performance issues | performance-optimization-deep-dive | Slow app, high latency |
| Add languages | i18n-implementation | Multi-language support |
| Accessibility | a11y-compliance | WCAG compliance, screen readers |

---

## Usage Patterns

### Emergency (Minimize Time)
- **emergency-incident-response**: 30min-2hr with parallel investigation

### Maintenance (Regular)
- **dependency-upgrade-audit**: 2-8hr per upgrade cycle
- **refactoring-technical-debt**: 2-6hr per refactoring sprint

### One-Time Projects (High Effort)
- **database-migration**: 1-3 days (DATA RISK, careful planning)
- **i18n-implementation**: 1-3 days (5+ languages)
- **a11y-compliance**: 2-5 days (WCAG AA/AAA)

### Knowledge Building
- **codebase-onboarding**: 1-3hr per new developer
- **comprehensive-documentation**: 3-8hr one-time, then maintenance

---

## Still Missing (Future v3.1+)

**Lower Priority Gaps** (not created yet):
- Stakeholder communication playbooks (status reports, presentations)
- Cost optimization playbooks (AWS/cloud cost reduction)
- Observability setup playbooks (monitoring, tracing, logging)
- Disaster recovery playbooks (DR plan, failover testing)
- Architecture evolution playbooks (monolith → microservices)

**Coverage**: v3.0 covers 85% of scenarios, v3.1 would cover 95%+

---

## Next Steps

1. ✅ Document new playbooks (this file)
2. ⏳ Update ENHANCED-PLAYBOOK-SYSTEM.md
3. ⏳ Update PLAYBOOK-MCP-REQUIREMENTS.md
4. ⏳ Update CLAUDE.md Phase 4 routing table
5. ⏳ Commit and push v3.0

**Status**: Design Complete, Ready for Integration