# CLAUDE.md v2.2.0 Changelog

**Release Date**: 2025-11-15
**Previous Version**: v2.1.0 (2025-11-15)
**Type**: Minor Enhancement (Best of Both Worlds)

---

## TL;DR

**Added Phase 4: Playbook/Skill Routing** between planning and execution.

**New Workflow**: intent → prompt → plan → **route** → execute (5 phases)
**Old Workflow**: intent → prompt → plan → execute (4 phases)

---

## What Changed

### 5-Phase Workflow (Was 4-Phase)

**v2.1.0** (4 phases):
```
1. Intent Analysis
2. Prompt Optimization
3. Strategic Planning
4. Execution
```

**v2.2.0** (5 phases):
```
1. Intent Analysis
2. Prompt Optimization
3. Strategic Planning
4. Playbook/Skill Routing ← NEW!
5. Execution
```

---

## New Feature: Phase 4 Routing

### What It Does

**After** creating the execution plan (Phase 3), **before** executing (Phase 5):
- Takes each task from the plan
- Selects the optimal playbook for that task
- Chooses the best skills/agents for execution
- Considers task complexity, domain, time constraints, MCP availability
- Outputs routing decisions for each task

### Why This Matters

**Separation of Concerns**:
- **Phase 3 (Planning)**: Answers "WHAT to do" and "WHEN" (strategy)
- **Phase 4 (Routing)**: Answers "HOW to do it" and "WITH WHAT" (tactics)
- **Phase 5 (Execution)**: Actually does the work

**Best of Both Worlds**:
- ✅ **v2.0 intelligence**: Smart playbook selection from Section 3
- ✅ **v2.1 intelligence**: Dependency detection + parallelization
- ✅ **v2.2 combined**: Plan the strategy, THEN select the tools

---

## Routing Criteria

Phase 4 uses these criteria to select playbooks:

| Task Type | Route To Playbook | When |
|-----------|-------------------|------|
| Simple feature | simple-feature-implementation | <4 hours, single component |
| Complex feature | three-loop-system (FLAGSHIP) | >4 hours, multi-component |
| Quick research | research-quick-investigation | <2 hours, specific question |
| Deep research | deep-research-sop (FLAGSHIP) | Multi-month, academic |
| Code quality | comprehensive-review | Audit, security, clarity |
| Bug fix | smart-bug-fix | Production issue |
| ML pipeline | ml-pipeline-development | Neural training |
| API development | backend-api-development | REST/GraphQL |
| Frontend | frontend-development | React/Vue/UI |
| Full-stack | feature-dev-complete | End-to-end workflow |

---

## Example: New Phase 4 in Action

**User Request**: "Build a REST API for user management"

### Phase 3 Output (Planning):
```
PHASE 1: Research best practices
PHASE 2 (Parallel): Backend API + Database + Auth
PHASE 3: Integration tests
PHASE 4: Documentation
```

### Phase 4 Output (Routing) ← NEW!
```
PHASE 1:
  Route: Skill("gemini-search") via research-quick-investigation
  Rationale: Quick research, Gemini optimal for best practices

PHASE 2 (Parallel):
  Task 1: Backend API
    Route: backend-api-development playbook
    Skills: backend-dev
    Rationale: Backend specialist playbook

  Task 2: Database
    Route: database-design playbook
    Skills: sql-database-specialist
    Rationale: Database specialist

  Task 3: Auth
    Route: simple-feature-implementation playbook
    Skills: sparc-methodology
    Rationale: Single feature with TDD

PHASE 3:
  Route: testing-quality playbook
  Skills: tester
  Rationale: Comprehensive testing

PHASE 4:
  Route: api-documentation-specialist playbook
  Skills: api-docs
  Rationale: OpenAPI/Swagger specialist
```

### Phase 5 Execution:
Uses the routed playbooks/skills from Phase 4.

---

## Benefits

### 1. **Intelligent Tool Selection**
- Each task gets the OPTIMAL playbook, not generic execution
- Considers task complexity, domain expertise needed, time constraints

### 2. **Flexibility**
- Can route different tasks to different playbooks
- Example: Research task → quick-investigation, Implementation → three-loop-system

### 3. **Transparency**
- User sees both WHAT will be done (Phase 3) AND HOW it will be done (Phase 4)
- Can approve or modify routing decisions before execution

### 4. **Best of v2.0 + v2.1**
- v2.0: Smart playbook routing (Section 3 keyword matching)
- v2.1: Strategic planning (dependencies, parallelization)
- v2.2: **Both!** Plan first, route second, execute third

---

## Migration Guide

### For Users

**No action required** - Automatic upgrade.

**What You'll Notice**:
- One extra phase (routing) between planning and execution
- More detailed output showing which playbooks/skills will be used
- Better tool selection (optimal playbook per task)

**Example Before (v2.1)**:
```
Plan:
- Phase 1: Research
- Phase 2: Implementation (parallel)
- Phase 3: Testing

[Immediately starts executing]
```

**Example After (v2.2)**:
```
Plan:
- Phase 1: Research
- Phase 2: Implementation (parallel)
- Phase 3: Testing

Routing:
- Phase 1 → research-quick-investigation playbook
- Phase 2 Task 1 → backend-api-development playbook
- Phase 2 Task 2 → database-design playbook
- Phase 3 → testing-quality playbook

[User approves, then execution starts]
```

### For Developers

**New Phase Required**:
After Phase 3 (planning), must add Phase 4 (routing) that:
- Takes plan tasks as input
- Matches each task to optimal playbook (using Section 3 criteria)
- Selects skills/agents for each task
- Outputs routing decisions

**No Breaking Changes**:
- Phases 1-3 unchanged
- Phase 5 execution unchanged (just uses routed playbooks now)
- Escape hatches still work (skip to Phase 5)

---

## Comparison: v2.0 vs v2.1 vs v2.2

| Version | Phases | Planning | Routing | Parallelization | Best For |
|---------|--------|----------|---------|-----------------|----------|
| v2.0 | 3 | ❌ None | ✅ Yes | ❌ Manual | Simple tasks |
| v2.1 | 4 | ✅ Yes | ❌ Generic | ✅ Automatic | Complex tasks |
| v2.2 | 5 | ✅ Yes | ✅ Smart | ✅ Automatic | **All tasks** |

**v2.2 = Best of both worlds!**

---

## Performance Impact

**Added Latency**: ~500-1000 tokens for routing phase
**Offset By**: Better tool selection = faster execution
**Net Result**: Neutral to positive (better tools = faster work)

---

## Backward Compatibility

**100% Backward Compatible**:
- v2.0 style: Routing still works (Phase 4 routes to playbooks)
- v2.1 style: Planning still works (Phase 3 plans, Phase 4 routes)
- Escape hatches: Still skip directly to execution

---

## Known Issues

**None** - Tested and validated

---

## Future Enhancements

**Planned for v2.3**:
- Dynamic re-routing (change playbook mid-execution if needed)
- Routing templates (common task patterns pre-defined)
- Cost-based routing (choose fastest/cheapest playbook option)

---

## Version Timeline

| Version | Date | Key Feature | Phases |
|---------|------|-------------|--------|
| v1.0 | 2025-11-13 | Monolithic CLAUDE.md | N/A |
| v2.0 | 2025-11-14 | Playbook-first workflow | 3 |
| v2.1 | 2025-11-15 | Strategic planning | 4 |
| **v2.2** | **2025-11-15** | **Intelligent routing** | **5** |

---

## Summary

**What**: Added Phase 4 (Playbook/Skill Routing) between planning and execution
**Why**: Separate strategy (planning) from tactics (tool selection)
**Benefit**: Best of v2.0 routing + v2.1 planning = optimal tool selection per task
**Impact**: Better, smarter execution with minimal overhead

**Workflow**: intent → prompt → plan → **route** → execute

---

**Status**: ✅ Production Ready
**Breaking Changes**: None
**Adoption**: Automatic