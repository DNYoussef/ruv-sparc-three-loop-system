# Obsidian Memory + Connascence Analyzer Integration Specification

## Overview

Integrate two production-ready MCP systems into the 12-Factor Agents ecosystem:
1. **Memory MCP Triple System** - 3-layer persistent memory with Obsidian vault synchronization
2. **Connascence Safety Analyzer** - Comprehensive code coupling analysis

**Objective**: Create an intelligent learning loop where code quality analysis results feed into long-term memory, enabling agents to learn from patterns and proactively suggest better implementations.

---

## Requirements

### Functional Requirements

#### 1. Memory System Integration
- [F1.1] Deploy Memory MCP Triple System as MCP server for Claude Code
- [F1.2] Configure 3-layer memory retention (24h short, 7d mid, 30d+ long)
- [F1.3] Enable Obsidian vault bidirectional sync for knowledge graphs
- [F1.4] Implement mode-aware context adaptation (Execution/Planning/Brainstorming)
- [F1.5] Integrate with existing claude-flow SQLite memory (hot working memory)

#### 2. Connascence Analyzer Integration
- [F2.1] Deploy Connascence Safety Analyzer as MCP server
- [F2.2] Configure all 9 connascence types (CoN, CoT, CoM, CoP, CoA, CoE, CoV, CoI, CoId)
- [F2.3] Integrate with theater-detection-audit (sequential pipeline)
- [F2.4] Enable workspace-wide analysis with incremental caching

#### 3. Learning Loop Implementation
- [F3.1] Store connascence analysis results in Memory MCP (short-term → mid-term → long-term)
- [F3.2] Pattern recognition: Extract recurring coupling patterns
- [F3.3] Agent learning: Load historical patterns from Obsidian on startup
- [F3.4] Proactive suggestions: Agents recommend better coupling approaches
- [F3.5] Cross-session continuity: Resume with full historical context

#### 4. Quality Pipeline Integration
- [F4.1] Post-code-generation hook: theater-detection → connascence analysis
- [F4.2] Pre-commit hook: Block commits with critical connascence violations
- [F4.3] CI/CD integration: Loop 3 feedback with connascence trends
- [F4.4] Dashboard: Visual metrics in Obsidian vault

### Non-Functional Requirements

#### Performance
- **P1**: Memory retrieval <200ms (execution mode)
- **P2**: Connascence analysis <5s (cached), <15s (workspace)
- **P3**: Obsidian sync <5min (batched every 1-5 min)
- **P4**: Total overhead <10% on existing workflows

#### Security
- **S1**: MCP server authentication via Claude Desktop config
- **S2**: Secrets redaction in memory storage (existing guardrails)
- **S3**: Vault encryption at rest (Obsidian native)
- **S4**: No sensitive code stored in long-term memory without redaction

#### Scalability
- **SC1**: Support 10,000+ memory entries with <500ms query time
- **SC2**: Support 100+ files analyzed per workspace
- **SC3**: Graceful degradation if Obsidian unavailable (fallback to SQLite)

#### Reliability
- **R1**: 99.9% uptime for MCP servers
- **R2**: Automatic reconnection on MCP server failure
- **R3**: Data consistency between SQLite (hot) and Obsidian (cold)
- **R4**: Zero data loss on system crashes (WAL logging)

---

## Constraints

### Technical Constraints
- **TC1**: Must use existing MCP protocol (claude-desktop config)
- **TC2**: Python 3.12+ required (Memory MCP dependencies)
- **TC3**: Windows compatibility required (user's OS)
- **TC4**: Integration must not break existing claude-flow memory
- **TC5**: Connascence analyzer must support JavaScript/TypeScript (primary codebase)

### Timeline Constraints
- **TL1**: Phase 1 (Memory MCP) deployment: 2 weeks
- **TL2**: Phase 2 (Connascence) deployment: 2 weeks
- **TL3**: Phase 3 (Learning Loop) activation: 3 weeks
- **TL4**: Phase 4 (Optimization) refinement: 3 weeks
- **Total**: 10 weeks to full production deployment

### Resource Constraints
- **RC1**: Single developer (user + Claude Code)
- **RC2**: Existing hardware (no cloud infrastructure budget)
- **RC3**: Must leverage existing codebases (90% Memory MCP, 100% Connascence ready)

---

## Success Criteria

1. ✅ **MCP Server Deployment**: Both servers running and registered in claude-desktop config
2. ✅ **Memory Persistence**: Data survives Claude Code restarts, recoverable from Obsidian
3. ✅ **Connascence Detection**: All 9 types working with 0% false positives
4. ✅ **Learning Loop**: Agents demonstrate learned patterns (3+ examples within 2 weeks)
5. ✅ **Performance Targets**: All NFRs met (P1-P4, R1-R4)
6. ✅ **Quality Improvement**: 40-60% reduction in coupling issues within 6 months
7. ✅ **User Satisfaction**: User confirms system enhances productivity (not hinders)

### Measurable Outcomes

| Metric | Baseline | Target (6 months) | Measurement |
|--------|----------|-------------------|-------------|
| Coupling Violations | Unknown | -40 to -60% | Connascence CI/CD reports |
| Agent Onboarding Time | N/A | -30 to -50% | Time to first productive code |
| Recurring Issues | Unknown | -50 to -70% | Pattern match frequency |
| Context Retention | 0% (no persistence) | 100% | Cross-session task completion |
| Maintainability Index | Unknown | +20 to +30% | Connascence Index score |

---

## Out of Scope

The following are explicitly **NOT** included in this integration:

1. ❌ Real-time collaborative memory (multi-user scenarios)
2. ❌ Advanced visualizations beyond Obsidian's native graph
3. ❌ Natural language memory updates (use structured API only)
4. ❌ Connascence analysis for languages beyond JS/TS (Python support deferred)
5. ❌ Custom Obsidian plugins (use vanilla Obsidian features only)
6. ❌ Cloud hosting of MCP servers (local-only deployment)
7. ❌ Migration of existing claude-flow memory to Obsidian (fresh start)

---

## Architecture Principles

### Layered Integration (No Redundancy)

```
┌─────────────────────────────────────────────┐
│ Claude Code (Execution Layer)              │
├─────────────────────────────────────────────┤
│ claude-flow (Hot Working Memory - SQLite)  │ ← Real-time coordination
├─────────────────────────────────────────────┤
│ Quality Analysis Layer                     │
│  ├─ theater-detection-audit               │ ← "Is it real?"
│  └─ connascence-analyzer                   │ ← "Is it well-coupled?"
├─────────────────────────────────────────────┤
│ Memory Persistence Layer                   │
│  ├─ Memory MCP (Hot: 24h, Mid: 7d)        │ ← Recent context
│  └─ Obsidian Vault (Long: 30d+)           │ ← Knowledge graphs
├─────────────────────────────────────────────┤
│ Learning & Pattern Recognition             │ ← Extract + Apply patterns
└─────────────────────────────────────────────┘
```

### Data Flow

```
Agent writes code
    ↓
Theater detection (is it real?)
    ↓
Connascence analysis (is it well-coupled?)
    ↓
Results → Memory MCP (short-term)
    ↓ (after 24h with score ≥50%)
Results → Memory MCP (mid-term)
    ↓ (after 7d with score ≥10%)
Results → Obsidian (long-term + compression)
    ↓
Pattern recognition extracts learnings
    ↓
Future agents load patterns on startup
    ↓
Proactive suggestions OR correct implementation
```

---

## Dependencies

### External Dependencies
1. **Memory MCP Triple System**: C:\Users\17175\Desktop\memory-mcp-triple-system
   - Status: 90% complete, production-ready
   - Required: Python 3.12+, ChromaDB, FastAPI

2. **Connascence Safety Analyzer**: C:\Users\17175\Desktop\connascence
   - Status: 100% complete, production-ready
   - Required: Python 3.12+, AST parsing

3. **Obsidian**: Desktop app
   - Version: Latest stable
   - Vault location: TBD (will create)

### Internal Dependencies
1. **claude-flow**: Already installed and working
2. **theater-detection-audit**: Existing skill (.claude/skills/)
3. **12-Factor Agents hooks**: Pre-task, post-task, post-edit

---

## Risk Assessment (Initial)

### High-Priority Risks

1. **Performance Degradation** (Severity: High)
   - Risk: Analysis overhead slows development workflow
   - Mitigation: Async analysis, aggressive caching, user-configurable thresholds

2. **Memory Inconsistency** (Severity: High)
   - Risk: SQLite and Obsidian diverge, data loss
   - Mitigation: Conflict resolution (vault wins), WAL logging, regular sync verification

3. **Windows Compatibility** (Severity: Medium)
   - Risk: MCP servers have Unix-specific dependencies (hnswlib-node issue observed)
   - Mitigation: Pre-compiled binaries, Docker fallback (WSL2)

4. **Learning Loop Noise** (Severity: Medium)
   - Risk: False patterns learned, agent suggestions become annoying
   - Mitigation: High confidence threshold (≥80%), user feedback loop, pattern validation

5. **Obsidian Sync Conflicts** (Severity: Low)
   - Risk: User manually edits vault while sync running
   - Mitigation: File locking, conflict detection, manual merge UI

---

## Phased Rollout Plan (High-Level)

### Phase 1: Memory MCP Foundation (Weeks 1-2)
- Deploy Memory MCP server
- Configure claude-desktop MCP integration
- Test 3-layer retention
- Validate Obsidian sync (read-only first)

### Phase 2: Connascence Integration (Weeks 3-4)
- Deploy Connascence MCP server
- Integrate with theater-detection pipeline
- Configure workspace analysis
- Test all 9 connascence types

### Phase 3: Learning Loop Activation (Weeks 5-7)
- Implement results storage in Memory MCP
- Build pattern recognition module
- Enable agent pattern loading
- Test proactive suggestions

### Phase 4: Optimization & Refinement (Weeks 8-10)
- Performance tuning (caching, batching)
- Dashboard creation in Obsidian
- User feedback integration
- Documentation finalization

---

## Acceptance Criteria

Before marking this integration complete, verify:

- [ ] Both MCP servers running and healthy (health checks passing)
- [ ] Memory persistence working (restart test)
- [ ] Connascence analysis functional (test on sample codebase)
- [ ] Learning loop demonstrates 3+ learned patterns
- [ ] All performance targets met (P1-P4)
- [ ] Zero critical bugs in 2-week stabilization period
- [ ] User confirms productivity enhancement
- [ ] Documentation complete (setup, usage, troubleshooting)

---

**Version**: 1.0.0
**Created**: 2025-11-01
**Author**: DNYoussef + Claude Code
**Status**: Specification Phase Complete
