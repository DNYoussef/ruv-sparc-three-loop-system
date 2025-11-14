# Implementation Roadmap: sop-dogfooding-quality-detection Fixes

**Purpose**: Step-by-step guide to implement all 28 audit fixes  
**Total Effort**: 5-6 hours  
**Complexity**: Low-Medium  
**Risk Level**: Low

---

## Overview

```
Sprint 1: P1 Critical (2 hours)        → Beta Release
    ↓
Sprint 2: P2 High (2 hours)             → GA Release  
    ↓
Sprint 3: P3 Medium (1 hour)            → v1.0 Stable
```

---

## Sprint 1: P1 Critical (2 hours) - BLOCKING

**Objective**: Make skill executable and deployable

### Task 1.1: Create generate-quality-summary.js (45 min)

**File**: C:\Users\17175\scripts\generate-quality-summary.js  
**Status**: MISSING - Must create  
**Effort**: 45 minutes  
**Reference**: Fix #21 in full audit

```bash
# Location where to create:
C:\Users\17175\scripts\

# Copy content from audit Fix #21 (JavaScript code block)
# Key features:
# - Reads JSON analysis results
# - Generates human-readable summary
# - Categorizes violations by severity
# - Provides recommendations
# - Outputs to .txt file

# Verify:
node C:\Users\17175\scripts\generate-quality-summary.js --output C:\Users\17175\metrics\dogfooding\test-summary.txt
# Should create summary file with no errors
```

**Success Criteria**:
- ✅ File exists and is readable
- ✅ Can be executed: `node generate-quality-summary.js`
- ✅ Accepts `--output` parameter
- ✅ Reads JSON input from memory-mcp analysis
- ✅ Creates formatted summary file

---

### Task 1.2: Create dogfood-memory-retrieval.bat (30 min)

**File**: C:\Users\17175\scripts\dogfood-memory-retrieval.bat  
**Status**: MISSING - Must create  
**Effort**: 30 minutes  
**Reference**: Fix #22 in full audit

```bash
# Location:
C:\Users\17175\scripts\

# Copy content from audit Fix #22 (batch script)
# Key features:
# - Queries Memory-MCP for violations
# - Searches for similar fixes
# - Returns top 5 matches
# - Shows match details (project, type, severity)

# Verify:
dogfood-memory-retrieval.bat "parameter bomb fix"
# Should return search results from Memory-MCP
```

**Success Criteria**:
- ✅ File exists and is readable
- ✅ Can be executed as batch script
- ✅ Accepts search query parameter
- ✅ Connects to Memory-MCP
- ✅ Returns search results

---

### Task 1.3: Update YAML Metadata (20 min)

**File**: C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md  
**Lines**: 1-6 (YAML frontmatter)  
**Status**: INCOMPLETE - Must update  
**Effort**: 20 minutes  
**Reference**: Fix #13 in full audit

**Current YAML** (Lines 1-6):
```yaml
---
name: sop-dogfooding-quality-detection
description: 3-part dogfooding workflow Phase 1 - Run Connascence analysis, store results in Memory-MCP with WHO/WHEN/PROJECT/WHY tagging. 30-60 seconds execution time.
agents: code-analyzer, reviewer
mcp_tools: connascence-analyzer, memory-mcp
scripts: dogfood-quality-check.bat, store-connascence-results.js
---
```

**Replace with** (from Fix #13):
```yaml
---
name: sop-dogfooding-quality-detection
alias: [quality-detection, dogfooding-phase1, connascence-check]
description: Phase 1 of 3-part dogfooding workflow - Automatically detect code quality violations (7 types via Connascence analysis) and store findings in Memory-MCP with WHO/WHEN/PROJECT/WHY tagging. Execution time 30-60 seconds. Triggers sop-dogfooding-pattern-retrieval for Phase 2.
category: code-quality
skill_type: analysis
version: 1.0.0
last_updated: 2025-11-02
authors: [connascence-team, dogfooding-system]

agents: [code-analyzer, reviewer]
required_agent_capabilities: [code-analysis, quality-detection, memory-mcp-write]

mcp_tools:
  - connascence-analyzer:analyze_workspace
  - connascence-analyzer:health_check
  - memory-mcp:memory_store
  - memory-mcp:vector_search

scripts:
  - dogfood-quality-check.bat
  - store-connascence-results.js
  - generate-quality-summary.js
  - dogfood-memory-retrieval.bat

intent: quality-detection
intent_confidence: 0.95
auto_trigger: false
manual_trigger_command: "dogfood-quality-check.bat <project>"
trigger_keywords: 
  - dogfooding
  - code-quality
  - violations
  - connascence
  - quality-detection
  - code-analysis
  - nasa-compliance
  - parameter-bomb
  - god-object

parent_workflow: dogfooding-3-part-system
phase: 1
next_phase_skill: sop-dogfooding-pattern-retrieval
triggered_by: [functionality-audit, code-review-assistant, production-readiness]

execution_time_seconds: [30, 60]
success_rate_target: 0.95
---
```

**Success Criteria**:
- ✅ YAML parses without errors
- ✅ intent field present
- ✅ trigger_keywords array populated
- ✅ required_agent_capabilities defined
- ✅ All scripts listed including new ones

---

### Task 1.4: Add Input Parameters Section (20 min)

**File**: C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md  
**Insert After**: YAML frontmatter (before "System Architecture")  
**Content**: Fix #1 from full audit  
**Effort**: 20 minutes

```markdown
## Input Parameters

**Required**:
- `project_name` (string): Name of target project (e.g., "memory-mcp", "connascence-analyzer")
- `project_path` (optional string): Full path to project. Defaults to `C:\Users\17175\Desktop\{project_name}`

**Optional**:
- `violation_threshold` (integer): Only report violations with severity ≥ threshold. Default: 0 (all)
- `output_format` (enum): "json" | "summary" | "both". Default: "both"

**Examples**:
- Single project: `dogfood-quality-check.bat memory-mcp`
- All projects: `dogfood-quality-check.bat all`
```

**Success Criteria**:
- ✅ Parameters documented
- ✅ Examples provided
- ✅ Defaults specified

---

### Task 1.5: Add Skill-Level Success Criteria (15 min)

**File**: C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md  
**Insert After**: Input Parameters section  
**Content**: Fix #3 from full audit  
**Effort**: 15 minutes

```markdown
## Skill-Level Success Criteria

**PASS Conditions** (ALL must be true):
- ✅ Connascence analysis completes without errors (exit code 0)
- ✅ JSON output file generated with >0 files analyzed
- ✅ Violations categorized by all 7 types
- ✅ Results stored in Memory-MCP (confirmed via collection.count())
- ✅ Summary report generated with recommendations
- ✅ Dashboard database updated with metrics

**FAIL Conditions** (ANY triggers automatic rollback):
- ❌ Connascence analyzer health check fails
- ❌ ChromaDB connection fails
- ❌ JSON parsing fails
- ❌ Memory-MCP storage fails after 3 retries
- ❌ Execution exceeds 120 seconds (2x time budget)

**Measurement**: `success_rate = (passes / total_runs) * 100`. Target: ≥95%.
```

**Success Criteria**:
- ✅ Clear pass conditions
- ✅ Clear fail conditions
- ✅ Measurable metric

---

### Task 1.6: Add Pre-Use Validation Checklist (20 min)

**File**: C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md  
**Insert Before**: "System Architecture" section  
**Content**: Fix #25 from full audit  
**Effort**: 20 minutes

Includes:
- 8 pre-flight checks (Connascence, Memory-MCP, directories, scripts, etc.)
- Troubleshooting section for each failed check
- Commands to verify/fix each issue

**Success Criteria**:
- ✅ Checklist provided (8-10 items)
- ✅ Each item has verification command
- ✅ Troubleshooting steps provided

---

### Sprint 1 Verification

```bash
# After all 6 tasks complete:

# 1. Verify scripts exist
ls C:\Users\17175\scripts\generate-quality-summary.js
ls C:\Users\17175\scripts\dogfood-memory-retrieval.bat

# 2. Verify SKILL.md parses
head -20 C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md | grep -E "^(name|intent|trigger_keywords)"

# 3. Test Phase 1-2 locally
C:\Users\17175\scripts\dogfood-quality-check.bat memory-mcp

# 4. Check for errors
# Expected: "✓ Analysis complete", "✓ Results stored in Memory MCP"
```

**Exit Criteria for Sprint 1**:
- ✅ 2 missing scripts created
- ✅ YAML metadata complete with intent & keywords
- ✅ Input parameters documented
- ✅ Success/failure criteria defined
- ✅ Validation checklist exists
- ✅ Skill can execute Phase 1 & 2 without errors
- ✅ Ready for Beta release

---

## Sprint 2: P2 High (2 hours) - QUALITY

**Objective**: Improve usability and debuggability

### Task 2.1: Add Real-World Examples (45 min)

**File**: C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md  
**Insert After**: Intent Classification  
**Content**: Fix #6 from full audit  
**Effort**: 45 minutes

3 detailed use cases:
1. **Post-development QA** - Automatic detection after functionality-audit
2. **Pre-deployment Quality Gate** - Manual trigger before production
3. **Continuous Dogfooding Integration** - Scheduled CI/CD runs

Each includes:
- Trigger point
- Scenario description
- Step-by-step workflow
- Actual example commands & outputs
- Expected results

**Success Criteria**:
- ✅ 3+ concrete walkthroughs provided
- ✅ Each shows actual commands
- ✅ Each shows expected output
- ✅ "When to use" / "When NOT to use" sections included

---

### Task 2.2: Document Dependencies (30 min)

**File**: C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md  
**Insert After**: "MCP Tools Mapping"  
**Content**: Fix #23 from full audit  
**Effort**: 30 minutes

Includes:
- Python dependencies (requirements.txt)
- Node.js dependencies (package.json)
- System requirements
- Installation instructions
- Verification commands

**Also Create**:
- C:\Users\17175\metrics\dogfooding\create-tables.sql (database schema)

**Success Criteria**:
- ✅ Python dependencies listed
- ✅ Node dependencies listed
- ✅ Installation instructions provided
- ✅ Verification commands work
- ✅ Database schema provided

---

### Task 2.3: Integrate Error Handling into Phases (30 min)

**File**: C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md  
**Restructure**: Current "Error Handling" section (Lines 135-168)  
**Content**: Fix #9 from full audit  
**Effort**: 30 minutes

Move error handling out of separate section into phase workflows:
- "If Phase 1 health check fails" (integrated)
- "If Phase 2 analysis fails" (integrated)
- "If Phase 3 storage fails" (integrated)
- "If Phase 4 report fails" (integrated)
- "If Phase 5 dashboard update fails" (integrated)

Plus "Cross-Phase Failure Recovery" section.

**Success Criteria**:
- ✅ Error handling integrated into phases
- ✅ Each phase has failure path documented
- ✅ Recovery steps provided
- ✅ Memory-MCP error storage documented

---

### Task 2.4: Add Troubleshooting Section (25 min)

**File**: C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md  
**Insert**: After error handling  
**Content**: Fix #25 (Troubleshooting subsection) from full audit  
**Effort**: 25 minutes

Common issues & solutions:
- "Connascence health check failed" - 5 diagnostic steps
- "Memory-MCP initialization failed" - 4 diagnostic steps
- "Output directory not writable" - 3 diagnostic steps
- "SQLite database corrupted" - 3 diagnostic steps

**Success Criteria**:
- ✅ 4+ common issues documented
- ✅ Each has diagnostic commands
- ✅ Each has fix steps
- ✅ Includes commands user can copy-paste

---

### Task 2.5: Add JSON Output Schema (20 min)

**File**: C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md  
**Insert After**: "Output Specification" heading  
**Content**: Fix #26 from full audit  
**Effort**: 20 minutes

Includes:
- JSON Schema (formal specification)
- Example JSON output
- Field descriptions
- Validation instructions

**Success Criteria**:
- ✅ JSON Schema provided (JSON-Schema.org format)
- ✅ Example output shown
- ✅ All fields documented
- ✅ Validation method shown

---

### Sprint 2 Verification

```bash
# After all 5 tasks complete:

# 1. Check all sections added
grep -c "Real-World Examples" SKILL.md
grep -c "Python Dependencies" SKILL.md
grep -c "If Phase 1" SKILL.md
grep -c "Troubleshooting" SKILL.md
grep -c "JSON Schema" SKILL.md

# 2. Run full quality check
C:\Users\17175\scripts\dogfood-quality-check.bat all

# 3. Test error handling
# Manually break something and verify error path works
```

**Exit Criteria for Sprint 2**:
- ✅ Real-world examples provided (3+ with actual output)
- ✅ All dependencies documented (Python, Node, System)
- ✅ Error handling integrated into phase workflows
- ✅ Troubleshooting section with common issues
- ✅ JSON output schema provided
- ✅ Ready for GA release

---

## Sprint 3: P3 Medium (1 hour) - POLISH

**Objective**: Improve maintainability and completeness

### Task 3.1: Add Integration Test Case (30 min)

**File**: C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md  
**Insert**: In "Validation" section  
**Content**: Fix #25 (Integration Testing subsection) from full audit  
**Effort**: 30 minutes

Complete end-to-end test:
- Setup phase (capture baseline)
- Execution phase (run quality check)
- Validation phase (verify all outputs)
- Success criteria (8 checkpoints)
- Failure diagnosis (what to check if something fails)

**Success Criteria**:
- ✅ Full test case documented
- ✅ Setup, exec, validation phases clear
- ✅ 8+ success criteria
- ✅ Can be executed step-by-step

---

### Task 3.2: Add Memory-MCP Verification (15 min)

**File**: C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md  
**Insert**: After Phase 3 storage  
**Content**: Fix #27 from full audit  
**Effort**: 15 minutes

Includes:
- Commands to check collection size
- Query stored data example
- Metadata verification
- What to do if verification fails

---

### Task 3.3: Add Dashboard Verification (15 min)

**File**: C:\Users\17175\skills\sop-dogfooding-quality-detection\SKILL.md  
**Insert**: After Phase 5 dashboard update  
**Content**: Fix #28 from full audit  
**Effort**: 15 minutes

Includes:
- How to check SQLite was updated
- How to verify Grafana dashboard refreshed
- How to check logs for errors
- Fallback if dashboard not available

---

### Sprint 3 Verification

```bash
# After all 3 tasks complete:

# 1. Check sections added
grep -c "Integration Testing" SKILL.md
grep -c "Memory-MCP Verification" SKILL.md
grep -c "Dashboard Verification" SKILL.md

# 2. Run integration test
# Follow steps in Integration Testing section

# 3. Verify all data appears correctly
# Check JSON, Memory-MCP, SQLite, Grafana
```

**Exit Criteria for Sprint 3**:
- ✅ Integration test case documented
- ✅ Can verify Memory-MCP storage
- ✅ Can verify Dashboard updates
- ✅ Skill is "v1.0 Stable"

---

## Optional P4 Polish Tasks (1+ hours)

If time permits, these improve code quality:

### Task 4.1: Standardize Placeholder Format (15 min)
- Add section documenting `<placeholder>` format
- Ensure consistent usage throughout
- Reference: Fix #17

### Task 4.2: Add Phase Transition Guide (20 min)
- Explain Phase 1 → Phase 2 flow
- Show data continuity
- Include example workflow
- Reference: Fix #12

### Task 4.3: Fix API Patterns (45 min)
- Replace `await Task()` with correct API
- Update all code examples
- Reference: Fix #16

### Task 4.4: Create Metadata Table (10 min)
- Summarize key info in table format
- Reference: Fix #14

### Task 4.5: Specify MCP Functions (15 min)
- Detail each MCP tool's interface
- Reference: Fix #15

---

## Full Implementation Checklist

### Sprint 1: P1 Critical (BLOCKING)
- [ ] Task 1.1: Create generate-quality-summary.js (45 min)
- [ ] Task 1.2: Create dogfood-memory-retrieval.bat (30 min)
- [ ] Task 1.3: Update YAML metadata (20 min)
- [ ] Task 1.4: Add input parameters section (20 min)
- [ ] Task 1.5: Add success/failure criteria (15 min)
- [ ] Task 1.6: Add validation checklist (20 min)
- [ ] Sprint 1 verification (15 min)
- **Subtotal: 2.5 hours**

### Sprint 2: P2 High (QUALITY)
- [ ] Task 2.1: Add real-world examples (45 min)
- [ ] Task 2.2: Document dependencies (30 min)
- [ ] Task 2.3: Integrate error handling (30 min)
- [ ] Task 2.4: Add troubleshooting (25 min)
- [ ] Task 2.5: Add JSON schema (20 min)
- [ ] Sprint 2 verification (15 min)
- **Subtotal: 2 hours 45 minutes**

### Sprint 3: P3 Medium (POLISH)
- [ ] Task 3.1: Add integration test (30 min)
- [ ] Task 3.2: Add Memory-MCP verification (15 min)
- [ ] Task 3.3: Add Dashboard verification (15 min)
- [ ] Sprint 3 verification (15 min)
- **Subtotal: 1 hour 15 minutes**

### Optional P4 Polish (1+ hours)
- [ ] Task 4.1: Standardize placeholders (15 min)
- [ ] Task 4.2: Add phase transitions (20 min)
- [ ] Task 4.3: Fix API patterns (45 min)
- [ ] Task 4.4: Create metadata table (10 min)
- [ ] Task 4.5: Specify MCP functions (15 min)

---

## Timeline

### Fast Track (Critical Only)
```
Sprint 1: 2.5 hours
  ↓
Beta Release
  ↓
[Team uses beta, provides feedback]
  ↓
Sprint 2: 2 hours 45 minutes
  ↓
GA Release
```
**Total**: ~5.25 hours over 1-2 weeks

### Standard Path (All Fixes)
```
Sprint 1: 2.5 hours
Sprint 2: 2 hours 45 minutes
Sprint 3: 1 hour 15 minutes
  ↓
v1.0 Stable Release
```
**Total**: ~6.5 hours over 3 weeks

### Agile Path (Incremental)
```
Week 1: Sprint 1 → Beta (2.5 hours)
Week 2: Sprint 2 → GA (2 hours 45 minutes)
Week 3: Sprint 3 → Stable (1 hour 15 minutes)
```
**Total**: ~6.5 hours over 3 weeks

---

## Resources Needed

- **1 Senior Engineer** (5-6 hours)
- **1 QA** (1-2 hours for testing)
- **Text Editor** (VS Code, etc.)
- **Git** (for version control)
- **1 Hour** review/discussion time

---

## Risk Mitigation

### Risk: Tasks take longer than estimated
**Mitigation**: P1 and P2 are critical, P3/P4 can be deferred

### Risk: Dependencies already installed differently
**Mitigation**: Document current setup during Task 2.2, adjust if needed

### Risk: Breaking changes to existing skill
**Mitigation**: Work on new skill version first, tag as 1.0.1

### Risk: Integration test fails
**Mitigation**: Skip P3, fix issues in P2, re-run P3 next week

---

## Success Metrics

**After Sprint 1 (Beta)**:
- Skill executes without errors
- All phases complete successfully
- Phase 4 summary generation works
- Error handling triggers correctly

**After Sprint 2 (GA)**:
- Users can self-service troubleshoot
- Dependencies clearly documented
- Real-world examples guide usage
- 90% fewer support questions

**After Sprint 3 (v1.0 Stable)**:
- Integration test passes consistently
- Score improves from 6.3/10 → 8.5+/10
- Meets skill-forge quality bar
- Ready for broader team rollout

---

## Sign-Off

**Implementation Lead**: _______________  
**Date Started**: _______________  
**Date Completed**: _______________  
**Issues Encountered**: _______________  
**Notes**: _______________

---

**Document Version**: 1.0  
**Created**: 2025-11-02  
**Last Updated**: 2025-11-02  
**References**: audit-phase1-quality-detection.md (full audit with all fixes)
