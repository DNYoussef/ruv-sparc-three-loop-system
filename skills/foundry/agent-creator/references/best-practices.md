# Agent Creator Best Practices

## Evidence-Based Prompting Principles

### 1. Self-Consistency Validation

**Principle**: Generate multiple reasoning paths and select the most consistent answer.

**Implementation in Agents**:
```markdown
### Self-Consistency Validation
Before finalizing deliverables, I validate from multiple angles:
1. [Domain-specific validation 1]
2. [Domain-specific validation 2]
3. [Cross-check with standards]
```

**Example (Marketing Specialist)**:
```markdown
### Self-Consistency Validation
Before finalizing campaign strategy, I validate:
1. **Data Validation**: Do ROI calculations match historical benchmarks?
2. **Segment Validation**: Did I analyze performance across all audience segments?
3. **Statistical Validation**: Are A/B test conclusions statistically significant (p<0.05)?
4. **Budget Validation**: Does total allocation equal budget (±5% for testing reserve)?
```

**Benefits**:
- Reduces hallucinations by cross-checking reasoning
- Improves accuracy by 15-30% (proven in research)
- Builds agent confidence calibration

---

### 2. Program-of-Thought (PoT) Decomposition

**Principle**: Decompose complex tasks into intermediate steps BEFORE execution.

**Implementation in Agents**:
```markdown
### Program-of-Thought Decomposition
For complex tasks, I decompose BEFORE execution:
1. [Domain-specific decomposition pattern]
2. [Dependency analysis]
3. [Risk assessment]
```

**Example (DevOps Coordinator)**:
```markdown
### Program-of-Thought Decomposition
For deployment orchestration, I decompose BEFORE spawning agents:
1. **Parse**: Load manifest → Validate schema → Extract services + dependencies
2. **Plan**: Build DAG → Identify execution waves → Estimate duration → Check capacity
3. **Prepare**: Store rollback manifest → Initialize monitoring → Reserve resources
4. **Execute**: Spawn agents wave-by-wave → Monitor health → Handle errors
5. **Verify**: Run health checks → Validate endpoints → Confirm metrics
6. **Finalize**: Store deployment state → Clean up temp resources → Notify stakeholders
```

**Benefits**:
- Prevents premature execution (failing fast)
- Improves planning quality by 20-40%
- Enables better error recovery (clear rollback points)

---

### 3. Plan-and-Solve Execution

**Principle**: Explicit planning phase before execution, with validation at each step.

**Implementation in Agents**:
```markdown
### Plan-and-Solve Execution
My standard workflow:
1. **PLAN**: [Domain-specific planning]
2. **VALIDATE**: [Domain-specific validation]
3. **EXECUTE**: [Domain-specific execution]
4. **VERIFY**: [Domain-specific verification]
5. **DOCUMENT**: [Memory storage patterns]
```

**Example (Full-Stack Developer)**:
```markdown
### Plan-and-Solve Execution
My standard full-stack workflow:
1. **PLAN**: API contract design → Define types → Estimate complexity
2. **VALIDATE**: Types consistent → No circular dependencies → Feasibility confirmed
3. **EXECUTE**: Backend first → Frontend next → Integration tests
4. **VERIFY**: All tests pass → CI/CD green → Performance meets SLA
5. **DOCUMENT**: Store API contract in memory → Tag with metadata → Notify stakeholders
```

**Benefits**:
- Reduces errors by 25-35% (validation catches mistakes early)
- Improves consistency (repeatable workflow)
- Enables better collaboration (clear phases)

---

## System Prompt Optimization Techniques

### 1. Role Clarity with Specific Expertise

**Bad** (vague):
```markdown
I am a developer who can build applications.
```

**Good** (specific):
```markdown
I am a **Full-Stack Developer Agent** with deep expertise across frontend (React/TypeScript), backend (Node.js/Express), database (PostgreSQL/Prisma), and DevOps (Docker/CI-CD). Through systematic domain analysis, I possess precision-level understanding of:
- **Frontend Development** - React 18, hooks, TypeScript generics, TailwindCSS, accessibility
- **Backend Development** - RESTful APIs, JWT authentication, Zod validation, middleware patterns
...
```

**Why Better**:
- Activates specific knowledge domains
- Sets clear scope boundaries
- Enables better task routing (agent knows what it can/can't do)

---

### 2. Command Specifications with WHEN and HOW

**Bad** (incomplete):
```markdown
- /file-read - Read files
```

**Good** (complete):
```markdown
- `/file-read`, `/file-write`, `/glob-search`, `/grep-search`
- **WHEN**: Reading campaign performance CSVs, writing strategy documents, searching for historical data
- **HOW**: `/file-read campaigns/q3-performance.csv` → parse metrics → analyze trends
```

**Why Better**:
- Provides context for command usage (not just syntax)
- Shows exact patterns (HOW examples)
- Reduces command misuse

---

### 3. Guardrails with WRONG vs. CORRECT Examples

**Bad** (only states rule):
```markdown
Never deploy without tests.
```

**Good** (shows contrast):
```markdown
### ❌ NEVER: Deploy without CI/CD pipeline passing
**WHY**: Broken tests = broken production

**WRONG**:
```bash
# Skip tests, deploy directly
git push origin main
```

**CORRECT**:
```bash
# CI/CD pipeline runs tests automatically
git push origin main
# Wait for GitHub Actions: ✅ All tests pass
# THEN deploy
```
```

**Why Better**:
- Concrete examples (not abstract rules)
- Shows consequences (WHY explanations)
- Reduces violations by 40-60%

---

### 4. Workflow Examples with Exact Commands

**Bad** (high-level):
```markdown
1. Analyze data
2. Create strategy
3. Deploy
```

**Good** (exact commands):
```markdown
**Step-by-Step Commands**:
```yaml
Step 1: Gather Historical Data
  COMMANDS:
    - /file-read campaigns/q3-2024-performance.csv
    - /memory-retrieve --key "marketing/audience-personas"
  OUTPUT: Historical ROAS, CAC, conversion rates by segment
  VALIDATION: Sufficient data (3+ months), consistent tracking

Step 2: Audience Segmentation
  COMMANDS:
    - /audience-segment --criteria "repeat-purchaser,high-ltv"
  OUTPUT: 3-5 audience segments with LTV, CAC
  VALIDATION: Segments cover 80%+ of target audience
```
```

**Why Better**:
- Executable (agent can copy-paste commands)
- Testable (can verify workflow)
- Improves consistency by 30-50%

---

## Agent Specialization Patterns

### Pattern 1: Analytical Agents

**Focus**: Evidence evaluation, data quality, validation

**Phase 1 Emphasis**:
- Statistical methods (A/B testing, significance tests)
- Data quality standards (completeness, accuracy)
- Evidence hierarchy (peer-reviewed > blog posts)

**Phase 2 Emphasis**:
- Analytical heuristics ("Always validate sample size before conclusions")
- Validation frameworks (checklist-based)

**Phase 3 Emphasis**:
- Self-consistency checking (multi-angle validation)
- Confidence calibration (quantify uncertainty)

**Phase 4 Emphasis**:
- Statistical validation code (significance tests, power analysis)
- Error detection patterns (outliers, missing data)

**Example Agents**: Data Analyst, Research Analyst, Quality Auditor

---

### Pattern 2: Generative Agents

**Focus**: Content creation, design, synthesis

**Phase 1 Emphasis**:
- Quality criteria (readability, engagement, accuracy)
- Template patterns (proven structures)
- Audience understanding (who, what, why)

**Phase 2 Emphasis**:
- Creative heuristics ("Show, don't tell", "Edit ruthlessly")
- Refinement cycles (draft → review → revise)

**Phase 3 Emphasis**:
- Plan-and-solve frameworks (outline → draft → polish)
- Requirement tracking (checklist of must-haves)

**Phase 4 Emphasis**:
- Generation patterns (templates, examples)
- Quality validation code (readability scores, plagiarism checks)

**Example Agents**: Content Writer, Marketing Copywriter, UI Designer

---

### Pattern 3: Diagnostic Agents

**Focus**: Problem identification, debugging, root cause analysis

**Phase 1 Emphasis**:
- Problem patterns (common failures)
- Debugging workflows (systematic troubleshooting)
- Diagnostic tools (logs, profilers, debuggers)

**Phase 2 Emphasis**:
- Hypothesis generation ("If X is broken, Y symptoms appear")
- Systematic testing (isolate variables)

**Phase 3 Emphasis**:
- Program-of-thought decomposition (break problem into sub-problems)
- Evidence tracking (what's ruled out, what's confirmed)

**Phase 4 Emphasis**:
- Detection scripts (log parsing, error pattern matching)
- Root cause analysis patterns (5 Whys, Fishbone diagrams)

**Example Agents**: Debugging Specialist, Performance Analyzer, Security Auditor

---

### Pattern 4: Orchestration Agents

**Focus**: Workflow coordination, agent management, dependency resolution

**Phase 1 Emphasis**:
- Workflow patterns (sequential, parallel, conditional)
- Dependency management (DAGs, critical paths)
- Error recovery (retry, rollback, escalation)

**Phase 2 Emphasis**:
- Coordination heuristics ("Parallelize independent tasks")
- Error recovery frameworks (classify → retry/rollback)

**Phase 3 Emphasis**:
- Plan-and-solve with dependencies (PERT charts)
- Progress tracking (milestones, health checks)

**Phase 4 Emphasis**:
- Orchestration code (workflow engines, DAG libraries)
- Retry logic (exponential backoff, circuit breakers)
- Escalation paths (when to involve humans)

**Example Agents**: DevOps Coordinator, Project Manager, Workflow Orchestrator

---

## Testing & Validation Best Practices

### 1. Validation Checklist (Complete)

Use this checklist for EVERY agent before production deployment:

- [ ] **Identity**: Agent maintains consistent role across interactions
- [ ] **Commands**: Uses universal commands correctly (file ops, git, memory)
- [ ] **Specialist Skills**: Demonstrates domain expertise (not generic responses)
- [ ] **MCP Integration**: Coordinates via memory, spawns sub-agents when needed
- [ ] **Guardrails**: Prevents identified failure modes (examples work as expected)
- [ ] **Workflows**: Executes workflow examples successfully (copy-paste commands work)
- [ ] **Metrics**: Tracks performance data (task completion, quality, efficiency)
- [ ] **Code Patterns**: Applies exact patterns from Phase 4 (code snippets work)
- [ ] **Error Handling**: Escalates appropriately (knows when stuck)
- [ ] **Consistency**: Produces stable outputs on repeat (not random)

### 2. Test Suite Structure

**Typical Cases** (80% of usage):
- Standard workflows, common inputs
- Expected behavior on normal data

**Edge Cases** (15% of usage):
- Boundary conditions, unusual inputs
- Empty data, very large data, malformed data

**Error Cases** (5% of usage):
- Graceful failure, recovery
- Missing dependencies, invalid configurations

**Integration Cases** (cross-agent):
- End-to-end workflows with other agents
- Memory coordination, event propagation

**Performance Cases**:
- Speed (latency, throughput)
- Efficiency (token usage, API calls)
- Resource usage (memory, CPU)

### 3. Metrics to Track

**Task Completion**:
- Tasks completed per day/week
- Task duration (avg, p50, p95)
- Success rate (completed / attempted)

**Quality**:
- Validation passes (checklist items ✅)
- Escalations (needed help from other agents)
- Error rate (failures / attempts)
- Recommendation adoption (by users/clients)

**Efficiency**:
- Commands per task (avg)
- MCP tool usage (frequency)
- Token usage (per task)
- API calls (per task)

**Business Impact** (if applicable):
- Revenue influenced (for marketing/sales agents)
- Cost savings (for optimization agents)
- Time savings (vs. manual work)
- Customer satisfaction (NPS, CSAT scores)

---

## Continuous Improvement Cycle

### 1. Weekly Review

**What to Review**:
- Performance metrics (task completion, quality, efficiency)
- Failure cases (errors, escalations)
- User feedback (satisfaction scores, complaints)

**Actions**:
- Identify top 3 failure modes
- Update guardrails to prevent
- Add new code patterns (if discovered)

### 2. Monthly Iteration

**What to Update**:
- System prompt refinements (v2.1, v2.2, ...)
- New workflow examples (from successful tasks)
- Updated technology stack (new tools, frameworks)

**Version Control**:
- v1.0: Base prompt from Phase 3
- v1.x: Minor refinements from testing
- v2.0: Enhanced with Phase 4 patterns
- v2.x: Production iterations

### 3. Quarterly Major Updates

**What to Add**:
- New capabilities (expand domain coverage)
- Integration with new MCP servers
- Advanced coordination patterns

**Re-validation**:
- Re-run complete test suite
- Validate all workflow examples still work
- Update documentation

---

## Common Pitfalls & How to Avoid

### Pitfall 1: Shallow Expertise (Jack-of-All-Trades)

**Symptom**: Agent provides generic advice, lacks domain-specific insights

**Cause**: Phase 1 analysis too broad, Phase 2 expertise extraction incomplete

**Fix**:
- Spend more time on Phase 1 domain research (read expert blogs, documentation)
- In Phase 2, identify 3-5 specific expertise domains (not generic "development")
- Add 10+ domain-specific heuristics (not generic "write good code")

### Pitfall 2: Incomplete Guardrails

**Symptom**: Agent makes mistakes despite warnings in prompt

**Cause**: Guardrails stated as rules (abstract) not examples (concrete)

**Fix**:
- Convert every guardrail to WRONG vs. CORRECT example
- Add WHY explanations (consequences of violation)
- Test guardrails explicitly (try to trigger violation)

### Pitfall 3: Missing Code Patterns (Phase 4)

**Symptom**: Agent produces correct logic but inefficient/buggy code

**Cause**: Phase 4 skipped or rushed, no exact code patterns

**Fix**:
- Extract 3-5 code patterns from codebase (with file:line references)
- Include edge case handling in patterns
- Add comments explaining "When I see X, I know Y"

### Pitfall 4: Poor MCP Integration

**Symptom**: Agent doesn't coordinate with other agents, doesn't use memory

**Cause**: MCP integration patterns not specified, namespace conventions unclear

**Fix**:
- Define exact MCP tool usage patterns (with code examples)
- Specify namespace conventions (e.g., `{agent-role}/{task-id}/{data-type}`)
- Add workflow examples showing memory storage/retrieval

### Pitfall 5: No Performance Tracking

**Symptom**: Can't measure agent improvement over time

**Cause**: Metrics not defined in Phase 4

**Fix**:
- Define 5-10 key metrics (task completion, quality, efficiency)
- Specify how to track (memory storage keys)
- Set up weekly review process

---

## Summary

**Key Takeaways**:
1. ✅ Use evidence-based techniques (self-consistency, PoT, plan-and-solve) in all agents
2. ✅ Optimize system prompts with specific expertise, exact commands, concrete examples
3. ✅ Match agent pattern to domain (analytical, generative, diagnostic, orchestration)
4. ✅ Validate thoroughly with complete checklist before production
5. ✅ Track metrics and iterate continuously (weekly reviews, monthly updates)

**Avoid Common Pitfalls**:
- ❌ Shallow expertise (spend time on Phase 1+2 domain analysis)
- ❌ Abstract guardrails (use WRONG vs. CORRECT examples)
- ❌ Missing code patterns (extract exact patterns in Phase 4)
- ❌ Poor MCP integration (specify exact tool usage)
- ❌ No metrics (define and track performance)

**Result**: Production-ready agents with deeply embedded expertise, robust guardrails, and measurable performance.
