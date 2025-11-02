# Example 3: Conditional Cascade - Adaptive Bug Fix Workflow

## Overview

This example demonstrates a **conditional branching cascade** where execution paths diverge based on runtime conditions. The workflow adapts its strategy based on bug complexity, automatically choosing between quick fixes, comprehensive analysis, or multi-model collaboration.

## Use Case

Analyze a bug report, assess complexity, and route to the appropriate fix strategy:
- **Simple bugs** → Codex quick fix
- **Medium bugs** → Claude analysis + implementation
- **Complex bugs** → Multi-model collaboration (Gemini + Codex + Claude)

## Cascade Definition

```yaml
cascade:
  name: adaptive-bug-fix-workflow
  description: Intelligently route bug fixes based on complexity analysis
  version: 1.0.0

  inputs:
    - name: issue_id
      type: string
      description: GitHub issue ID or bug tracker reference
      required: true

    - name: repository
      type: string
      description: Repository containing the bug
      required: true

    - name: auto_commit
      type: boolean
      description: Automatically commit fixes if tests pass
      default: false

  stages:
    - stage_id: analyze-bug
      name: Initial Bug Analysis
      model: claude
      skill: analyze-bug-report
      inputs:
        issue_id: ${inputs.issue_id}
        repository: ${inputs.repository}
      outputs:
        - bug_description
        - affected_files
        - stack_trace
        - complexity_score  # 0-100
        - estimated_effort  # simple, medium, complex
        - requires_context  # boolean

    - stage_id: root-cause-analysis
      name: Root Cause Analysis
      model: auto-select
      skill: root-cause-analyzer
      inputs:
        bug: ${analyze-bug.bug_description}
        files: ${analyze-bug.affected_files}
        trace: ${analyze-bug.stack_trace}
      outputs:
        - root_cause
        - related_code
        - test_cases
        - fix_strategy

    - stage_id: complexity-routing
      name: Route Based on Complexity
      type: conditional
      condition: ${analyze-bug.complexity_score}
      branches:
        simple:  # complexity_score < 30
          conditions: ${analyze-bug.complexity_score} < 30
          stages:
            - stage: codex-quick-fix
              name: Codex Quick Fix
              model: codex-auto
              skill: generate-fix
              inputs:
                bug: ${root-cause-analysis.root_cause}
                files: ${analyze-bug.affected_files}
                mode: "fast"
              outputs:
                - fix_code
                - modified_files

            - stage: test-fix
              name: Test Quick Fix
              type: codex-sandbox
              skill: run-tests
              inputs:
                changes: ${codex-quick-fix.modified_files}
              codex_config:
                mode: full-auto
                sandbox: true
                max_iterations: 3
              outputs:
                - test_results
                - all_passed

        medium:  # 30 <= complexity_score < 70
          conditions: ${analyze-bug.complexity_score} >= 30 AND ${analyze-bug.complexity_score} < 70
          stages:
            - stage: claude-analysis
              name: Comprehensive Analysis
              model: claude
              skill: deep-code-analysis
              inputs:
                bug: ${root-cause-analysis.root_cause}
                files: ${root-cause-analysis.related_code}
                context: ${analyze-bug.requires_context}
              outputs:
                - analysis_report
                - fix_plan

            - stage: implement-fix
              name: Implement Fix
              model: claude
              skill: implement-solution
              inputs:
                plan: ${claude-analysis.fix_plan}
                tests: ${root-cause-analysis.test_cases}
              outputs:
                - fix_code
                - modified_files

            - stage: test-fix
              name: Test Comprehensive Fix
              type: codex-sandbox
              skill: run-tests
              inputs:
                changes: ${implement-fix.modified_files}
              codex_config:
                mode: full-auto
                sandbox: true
                max_iterations: 5
              outputs:
                - test_results
                - all_passed

        complex:  # complexity_score >= 70
          conditions: ${analyze-bug.complexity_score} >= 70
          stages:
            - stage: gemini-megacontext
              name: Large Context Analysis
              model: gemini-megacontext
              skill: analyze-full-codebase
              inputs:
                repository: ${inputs.repository}
                bug: ${root-cause-analysis.root_cause}
                context_size: "1M"
              outputs:
                - architecture_insights
                - dependency_map
                - impact_analysis

            - stage: codex-reasoning
              name: Explore Fix Approaches
              model: codex-reasoning
              skill: explore-solutions
              inputs:
                bug: ${root-cause-analysis.root_cause}
                insights: ${gemini-megacontext.architecture_insights}
                constraints: ["backward_compatible", "performance"]
              outputs:
                - approach_candidates
                - tradeoffs

            - stage: claude-implementation
              name: Implement Best Approach
              model: claude
              skill: implement-solution
              inputs:
                approach: ${codex-reasoning.approach_candidates[0]}
                architecture: ${gemini-megacontext.architecture_insights}
                tests: ${root-cause-analysis.test_cases}
              outputs:
                - fix_code
                - modified_files
                - migration_guide

            - stage: test-fix
              name: Comprehensive Testing
              type: codex-sandbox
              skill: run-full-test-suite
              inputs:
                changes: ${claude-implementation.modified_files}
              codex_config:
                mode: full-auto
                sandbox: true
                max_iterations: 10
              outputs:
                - test_results
                - regression_results
                - all_passed

    - stage_id: validate-fix
      name: Final Validation
      model: claude
      skill: validate-solution
      inputs:
        bug: ${analyze-bug.bug_description}
        fix: ${complexity-routing.*.fix_code}
        tests: ${complexity-routing.*.test_results}
      outputs:
        - validation_report
        - is_valid
        - confidence_score

    - stage_id: commit-decision
      name: Decide on Commit
      type: conditional
      condition: ${validate-fix.is_valid} AND ${complexity-routing.*.all_passed}
      branches:
        commit:
          conditions: ${validate-fix.is_valid} == true AND ${inputs.auto_commit} == true
          stages:
            - stage: create-commit
              skill: git-commit
              inputs:
                files: ${complexity-routing.*.modified_files}
                message: "fix: ${analyze-bug.bug_description}"

            - stage: create-pr
              skill: github-create-pr
              inputs:
                title: "Fix: ${analyze-bug.bug_description}"
                body: ${validate-fix.validation_report}
                labels: ["bug-fix", "automated"]

        review-required:
          conditions: ${validate-fix.is_valid} == true AND ${inputs.auto_commit} == false
          stages:
            - stage: create-branch
              skill: git-create-branch
              inputs:
                name: "fix/${inputs.issue_id}"

            - stage: update-issue
              skill: github-update-issue
              inputs:
                issue_id: ${inputs.issue_id}
                comment: |
                  ## Fix Analysis Complete

                  **Complexity**: ${analyze-bug.estimated_effort}
                  **Strategy Used**: ${complexity-routing.branch_taken}
                  **Tests Passed**: ${complexity-routing.*.all_passed}
                  **Confidence**: ${validate-fix.confidence_score}

                  [View detailed report](${validate-fix.validation_report})

                  Branch ready for review: `fix/${inputs.issue_id}`

        failed:
          conditions: ${validate-fix.is_valid} == false OR ${complexity-routing.*.all_passed} == false
          stages:
            - stage: escalate
              skill: escalate-to-human
              inputs:
                issue_id: ${inputs.issue_id}
                reason: "Automated fix validation failed"
                details: ${validate-fix.validation_report}

  memory:
    persistence: enabled
    scope: global
    keys:
      - bug_fix_history
      - complexity_patterns
      - success_rates_by_strategy

  github_integration:
    create_pr: conditional
    update_issue: always
    labels: ["automated-fix"]
```

## Execution Flow

```
┌─────────────────┐
│  Analyze Bug    │
│   (Stage 1)     │
└────────┬────────┘
         │ complexity_score
         ▼
┌─────────────────┐
│Root Cause (RCA) │
│   (Stage 2)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│      Complexity Routing (Stage 3)           │
│           (Conditional Branch)              │
└────────┬────────────────────────────────────┘
         │
   ┌─────┴──────┬──────────────┬──────────────┐
   │            │              │              │
   ▼            ▼              ▼              │
┌──────┐   ┌────────┐    ┌───────────┐       │
│Simple│   │ Medium │    │  Complex  │       │
│ <30  │   │ 30-70  │    │   >=70    │       │
└──┬───┘   └───┬────┘    └─────┬─────┘       │
   │           │               │              │
   │           │               │              │
   ▼           ▼               ▼              │
┌──────┐   ┌────────┐    ┌────────────┐      │
│Codex │   │Claude  │    │Gemini      │      │
│Quick │   │Analysis│    │Megacontext │      │
│Fix   │   │        │    │            │      │
└──┬───┘   └───┬────┘    └─────┬──────┘      │
   │           │               │              │
   │           ▼               ▼              │
   │      ┌────────┐    ┌────────────┐       │
   │      │Implement    │Codex       │       │
   │      │Fix     │    │Reasoning   │       │
   │      └───┬────┘    └─────┬──────┘       │
   │          │               │              │
   │          │               ▼              │
   │          │         ┌────────────┐       │
   │          │         │Claude      │       │
   │          │         │Implement   │       │
   │          │         └─────┬──────┘       │
   │          │               │              │
   ▼          ▼               ▼              │
┌──────────────────────────────────┐         │
│     Test Fix (Codex Sandbox)     │         │
│    3/5/10 iterations based on    │         │
│         complexity branch         │         │
└──────────────┬───────────────────┘         │
               │                             │
               └─────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │Validate Fix    │
                    │   (Stage 4)    │
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │Commit Decision │
                    │   (Stage 5)    │
                    └────────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌────────┐    ┌──────────┐   ┌────────┐
         │ Commit │    │  Review  │   │ Failed │
         │   PR   │    │ Required │   │Escalate│
         └────────┘    └──────────┘   └────────┘
```

## Conditional Logic Examples

### Example 1: Simple Bug (Complexity = 25)

```yaml
# Bug: Typo in function name
analyze-bug:
  complexity_score: 25
  estimated_effort: "simple"
  affected_files: ["src/utils/helpers.js"]

# Route to simple branch
complexity-routing:
  branch_taken: "simple"

  # Execute Codex quick fix
  codex-quick-fix:
    fix_code: |
      - function calcluateTotal(items) {
      + function calculateTotal(items) {
          return items.reduce((sum, item) => sum + item.price, 0);
      }
    modified_files: ["src/utils/helpers.js"]

  # Test in sandbox (3 iterations max)
  test-fix:
    iteration: 1
    all_passed: true
    test_results:
      unit_tests: "✅ 45/45 passed"
      integration_tests: "✅ 12/12 passed"

# Validation
validate-fix:
  is_valid: true
  confidence_score: 0.95

# Auto-commit (if enabled)
commit-decision:
  branch_taken: "commit"
  create-pr:
    pr_url: "https://github.com/company/repo/pull/1234"
    title: "Fix: Typo in calculateTotal function"
```

### Example 2: Medium Bug (Complexity = 55)

```yaml
# Bug: Race condition in async handler
analyze-bug:
  complexity_score: 55
  estimated_effort: "medium"
  affected_files: ["src/api/handlers/user.js", "src/db/queries.js"]

# Route to medium branch
complexity-routing:
  branch_taken: "medium"

  # Claude analysis
  claude-analysis:
    analysis_report: |
      Race condition detected in user creation flow:
      1. Concurrent requests can create duplicate users
      2. Database constraint not enforced at app level
      3. Missing transaction boundaries
    fix_plan: |
      1. Add distributed lock with Redis
      2. Wrap creation in database transaction
      3. Add retry logic for lock contention

  # Implement fix
  implement-fix:
    fix_code: |
      const lock = await acquireRedisLock(`user:create:${email}`);
      try {
        await db.transaction(async (trx) => {
          const existing = await trx('users').where({ email }).first();
          if (existing) throw new ConflictError();
          await trx('users').insert({ email, ...data });
        });
      } finally {
        await lock.release();
      }
    modified_files: ["src/api/handlers/user.js", "src/db/queries.js"]

  # Test in sandbox (5 iterations max)
  test-fix:
    iteration: 1
    all_passed: true
    test_results:
      unit_tests: "✅ 78/78 passed"
      integration_tests: "✅ 23/23 passed"
      race_condition_tests: "✅ 10/10 passed"

# Validation
validate-fix:
  is_valid: true
  confidence_score: 0.88

# Review required (manual approval)
commit-decision:
  branch_taken: "review-required"
  create-branch:
    branch_name: "fix/issue-5678"
  update-issue:
    comment_posted: true
```

### Example 3: Complex Bug (Complexity = 85)

```yaml
# Bug: Memory leak in event processing pipeline
analyze-bug:
  complexity_score: 85
  estimated_effort: "complex"
  affected_files: [
    "src/events/processor.js",
    "src/events/queue.js",
    "src/events/handlers/*.js",
    "src/memory/pool.js"
  ]
  requires_context: true

# Route to complex branch
complexity-routing:
  branch_taken: "complex"

  # Gemini megacontext analysis
  gemini-megacontext:
    architecture_insights: |
      Event processing system uses:
      - Custom memory pool with manual allocation
      - Event queue with 15 different handler types
      - Circular reference between handlers and pool
      - No garbage collection hooks

      Memory leak occurs when:
      1. Handler throws exception mid-processing
      2. Event remains in "processing" state
      3. Memory pool entry never released
      4. Pool grows unbounded over time

    dependency_map:
      processor → queue → handlers → pool → processor (CIRCULAR!)

    impact_analysis: |
      - Production memory usage grows 500MB/hour
      - Requires restart every 6 hours
      - Affects 45% of event types
      - Critical for payment processing events

  # Codex reasoning
  codex-reasoning:
    approach_candidates:
      - approach_1:
          name: "Break circular dependency"
          pros: ["Clean architecture", "Long-term fix"]
          cons: ["Large refactor", "Risky deployment"]
          risk: "high"

      - approach_2:
          name: "Add finally block for cleanup"
          pros: ["Simple", "Targeted fix", "Low risk"]
          cons: ["Doesn't fix circular dependency"]
          risk: "low"

      - approach_3:
          name: "WeakMap-based memory pool"
          pros: ["GC-friendly", "Modern approach"]
          cons: ["Requires Node 14+", "Performance impact"]
          risk: "medium"

    tradeoffs:
      selected: "approach_2"
      rationale: "Immediate fix with low risk, refactor in follow-up"

  # Claude implementation
  claude-implementation:
    fix_code: |
      async processEvent(event) {
        const poolEntry = this.pool.allocate(event.id);
        try {
          const handler = this.getHandler(event.type);
          const result = await handler.process(event, poolEntry);
          await this.queue.acknowledge(event.id);
          return result;
        } catch (error) {
          await this.queue.nack(event.id);
          throw error;
        } finally {
          // CRITICAL: Always release pool entry
          this.pool.release(poolEntry);

          // Remove circular reference
          poolEntry.handler = null;
          poolEntry.event = null;
        }
      }

    modified_files: [
      "src/events/processor.js",
      "src/memory/pool.js"
    ]

    migration_guide: |
      # Deployment Steps
      1. Deploy to canary (10% traffic)
      2. Monitor memory usage for 2 hours
      3. If stable, roll out to 50%
      4. Full rollout after 24 hours stable

  # Test in sandbox (10 iterations max)
  test-fix:
    iteration: 1
    all_passed: true
    test_results:
      unit_tests: "✅ 156/156 passed"
      integration_tests: "✅ 45/45 passed"
      memory_leak_tests: "✅ 8/8 passed (no leaks detected)"
      regression_tests: "✅ 234/234 passed"

# Validation
validate-fix:
  is_valid: true
  confidence_score: 0.92

# Review required for complex fix
commit-decision:
  branch_taken: "review-required"
  create-branch:
    branch_name: "fix/memory-leak-9012"
  update-issue:
    comment_posted: true
    additional_context: "Multi-model analysis (Gemini+Codex+Claude)"
```

## Performance Characteristics

```yaml
timing_by_complexity:
  simple:
    total: "45-60s"
    breakdown:
      analyze: 15s
      rca: 10s
      codex_fix: 15s
      test: 10s

  medium:
    total: "2-3min"
    breakdown:
      analyze: 20s
      rca: 25s
      claude_analysis: 40s
      implement: 50s
      test: 30s

  complex:
    total: "5-8min"
    breakdown:
      analyze: 30s
      rca: 45s
      gemini_megacontext: 90s
      codex_reasoning: 60s
      claude_implement: 120s
      test: 90s

success_rates:
  simple: 95%
  medium: 85%
  complex: 72%
```

## When to Use This Pattern

**Best for:**
- Adaptive workflows
- Multi-path decision making
- Risk-based routing
- Complexity-aware processing
- Resource optimization

**Not ideal for:**
- Always-same execution path
- Simple linear workflows
- Binary yes/no decisions (use simple if/else)

## Related Examples

- **example-1-sequential.md**: Sequential execution
- **example-2-parallel.md**: Parallel execution
- **references/orchestration-patterns.md**: Advanced conditional patterns

---

**Key Takeaway**: Conditional cascades enable intelligent workflow adaptation based on runtime analysis.
