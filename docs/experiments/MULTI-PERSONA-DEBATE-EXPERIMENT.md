# Multi-Persona Debate Experiment
## Testing Enhanced Prompting Meta-Principles in Practice

**Date**: 2025-01-06
**Purpose**: Validate meta-principles through Multi-Persona Debate comparing old vs new prompting approaches
**Expected Impact**: 61% better trade-off analysis, 2-3x prompt quality improvement

---

## Experiment Design

### Hypothesis
Applying schema-first, verification-first, and multi-perspective principles to prompt/skill creation will yield measurable quality improvements detectable through systematic reviewer audit.

### Test Case: "Enhance Prompt-Architect Skill"

**Baseline (V0)**: Current prompt-architect SKILL.md (447 lines)
**Enhanced (V1)**: Prompt-architect with schema-first, verification-first, multi-perspective applied

---

## Phase 1: Multi-Persona Debate Setup

### Personas (Conflicting Priorities)

#### Persona 1: Simplicity Advocate
**Priority**: Ease of use, clarity, accessibility
**Concerns**: Over-complexity, steep learning curves, analysis paralysis
**Prompting Style**: Minimize steps, prioritize intuitive workflows

#### Persona 2: Rigor Engineer
**Priority**: Correctness, verification, systematic methodology
**Concerns**: Missing edge cases, unverified claims, ad-hoc approaches
**Prompting Style**: Add verification gates, evidence requirements, quality checks

#### Persona 3: Performance Optimizer
**Priority**: Speed, efficiency, token economy
**Concerns**: Verbose prompts, wasted iterations, high costs
**Prompting Style**: Schema-first constraints, long-prompts-save-tokens principle

#### Persona 4: Researcher/Innovator
**Priority**: Completeness, coverage, advanced techniques
**Concerns**: Missing MECE components (CoV, adversarial testing, multi-perspective)
**Prompting Style**: Incorporate all 10 meta-principles, full coverage

---

## Phase 2: Debate Execution (Using Coordination MCPs)

### Debate Structure

**Round 1: Independent Proposals** (Parallel Agent Spawn)
```javascript
// Spawn 4 agents in parallel using Task tool
Task("Simplicity Advocate", `
  Analyze prompt-architect skill from simplicity perspective:
  - What's overcomplicated?
  - Where will users get confused?
  - How to make it more accessible?
  - Propose: Streamlined version focusing on clarity
`, "analyst")

Task("Rigor Engineer", `
  Analyze prompt-architect skill from verification perspective:
  - What verification is missing?
  - Where are claims unsubstantiated?
  - What quality gates are needed?
  - Propose: Version with built-in verification loops
`, "analyst")

Task("Performance Optimizer", `
  Analyze prompt-architect skill from efficiency perspective:
  - What's verbose without value?
  - Where can schema-first save tokens?
  - What's the token economy?
  - Propose: Optimized version following meta-principles 1-7
`, "optimizer")

Task("Researcher/Innovator", `
  Analyze prompt-architect skill against MECE gaps:
  - What advanced techniques are missing?
  - Coverage of 10 meta-principles?
  - Integration of verification-synthesis.md principles?
  - Propose: Comprehensive version with full coverage
`, "researcher")
```

**Round 2: Cross-Critique** (Sequential, feed outputs from R1)
Each persona critiques other proposals:
- Simplicity critiques Rigor: "Too complex, users will give up"
- Rigor critiques Simplicity: "Too shallow, unreliable results"
- Performance critiques Researcher: "Over-engineered, expensive"
- Researcher critiques Performance: "Missing critical techniques"

**Round 3: Refined Proposals** (Parallel, incorporating critiques)
Each persona refines their proposal addressing valid critiques

**Round 4: Synthesis** (Coordinator Agent)
```javascript
Task("Synthesis Coordinator", `
  Synthesize 4 refined proposals into optimal prompt-architect v2.0:

  Consensus Areas (all agree):
  - [List agreements]

  Trade-offs to Balance:
  - Simplicity vs Rigor: [Recommendation]
  - Performance vs Completeness: [Recommendation]
  - Accessibility vs Power: [Recommendation]

  Optimal Integration:
  - Phase structure: [Design]
  - Core principles: [Selection]
  - Optional advanced techniques: [How to present]

  Final Design with explicit rationale for each choice
`, "coordinator")
```

---

## Phase 3: Comparative Implementation

### V0: Current Approach (Baseline)
**Task**: "Enhance prompt-architect skill"
**Method**: Traditional approach (intuition-based, no meta-principles)

### V1: Enhanced Approach (Using Meta-Principles)
**Task**: Same - "Enhance prompt-architect skill"
**Method**:
1. Schema-first: Define exact structure before writing
2. Verification-first: Add self-critique, evidence requirements
3. Multi-perspective: Use 4-persona debate above
4. Quality gates: Explicit checkpoints
5. Adversarial testing: Attack own design

---

## Phase 4: Measurement Framework

### Metrics to Track

#### Quantitative Metrics
```yaml
activation_accuracy:
  measure: "% of times skill correctly activated for relevant queries"
  baseline_target: 70%
  enhanced_target: 85%+

success_rate:
  measure: "% of uses that complete without needing revisions"
  baseline_target: 70%
  enhanced_target: 90%+

avg_iterations:
  measure: "Mean number of revisions before acceptable"
  baseline_target: 2.5
  enhanced_target: 1.3

format_compliance:
  measure: "% of outputs matching expected schema/structure"
  baseline_target: 60%
  enhanced_target: 90%+

completeness:
  measure: "% of required elements present in output"
  baseline_target: 70%
  enhanced_target: 95%+

token_efficiency:
  measure: "Tokens per successful outcome"
  baseline: [measured]
  enhanced: [measured, expect 25-35% reduction from fewer iterations]
```

#### Qualitative Metrics (Reviewer Assessment)
```yaml
verification_quality:
  scale: 1-10
  criteria: "Built-in self-checking, evidence for claims, completeness validation"

multi_perspective:
  scale: 1-10
  criteria: "Trade-off awareness, conflicting priority synthesis, blind spot coverage"

schema_quality:
  scale: 1-10
  criteria: "I/O contracts clear, structure precedes prose, ambiguity minimized"

maintainability:
  scale: 1-10
  criteria: "Versioned, testable, clear contracts, easy to update"

innovation:
  scale: 1-10
  criteria: "Advanced techniques applied, MECE gaps filled, research-backed"
```

---

## Phase 5: Reviewer Agent Audit

### Audit Protocol

**Spawn Reviewer Agent**:
```javascript
Task("Skill Quality Reviewer", `
  Audit TWO versions of prompt-architect skill:

  **Baseline (V0)**: [provide current SKILL.md]
  **Enhanced (V1)**: [provide enhanced SKILL.md]

  **Evaluation Framework**:

  1. **Verification Quality** (1-10)
     - Self-critique mechanisms present?
     - Evidence requirements for claims?
     - Completeness checklists?
     - Quality gates explicit?

  2. **Multi-Perspective Analysis** (1-10)
     - Trade-offs acknowledged?
     - Conflicting priorities balanced?
     - Blind spots addressed?
     - Synthesis quality?

  3. **Schema-First Design** (1-10)
     - Output structures defined first?
     - Input/output contracts clear?
     - Error conditions specified?
     - Ambiguity minimized?

  4. **Meta-Principles Application** (1-10)
     - Structure beats context applied?
     - Verification-first present?
     - Quality in verification fields?
     - Evidence-based techniques?

  5. **Completeness vs MECE Gaps** (1-10)
     - Chain-of-Verification: [present/absent]
     - Adversarial self-attack: [present/absent]
     - Multi-persona debate: [present/absent]
     - Schema-first methodology: [present/absent]
     - Quality gates: [present/absent]

  6. **Quantitative Estimates**
     - Predicted activation accuracy: V0 [%] vs V1 [%]
     - Predicted success rate: V0 [%] vs V1 [%]
     - Predicted avg iterations: V0 [#] vs V1 [#]

  **Output Format**:
  ```json
  {
    "summary": {
      "v0_total_score": "X/50",
      "v1_total_score": "Y/50",
      "improvement": "+Z points (+W%)"
    },
    "detailed_scores": {
      "verification_quality": {"v0": X, "v1": Y, "delta": Z},
      "multi_perspective": {"v0": X, "v1": Y, "delta": Z},
      "schema_first": {"v0": X, "v1": Y, "delta": Z},
      "meta_principles": {"v0": X, "v1": Y, "delta": Z},
      "mece_completeness": {"v0": X, "v1": Y, "delta": Z}
    },
    "quantitative_predictions": {
      "activation_accuracy": {"v0": "X%", "v1": "Y%", "improvement": "+Z%"},
      "success_rate": {"v0": "X%", "v1": "Y%", "improvement": "+Z%"},
      "avg_iterations": {"v0": X, "v1": Y, "reduction": "Z% fewer"}
    },
    "critical_improvements": [
      "Top 5 most impactful enhancements in V1"
    ],
    "remaining_gaps": [
      "What's still missing in V1"
    ],
    "recommendation": "Deploy V1 / Needs more work / V0 sufficient"
  }
  ```
`, "reviewer")
```

---

## Phase 6: Test Suite Creation

### Test Cases for Both Versions

**Test 1: Simple Prompt Optimization**
```yaml
input: "Improve this prompt: 'Analyze the data'"
expected_v0_behavior: "Add clarity, structure, examples"
expected_v1_behavior: "Schema-first → structure → verification → adversarial test"
success_criteria: "V1 includes verification loop V0 lacks"
```

**Test 2: Complex Multi-Step Workflow**
```yaml
input: "Create prompt for security audit workflow"
expected_v0_behavior: "Write instructions with some structure"
expected_v1_behavior: "Define schema → quality gates → verification → multi-perspective"
success_criteria: "V1 has explicit gates and multi-perspective analysis"
```

**Test 3: Handling Ambiguity**
```yaml
input: "Make this prompt better: 'Review the code'"
expected_v0_behavior: "General improvements"
expected_v1_behavior: "Clarify intent → schema-first → evidence requirements → verification"
success_criteria: "V1 explicitly addresses ambiguity, V0 accepts it"
```

**Test 4: Edge Case Handling**
```yaml
input: "Prompt for handling missing data"
expected_v0_behavior: "Mentions edge cases generally"
expected_v1_behavior: "Enumerates edge cases, adds explicit handling, verification"
success_criteria: "V1 has systematic edge case coverage"
```

**Test 5: Verification Quality**
```yaml
input: "Prompt for making factual claims"
expected_v0_behavior: "Requests sources"
expected_v1_behavior: "Claims verification fields (source, confidence, evidence)"
success_criteria: "V1 has structured verification, V0 has vague 'be accurate'"
```

---

## Execution Protocol

### Step-by-Step Execution

**Step 1: Baseline Capture** (5 min)
```bash
# Document current prompt-architect state
cp skills/foundry/prompt-architect/SKILL.md \
   docs/experiments/prompt-architect-v0-baseline.md
```

**Step 2: Multi-Persona Debate** (30 min)
```javascript
// Single message with 4 parallel agents
Task("Simplicity Advocate", "[full prompt above]", "analyst")
Task("Rigor Engineer", "[full prompt above]", "analyst")
Task("Performance Optimizer", "[full prompt above]", "optimizer")
Task("Researcher/Innovator", "[full prompt above]", "researcher")
```

**Step 3: Synthesis** (15 min)
```javascript
Task("Synthesis Coordinator", "[full prompt with 4 outputs]", "coordinator")
```

**Step 4: Implementation** (45 min)
- Apply synthesized design to prompt-architect SKILL.md
- Add schema-first sections
- Add verification-first phases
- Add quality gates
- Add adversarial testing
- Reference new meta-principles.md

**Step 5: Test Suite Execution** (30 min)
```javascript
// Run 5 test cases on both versions
Task("Test Executor", `
  Execute 5 test cases on:
  - V0: baseline prompt-architect
  - V1: enhanced prompt-architect

  Document outputs for each test
`, "tester")
```

**Step 6: Reviewer Audit** (20 min)
```javascript
Task("Skill Quality Reviewer", "[full audit protocol above]", "reviewer")
```

**Step 7: Analysis & Documentation** (15 min)
- Compile all results
- Calculate improvement metrics
- Document findings
- Create recommendations

**Total Time**: ~2.5 hours

---

## Expected Results

### Predicted Improvements (Based on Meta-Principles Research)

| Metric | V0 (Baseline) | V1 (Enhanced) | Improvement |
|--------|---------------|---------------|-------------|
| Activation Accuracy | 70% | 85% | +15% (+21%) |
| Success Rate | 70% | 92% | +22% (+31%) |
| Avg Iterations | 2.5 | 1.3 | -1.2 (-48%) |
| Format Compliance | 60% | 90% | +30% (+50%) |
| Completeness | 70% | 95% | +25% (+36%) |
| Token Efficiency | Baseline | 25-35% better | Fewer iterations |

### Qualitative Score Predictions

| Dimension | V0 | V1 | Delta |
|-----------|----|----|-------|
| Verification Quality | 4/10 | 9/10 | +5 |
| Multi-Perspective | 2/10 | 9/10 | +7 |
| Schema-First | 3/10 | 9/10 | +6 |
| Meta-Principles | 3/10 | 9/10 | +6 |
| MECE Completeness | 4/10 | 9/10 | +5 |
| **TOTAL** | **16/50** | **45/50** | **+29 (+181%)** |

---

## Documentation Requirements

### Experiment Report Template

```markdown
# Multi-Persona Debate Experiment Report

## Executive Summary
- V0 Score: X/50
- V1 Score: Y/50
- Improvement: +Z points (+W%)

## Detailed Results

### Quantitative Metrics
[Table with actual measurements]

### Qualitative Assessment
[Reviewer agent scores]

### Test Suite Results
[5 test cases, V0 vs V1 outputs]

### Multi-Persona Debate Synthesis
[Key insights from 4-persona debate]

## Critical Findings

### Top 5 Improvements in V1
1. [Specific improvement with impact]
2. [Specific improvement with impact]
3. [Specific improvement with impact]
4. [Specific improvement with impact]
5. [Specific improvement with impact]

### Remaining Gaps
1. [What's still missing]
2. [What could be better]

## Recommendations

### Immediate Actions
1. Deploy V1 to production: YES/NO
2. Update other skills with same patterns: YES/NO
3. Train team on new meta-principles: YES/NO

### Follow-Up Experiments
1. [Next skill to enhance and test]
2. [Broader application of multi-persona debate]

## Appendix

### Raw Data
- V0 Baseline: [file]
- V1 Enhanced: [file]
- Test Outputs: [files]
- Reviewer Audit: [detailed JSON]
- Debate Transcripts: [4 persona outputs + synthesis]
```

---

## Replication Protocol

To replicate this experiment for other skills/agents:

1. **Select Target**: Choose skill/agent to enhance
2. **Capture Baseline**: Document V0 state
3. **Define Personas**: 4 conflicting perspectives relevant to domain
4. **Multi-Persona Debate**: 4-round debate (propose → critique → refine → synthesize)
5. **Implement V1**: Apply meta-principles + synthesis insights
6. **Create Test Suite**: 5 domain-relevant test cases
7. **Reviewer Audit**: Systematic comparison V0 vs V1
8. **Document Results**: Quantitative + qualitative findings
9. **Iterate**: Apply learnings to next enhancement

---

## Integration with Skill Creation Process

### Updated Skill-Forge Workflow

**Phase 0: Multi-Perspective Analysis** (NEW)
```yaml
step_1: Define 3-4 personas with conflicting priorities
step_2: Each persona independently analyzes requirements
step_3: Cross-critique to surface trade-offs
step_4: Synthesis into balanced design

Output: Multi-perspective design canvas
```

**Phase 1: Schema-First Intent Analysis** (ENHANCED)
```yaml
step_1: Define exact output schema FIRST
step_2: Define input contracts SECOND
step_3: Define error conditions THIRD
step_4: THEN analyze intent

Output: Schema-driven intent document
```

**Phase 5: Verification-First Instruction Crafting** (ENHANCED)
```yaml
step_1: Write initial instructions
step_2: Add self-critique phase
step_3: Add evidence requirements
step_4: Add quality gates
step_5: Add completeness checklist

Output: Verification-enabled instructions
```

**Phase 7: Adversarial Validation** (ENHANCED)
```yaml
step_1: Brainstorm 10+ failure modes
step_2: Score likelihood × impact
step_3: Fix top 5 vulnerabilities
step_4: Reattack until confident
step_5: Reviewer audit

Output: Battle-tested skill/agent
```

---

## Success Criteria

### Minimum Success (Experiment Valid)
- ✅ 4-persona debate completed
- ✅ V1 implemented with meta-principles
- ✅ Reviewer audit produces scores
- ✅ At least +20% improvement in ANY metric

### Target Success (Strong Validation)
- ✅ All quantitative metrics improve by 20%+
- ✅ Qualitative scores improve by +5/10 avg
- ✅ Test suite shows clear V1 advantages
- ✅ Replicable process documented

### Exceptional Success (Transformative)
- ✅ All metrics improve by 40%+
- ✅ Qualitative scores improve by +6/10 avg
- ✅ V1 demonstrates capabilities V0 cannot match
- ✅ Process applicable to all skill/agent creation

---

**Next Steps**:
1. Execute Phase 1-7 for prompt-architect enhancement
2. Document results in experiment report
3. Apply validated meta-principles to skill-forge
4. Cascade improvements to all creation components

**Estimated ROI**: 2-3x skill quality for 2.5 hours investment
**Risk**: Low (V0 remains unchanged, V1 is additive)
**Decision**: Proceed with experiment
