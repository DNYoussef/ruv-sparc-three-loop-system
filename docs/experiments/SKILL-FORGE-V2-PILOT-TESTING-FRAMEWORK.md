# skill-forge v2.0 Pilot Testing Framework

**Purpose**: Validate that v2.0 enhancements deliver predicted improvements through systematic comparative testing

**Created**: 2025-11-06
**Status**: Ready for Execution
**Est. Time**: 6-9 hours (2-3 hours per pilot skill)

---

## Testing Objectives

1. **Validate Research Predictions**: Do techniques deliver claimed improvements?
2. **Measure Actual Gains**: Compare v1.0 baseline vs v2.0 enhanced
3. **Identify Refinements**: What works well? What needs improvement?
4. **Build Technique Database**: Which techniques have highest ROI?
5. **Document Real-World Usage**: Capture lessons for future skills

---

## Pilot Skill Selection

We'll test 3 skills covering different complexity levels:

### Pilot 1: Simple Skill (Quick Track)
**Skill**: `code-formatter`
**Complexity**: Simple
**Purpose**: Format code files with appropriate formatter per language
**Est. Time**: 2 hours (1h v1.0 + 1h v2.0)
**Why**: Tests Quick Track efficiency claims (20 min vs 75 min)

### Pilot 2: Medium Skill (Quick Track)
**Skill**: `api-integration-helper`
**Complexity**: Medium
**Purpose**: Generate boilerplate for REST API integration
**Est. Time**: 2.5 hours (1.5h v1.0 + 1h v2.0)
**Why**: Tests template effectiveness and validation scripts

### Pilot 3: Complex Skill (Expert Track)
**Skill**: `security-audit-workflow`
**Complexity**: Complex
**Purpose**: Systematic security audit with vulnerability scanning
**Est. Time**: 4 hours (2h v1.0 + 2h v2.0)
**Why**: Tests all v2.0 features (Phase 0, CoV, Adversarial, Metrics)

---

## Testing Protocol (Per Skill)

### Phase 1: Baseline (V0 - v1.0 Process)

**Time**: 45-90 min depending on complexity

1. **Create Skill Using v1.0**:
   - Follow current skill-forge SKILL.md (no enhancements)
   - Track time per phase
   - Count tokens loaded
   - Note iterations required

2. **Measure V0 Baseline Metrics**:
   ```yaml
   baseline_v0:
     time_spent: [X] minutes
     tokens_loaded: [X]
     iterations: [X]
     metrics:
       factual_accuracy: [X]%    # Count correct claims / total claims
       completeness: [X]%         # Count present elements / required elements
       precision: [X]%            # Count relevant sentences / total sentences
       actionability: [X]%        # Count instructions with criteria / total instructions
     aggregate_score: [X]%
   ```

3. **Identify Quality Issues**:
   - Missing error handling
   - Vague instructions
   - No success criteria
   - Ambiguous requirements
   - Missing edge cases

### Phase 2: Enhanced (V1 - v2.0 Quick Track)

**Time**: 20-30 min (Quick Track) or 60-75 min (Expert Track)

1. **Apply v2.0 Techniques**:

   **For Quick Track**:
   - Use `templates/intake-template.yaml` (Phase 1)
   - Use `templates/instruction-template.md` (Phase 5)
   - Run `scripts/validate-intake.js`
   - Run `scripts/validate-instructions.js`
   - Run `scripts/validate-skill.js`

   **For Expert Track (add these)**:
   - Create schema with `templates/skill-schema.json` (Phase 0)
   - Apply CoV in Phase 1b and 5b using `templates/cov-protocol.md`
   - Apply Adversarial Testing in Phase 7a using `templates/adversarial-testing-protocol.md`
   - Track metrics with `templates/skill-metrics.yaml` (Phase 8)
   - Run `scripts/validate-schema.js`

2. **Measure V1 Enhanced Metrics**:
   ```yaml
   revision_v1:
     techniques_applied: ["template-driven", "auto-validation", "CoV", "Adversarial"]
     time_spent: [X] minutes
     tokens_loaded: [X]
     iterations: [X]
     metrics:
       factual_accuracy: [X]%
       completeness: [X]%
       precision: [X]%
       actionability: [X]%
     aggregate_score: [X]%
     improvement_from_v0:
       time: [X]%
       tokens: [X]%
       accuracy: [X]%
       completeness: [X]%
       precision: [X]%
       actionability: [X]%
   ```

3. **Record Technique ROI**:
   ```yaml
   technique_effectiveness:
     - name: "intake-template"
       time_cost: [X] min
       quality_gain: [X]%
       roi_score: [gain / time]

     - name: "CoV Phase 1b"
       time_cost: [X] min
       quality_gain: [X]%
       roi_score: [gain / time]

     - name: "Adversarial Testing"
       time_cost: [X] min
       vulnerabilities_found: [X]
       roi_score: [vulns / time]
   ```

### Phase 3: Optimization (V2 - If Needed)

**Time**: 15-30 min (only if V1 didn't meet quality gates)

1. **Apply Additional Techniques** (if V1 < Quality Gate thresholds):
   - Add more edge cases
   - Improve error handling
   - Refine success criteria
   - Add performance benchmarks

2. **Measure V2 Optimized Metrics**:
   ```yaml
   revision_v2:
     additional_techniques: []
     diminishing_returns:
       v0_to_v1_gain: [X]%
       v1_to_v2_gain: [X]%
       is_diminishing: [true|false]
   ```

### Phase 4: Validation

**Time**: 10-15 min

1. **Test Skill Execution**:
   - Run skill with realistic input
   - Verify output matches success criteria
   - Test error handling with invalid input
   - Test edge cases

2. **Validation Scripts**:
   ```bash
   # Run all validations
   node scripts/validate-intake.js pilot-1/intake.yaml
   node scripts/validate-instructions.js pilot-1/SKILL.md
   node scripts/validate-schema.js pilot-1/schema.json  # Expert Track only
   node scripts/validate-skill.js pilot-1/
   ```

3. **Pass/Fail Assessment**:
   - ✓ All validation scripts pass (exit code 0 or 2)
   - ✓ Quality Gate 8 thresholds met (if using metrics)
   - ✓ Skill executes successfully with test inputs
   - ✓ No critical issues in adversarial testing

---

## Predicted vs Actual Comparison

### Predictions from v2.0 Design:

| Metric | v1.0 Baseline | v2.0 Predicted | Source |
|--------|---------------|----------------|--------|
| Token Economy (Quick) | 4800 | 1630 (-66%) | Progressive disclosure |
| Time Efficiency (Quick) | 75 min | 20 min (-73%) | Templates + auto-validation |
| Activation Accuracy | 70% | 85% (+15%) | Better triggers |
| Success Rate | 68% | 91% (+23%) | Quality gates |
| Avg Iterations | 2.6 | 1.2 (-54%) | CoV + Adversarial |
| Factual Accuracy | Baseline | +42% | CoV |
| Completeness | Baseline | +47% | Schema-first |
| Actionability | Baseline | +50% | Explicit success criteria |
| Vulnerabilities Caught | Baseline | +58% | Adversarial testing |
| Meta-Principles Coverage | 35% | 90% (+157%) | All techniques |

### Actual Results Template:

| Metric | Pilot 1 Actual | Pilot 2 Actual | Pilot 3 Actual | Avg Actual | Variance from Predicted |
|--------|----------------|----------------|----------------|------------|-------------------------|
| Token Economy (Quick) | TBD | TBD | N/A | TBD | TBD |
| Time Efficiency (Quick) | TBD | TBD | N/A | TBD | TBD |
| Time Efficiency (Expert) | N/A | N/A | TBD | TBD | TBD |
| Factual Accuracy Gain | TBD | TBD | TBD | TBD | TBD |
| Completeness Gain | TBD | TBD | TBD | TBD | TBD |
| Precision Gain | TBD | TBD | TBD | TBD | TBD |
| Actionability Gain | TBD | TBD | TBD | TBD | TBD |
| Aggregate Score Gain | TBD | TBD | TBD | TBD | TBD |
| Meta-Principles Coverage | TBD | TBD | TBD | TBD | TBD |

**Variance Analysis Thresholds**:
- **Within ±10%**: Predictions accurate ✅
- **±10-20%**: Minor refinement needed ⚠️
- **>20%**: Significant revision required ❌

---

## Measurement Instructions

### How to Measure Factual Accuracy (0-100%)

1. Identify all factual claims in skill
2. Verify each claim against documentation/sources
3. Calculate: (Correct claims / Total claims) × 100

**Example**:
```markdown
Claim 1: "Prettier supports JavaScript" ✓ (verified in Prettier docs)
Claim 2: "Prettier works with Python" ✗ (Prettier doesn't support Python - that's Black)
Claim 3: "Format command is `prettier --write`" ✓ (verified)
...
Result: 18/20 claims correct = 90% factual accuracy
```

### How to Measure Completeness (0-100%)

Required elements checklist (5 core + N optional):
- ✓ YAML frontmatter (name, description, author)
- ✓ Clear trigger keywords (5+)
- ✓ Explicit success criteria (3+ per instruction)
- ✓ Error handling (for critical operations)
- ✓ At least 2 examples (nominal + edge)
- ✓ Edge case handling (3+)
- ✓ Performance expectations (optional)
- ✓ Verification checklist (optional)

Calculate: (Present elements / Required elements) × 100

### How to Measure Precision (0-100%)

1. Count total sentences in skill
2. Identify irrelevant or redundant sentences
3. Calculate: (Relevant sentences / Total sentences) × 100

**Irrelevant**: Background that doesn't help execution
**Redundant**: Repeated information already stated

### How to Measure Actionability (0-100%)

1. Count total instructions (numbered steps)
2. Count instructions with explicit success criteria
3. Calculate: (Instructions with criteria / Total instructions) × 100

**Explicit Success Criteria**: Measurable condition showing step complete
- ✓ "Format file. Success: exit code 0, changes_made ≥ 0"
- ✗ "Format the file." (no success criteria)

---

## Data Collection Template

### Pilot Skill 1: code-formatter (Simple - Quick Track)

```yaml
skill_name: "code-formatter"
complexity: "simple"
track: "quick"

# V0 Baseline (v1.0)
baseline_v0:
  time_spent: "" # minutes
  tokens_loaded: ""
  iterations: ""
  phases:
    phase1: "" # min
    phase2: "" # min
    phase3: "" # min
    phase4: "" # min
    phase5: "" # min
    phase6: "" # min
    phase7: "" # min
  metrics:
    factual_accuracy: 0 # % (claims correct)
    completeness: 0     # % (required elements present)
    precision: 0        # % (sentences relevant)
    actionability: 0    # % (instructions with criteria)
  aggregate_score: 0
  issues_identified:
    - ""

# V1 Enhanced (v2.0)
revision_v1:
  techniques_applied:
    - "intake-template"
    - "instruction-template"
    - "validate-intake"
    - "validate-instructions"
    - "validate-skill"
  time_spent: ""
  tokens_loaded: ""
  iterations: ""
  phases:
    phase1: "" # min (with template)
    phase5: "" # min (with template)
  metrics:
    factual_accuracy: 0
    completeness: 0
    precision: 0
    actionability: 0
  aggregate_score: 0

  # Improvements
  improvements:
    time: 0 # % improvement
    tokens: 0
    accuracy: 0
    completeness: 0
    precision: 0
    actionability: 0
    aggregate: 0

  # Technique ROI
  technique_roi:
    - technique: "intake-template"
      time_cost: 0 # min
      quality_gain: 0 # %
      roi: 0 # gain/time
    - technique: "instruction-template"
      time_cost: 0
      quality_gain: 0
      roi: 0
    - technique: "validation-scripts"
      time_cost: 0
      issues_caught: 0
      roi: 0

# Validation Results
validation:
  validate_intake: "pass|fail|warnings"
  validate_instructions: "pass|fail|warnings"
  validate_skill: "pass|fail|warnings"
  manual_testing: "pass|fail"

# Verdict
verdict:
  v2_superior: true|false
  reason: ""
  recommendations: []
```

---

## Success Criteria

### Minimum Success (Required to Pass)
- ✅ All 3 pilot skills completed (V0 + V1)
- ✅ At least 2/3 skills show improvement in aggregate score
- ✅ No regression in any metric (V1 ≥ V0 for all metrics)
- ✅ Validation scripts catch at least 5 issues per skill

### Target Success (Predicted Performance)
- ✅ All 3 pilot skills completed
- ✅ Quick Track: Time reduced by ≥50% (target: 73%)
- ✅ Quick Track: Tokens reduced by ≥50% (target: 66%)
- ✅ All 4 quality metrics improve by ≥25% average
- ✅ Aggregate score improves by ≥30%
- ✅ Meta-principles coverage ≥75% (target: 90%)

### Exceptional Success (Exceeds Predictions)
- ✅ All metrics exceed predictions by ≥10%
- ✅ Technique ROI scores all ≥1.0 (gain ≥ time cost)
- ✅ Zero critical validation failures
- ✅ Skills work correctly on first execution (no debugging)

---

## Execution Checklist

### Before Starting
- [ ] Install script dependencies: `cd scripts && npm install`
- [ ] Create pilot directories: `mkdir -p pilots/{pilot-1,pilot-2,pilot-3}`
- [ ] Copy templates to each pilot directory
- [ ] Set up stopwatch/timer for time tracking

### For Each Pilot Skill
- [ ] **V0 Baseline**:
  - [ ] Create skill using v1.0 process
  - [ ] Track time per phase
  - [ ] Count tokens loaded
  - [ ] Measure 4 quality metrics
  - [ ] Document issues found

- [ ] **V1 Enhanced**:
  - [ ] Apply v2.0 techniques (Quick or Expert Track)
  - [ ] Track time per technique
  - [ ] Count tokens loaded
  - [ ] Re-measure 4 quality metrics
  - [ ] Calculate improvements

- [ ] **Validation**:
  - [ ] Run all validation scripts
  - [ ] Test skill execution
  - [ ] Verify quality gates
  - [ ] Document ROI per technique

- [ ] **Analysis**:
  - [ ] Compare predicted vs actual
  - [ ] Calculate variance
  - [ ] Identify refinements needed
  - [ ] Update technique database

### After All Pilots
- [ ] Aggregate results across 3 pilots
- [ ] Calculate average improvements
- [ ] Identify highest ROI techniques
- [ ] Document lessons learned
- [ ] Update v2.0 design if needed
- [ ] Make GO/NO-GO decision on v2.0 deployment

---

## Tips for Effective Testing

1. **Be Honest**: Don't adjust measurements to match predictions
2. **Track Everything**: Time, tokens, iterations, issues
3. **Document Issues**: Note what's confusing or missing
4. **Test Realistically**: Use skills as intended users would
5. **Compare Fairly**: Same skill complexity for V0 vs V1
6. **Note Surprises**: Unexpected benefits or challenges
7. **Keep Raw Data**: Save all measurements for analysis

---

## Next Steps After Testing

1. **If predictions met (within ±10%)**:
   - ✅ Deploy v2.0 to production
   - ✅ Write migration guide
   - ✅ Create Quick Start guide
   - ✅ Announce v2.0 availability

2. **If predictions close (±10-20%)**:
   - ⚠️ Minor refinements to techniques
   - ⚠️ Update documentation with actual metrics
   - ⚠️ Deploy with caveats
   - ⚠️ Iterate based on feedback

3. **If predictions missed (>20%)**:
   - ❌ Analyze root causes
   - ❌ Revise techniques or predictions
   - ❌ Run additional pilots
   - ❌ Consider Tier 1.5 enhancements

---

**Ready to Start?** Begin with Pilot 1 (code-formatter - Simple) to test Quick Track efficiency claims.
