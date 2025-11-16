# Pilot 1: code-formatter - Execution Guide

**Skill**: code-formatter
**Complexity**: Simple
**Track**: Quick Track
**Estimated Time**: 2 hours (1h V0 + 1h V1)

---

## Overview

This is the first pilot skill to validate skill-forge v2.0 Quick Track predictions:
- **Time**: 75 min (v1.0) → 20 min (v2.0) = -73% reduction
- **Tokens**: 4800 → 1630 = -66% reduction
- **Quality**: +30-50% improvement across 4 metrics

---

## Pilot 1 - Part A: V0 Baseline (Using v1.0 Process)

**Time Budget**: 60 minutes
**Goal**: Create code-formatter skill using current v1.0 process

### Phase 1: Intent Archaeology (15 min)

**Start Timer**

**Task**: Analyze the requirement without any templates or CoV

**Requirement**:
> "Create a skill that formats code files automatically. It should detect the programming language, use the appropriate formatter (Prettier for JS, Black for Python, rustfmt for Rust), and provide clear feedback on what was changed."

**Your Analysis** (write in free-form, no template):
- What is the primary use case?
- Who will use this?
- What are the key requirements?
- What constraints exist?
- What edge cases might occur?

**Document your analysis** in natural prose (no structured format).

**Time Spent**: _____ min

---

### Phase 2: Use Case Crystallization (10 min)

**Task**: Generate 3-5 example scenarios

Write examples showing:
1. Nominal case (typical use)
2. Edge case (unusual but valid)
3. Error case (something goes wrong)

**Format**: Free-form, however you naturally write examples

**Time Spent**: _____ min

---

### Phase 3: Structural Architecture (10 min)

**Task**: Decide on skill structure

Consider:
- Will there be scripts? If so, what?
- Any template files needed?
- Reference documentation?
- How to organize?

**Document your structure** in outline form.

**Time Spent**: _____ min

---

### Phase 4: Metadata Engineering (5 min)

**Task**: Write YAML frontmatter

```yaml
---
name: code-formatter
description: [write description]
author: [your name]
---
```

Add any other metadata you think is important.

**Time Spent**: _____ min

---

### Phase 5: Instruction Crafting (15 min)

**Task**: Write instructions for Claude

Write step-by-step instructions on how to:
1. Detect file language
2. Choose appropriate formatter
3. Run formatter
4. Handle errors
5. Provide feedback

**Format**: However you naturally write instructions (no template)

**Time Spent**: _____ min

---

### Phase 6: Resource Development (Skip for V0 baseline)

**Time Spent**: 0 min

---

### Phase 7: Validation (5 min)

**Task**: Quick review

- Read through skill
- Check for obvious issues
- Test mentally with an example

**Time Spent**: _____ min

---

### V0 Baseline Complete

**Total Time**: _____ min (target: 60 min)

**Now measure V0 quality metrics**:

#### Factual Accuracy (0-100%)
Count all factual claims in your skill (e.g., "Prettier formats JavaScript", "Black supports Python", etc.)

- Total claims: _____
- Correct claims: _____ (verify against documentation)
- Accuracy: (correct / total) × 100 = _____%

#### Completeness (0-100%)
Required elements checklist:
- [ ] YAML frontmatter (name, description)
- [ ] Clear trigger keywords (5+)
- [ ] Explicit success criteria (3+ per instruction)
- [ ] Error handling (for critical operations)
- [ ] At least 2 examples (nominal + edge)

Elements present: _____ / 5
Completeness: (present / 5) × 100 = _____%

#### Precision (0-100%)
- Total sentences: _____
- Irrelevant/redundant sentences: _____
- Relevant sentences: _____ (total - irrelevant)
- Precision: (relevant / total) × 100 = _____%

#### Actionability (0-100%)
- Total instructions (numbered steps): _____
- Instructions with explicit success criteria: _____
- Actionability: (with criteria / total) × 100 = _____%

#### Aggregate Score
- Average of 4 metrics: _____%

**Issues Identified in V0**:
1. _____
2. _____
3. _____

---

## Pilot 1 - Part B: V1 Enhanced (Using v2.0 Quick Track)

**Time Budget**: 20 minutes
**Goal**: Create same skill using v2.0 templates and validation

### Setup (1 min)

Copy templates:
```bash
cp templates/intake-template.yaml pilots/pilot-1-code-formatter/intake.yaml
cp templates/instruction-template.md pilots/pilot-1-code-formatter/instructions.md
```

**Start Timer**

---

### Phase 1: Intake (5 min)

**Task**: Fill in `intake.yaml` template

Open `pilots/pilot-1-code-formatter/intake.yaml` and fill in:
- skill_name
- skill_category
- complexity_level
- problem_solved
- desired_outcome
- primary_users (1+)
- trigger_keywords (5+)
- negative_triggers (3+)
- example_usage_1, 2, 3 (nominal, edge, error)
- constraints (3+)
- must_have_features (3+)
- success_criteria (3+)
- failure_conditions (3+)

**Time Spent**: _____ min

**Validate**:
```bash
cd scripts
node validate-intake.js ../pilots/pilot-1-code-formatter/intake.yaml
```

**Validation Result**: PASS / FAIL / WARNINGS
**Issues Caught**: _____

---

### Phase 2: Instructions (10 min)

**Task**: Customize `instructions.md` template

Open `pilots/pilot-1-code-formatter/instructions.md` and:
1. Replace all [PLACEHOLDER] sections
2. Fill in Steps 1-4 with code-formatter logic
3. Add edge cases (3+)
4. Document error codes
5. Add success criteria per step

**Time Spent**: _____ min

**Validate**:
```bash
node validate-instructions.js ../pilots/pilot-1-code-formatter/instructions.md
```

**Validation Result**: PASS / FAIL / WARNINGS
**Issues Caught**: _____

---

### Phase 3: Final Skill Assembly (3 min)

**Task**: Create `SKILL.md` from intake + instructions

Combine:
1. YAML frontmatter (from intake metadata)
2. Overview (from problem_solved + desired_outcome)
3. Instructions (from instructions.md)

**Time Spent**: _____ min

---

### Phase 4: Final Validation (1 min)

**Validate Complete Skill**:
```bash
node validate-skill.js ../pilots/pilot-1-code-formatter/
```

**Validation Result**: PASS / FAIL / WARNINGS
**Final Issues**: _____

---

### V1 Enhanced Complete

**Total Time**: _____ min (target: 20 min)

**Now measure V1 quality metrics** (same process as V0):

#### Factual Accuracy
- Total claims: _____
- Correct claims: _____
- Accuracy: _____%

#### Completeness
- Elements present: _____ / 5
- Completeness: _____%

#### Precision
- Total sentences: _____
- Relevant sentences: _____
- Precision: _____%

#### Actionability
- Total instructions: _____
- With success criteria: _____
- Actionability: _____%

#### Aggregate Score
- Average: _____%

---

## Pilot 1 - Results Analysis

### Time Comparison

| Metric | V0 (v1.0) | V1 (v2.0) | Improvement | Predicted | Variance |
|--------|-----------|-----------|-------------|-----------|----------|
| **Total Time** | _____ min | _____ min | ____% | -73% | _____ |
| **Phase 1 Time** | _____ min | _____ min | ____% | -75% (15→5) | _____ |
| **Phase 5 Time** | _____ min | _____ min | ____% | -67% (15→10) | _____ |

### Quality Comparison

| Metric | V0 | V1 | Improvement | Predicted | Variance |
|--------|----|----|-------------|-----------|----------|
| **Factual Accuracy** | ____% | ____% | +____% | +42% | _____ |
| **Completeness** | ____% | ____% | +____% | +47% | _____ |
| **Precision** | ____% | ____% | +____% | +25% | _____ |
| **Actionability** | ____% | ____% | +____% | +50% | _____ |
| **Aggregate** | ____% | ____% | +____% | +41% | _____ |

### Technique ROI

| Technique | Time Cost | Quality Gain | ROI | Notes |
|-----------|-----------|--------------|-----|-------|
| intake-template | _____ min | +____% | _____ | _____ |
| instruction-template | _____ min | +____% | _____ | _____ |
| validate-intake | _____ min | _____ issues | _____ | _____ |
| validate-instructions | _____ min | _____ issues | _____ | _____ |
| validate-skill | _____ min | _____ issues | _____ | _____ |

### Tokens Loaded

| Version | Tokens | Improvement |
|---------|--------|-------------|
| **V0 (v1.0)** | ~4800 (full SKILL.md) | Baseline |
| **V1 (v2.0)** | intake (___) + instructions (___) = _____ | ____% |
| **Predicted** | 1630 | -66% |

---

## Key Findings

### What Worked Well
1. _____
2. _____
3. _____

### What Didn't Work
1. _____
2. _____
3. _____

### Surprises
1. _____
2. _____

### Issues with v2.0 Process
1. _____
2. _____

### Recommendations
1. _____
2. _____

---

## Verdict

**Overall Assessment**: V2.0 is SUPERIOR / SIMILAR / INFERIOR to V1.0

**Reasoning**: _____

**Confidence in Predictions**: HIGH / MEDIUM / LOW

**Proceed with Pilots 2-3**: YES / NO / REFINE FIRST

---

## Next Steps

If results are promising (improvements within ±20% of predictions):
→ Proceed to **Pilot 2: api-integration-helper** (Medium complexity)

If results are concerning (>20% variance or regression):
→ Analyze root causes before continuing

If results are exceptional (exceeds predictions by >10%):
→ Proceed with high confidence, document success factors

---

**Pilot 1 Status**: COMPLETE ✓
**Time Invested**: _____ hours
**Ready for Pilot 2**: YES / NO
