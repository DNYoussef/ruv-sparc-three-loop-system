# Prompt-Architect Analysis: source-credibility-analyzer v1

**Analyzed**: 2025-01-06
**Framework**: Evidence-Based Prompting (Self-Consistency, Program-of-Thought, Plan-and-Solve, Few-Shot)
**Target**: Standalone tool SOP for source evaluation

---

## Executive Summary

**Overall Grade**: B+ (Strong foundation, missing edge case handling and calibration examples)

**Strengths**:
- ✅ Excellent program-of-thought rubrics with explicit calculations
- ✅ Clear sequential workflow with Quality Gates
- ✅ Comprehensive scoring dimensions (credibility, bias, priority)
- ✅ Structured JSON output for machine readability
- ✅ Integration points documented

**Critical Gaps** (Priority 1):
1. No few-shot examples showing complete scoring calculations (good vs bad)
2. Missing edge case handling for ambiguous sources (preprints, gray literature, Wikipedia)
3. No calibration guidance for adjusting rubrics to domain-specific needs
4. Insufficient failure mode coverage (conflicting scores, missing metadata)
5. No visual markers (✅/⚠️) for required vs optional steps

**Medium Gaps** (Priority 2):
6. No self-consistency check (verify scoring reproducibility)
7. Missing cross-validation against manual scores
8. No batch processing guidance (multiple sources at once)

**Minor Gaps** (Priority 3):
9. No citation extraction automation
10. Missing confidence intervals for borderline scores

---

## Detailed Analysis

### 1. Intent and Clarity ✅ STRONG

**What Works**:
- Purpose crystal clear: "Automate credibility/bias/priority scoring"
- Target agent explicit: "analyst"
- Time expectations realistic: "5-15 minutes per source"
- Integration context provided (standalone OR within general-research-workflow)

**Gaps**:
- ❌ No explanation of WHEN to use this tool vs manual scoring
- ❌ Doesn't clarify if tool can score non-academic sources (podcasts, tweets, etc.)

**Recommendation**:
- Add "When to Use This Tool" section with decision tree
- Specify supported source types explicitly

---

### 2. Structural Organization ✅ STRONG

**What Works**:
- Sequential 5-step workflow (0-4) with clear objectives
- Quality Gates after each scoring step
- Deliverables well-defined
- Error handling table comprehensive

**Gaps**:
- ⚠️ No visual markers (✅ Required vs ⚠️ Optional) for metadata fields
- ⚠️ Steps 1-3 very similar structure (could use template pattern)

**Recommendation**:
- Add visual markers to metadata requirements (title ✅ required, citations ⚠️ optional)
- Create scoring template that applies to all three dimensions

---

### 3. Context Sufficiency ⚠️ MODERATE

**What Works**:
- Program-of-thought rubrics provide explicit calculation logic
- Example calculations show step-by-step reasoning
- Integration section explains where tool fits

**Critical Gaps**:
1. **NO FEW-SHOT EXAMPLES**: Missing complete end-to-end scoring examples
   - Need 3-5 examples showing: Input → Calculation → Output → Recommendation
   - Should include edge cases (preprints, Wikipedia, blogs)
   - Need "good vs bad" examples (correct vs incorrect scoring)

2. **NO EDGE CASE GUIDANCE**:
   - Preprints (arXiv, bioRxiv): High credibility but not peer-reviewed → how to score?
   - Wikipedia: Crowdsourced, verifiable, but not original research → credibility?
   - Gray literature (government reports, white papers): Variable quality → rubric?
   - Conference papers vs journal papers: Different credibility levels?
   - Self-published by experts (Gwern, Scott Alexander): No formal review but high-quality → score?

3. **NO CALIBRATION GUIDANCE**:
   - Rubrics may need adjustment for different fields (STEM vs humanities)
   - Citation counts vary by field (CS: 100+ is high, Math: 10+ is high)
   - Recency matters differently (AI: 5 years, History: 20 years)

**Recommendation**:
- Priority 1: Add 5 complete examples (academic paper, book, website, preprint, Wikipedia)
- Priority 1: Add edge case decision tree for ambiguous sources
- Priority 2: Add domain-specific calibration guidance

---

### 4. Evidence-Based Prompting Techniques ✅ STRONG

**Program-of-Thought**: ✅ Excellent
- Explicit calculations with start/add/subtract logic
- Transparent reasoning for each score
- Auditable and reproducible

**Plan-and-Solve**: ✅ Good
- Sequential workflow with clear sub-steps
- Quality Gates enforce correctness

**Self-Consistency**: ❌ MISSING
- No mechanism to verify scoring reproducibility
- No guidance on re-scoring to check consistency
- Borderline scores (e.g., 2.5 → round to 2 or 3?) unhandled

**Few-Shot Learning**: ❌ CRITICAL GAP
- Zero complete examples showing input → output
- Example calculations embedded but not standalone
- No "good vs bad" scoring comparisons

**Recommendation**:
- Priority 1: Add 5 complete few-shot examples
- Priority 2: Add self-consistency check (re-score ambiguous sources, compare results)

---

### 5. Failure Modes ⚠️ MODERATE

**Covered Well**:
- Missing metadata
- Invalid scores
- Storage failures
- Ambiguous source types

**Critical Gaps**:
1. **CONFLICTING SCORES**: High credibility + high bias (peer-reviewed but advocacy journal)
   - Example: Lancet paper funded by pharma → Credibility 5, Bias 2 → How to recommend?
   - Need tie-breaking logic

2. **BORDERLINE SCORES**: Score = 2.5 after calculation
   - Round up or down? Conservative or liberal?
   - Need rounding policy

3. **CONTRADICTORY METADATA**: Title says "peer-reviewed" but venue is blog
   - Metadata conflicts → which to trust?
   - Need verification step

4. **BATCH CONFLICTS**: Scoring 50 sources, run out of time
   - Partial results → save and resume?
   - Need checkpoint/resume mechanism

5. **SCORE DRIFT**: Same source scored differently on different days
   - Agent interpretation varies
   - Need calibration examples to anchor judgments

**Recommendation**:
- Priority 1: Add conflicting score resolution logic (credibility vs bias trade-offs)
- Priority 1: Add borderline score rounding policy
- Priority 2: Add metadata verification step
- Priority 3: Add batch checkpoint/resume

---

### 6. Formatting and Accessibility ⚠️ MODERATE

**What Works**:
- Clear section headers
- Tables for workflow steps
- Code blocks for examples
- Error handling table

**Gaps**:
- ⚠️ No visual markers (✅/⚠️/💡/🚨)
- ⚠️ No progressive disclosure (basic vs advanced rubrics)
- ⚠️ Rubrics are long blocks of text (could use tables)

**Recommendation**:
- Priority 1: Add visual markers for required (✅) vs optional (⚠️) metadata
- Priority 2: Convert rubric text to tables for scanability
- Priority 3: Add progressive disclosure (basic 3-rule rubric, advanced 6-rule rubric)

---

## Missing Components

### Critical (Priority 1)

**1. Complete Few-Shot Examples** (HIGHEST PRIORITY)
- **Impact**: Without examples, agents may score inconsistently
- **Solution**: Add 5 complete examples covering:
  1. ✅ Academic paper (Credibility 5, Bias 5, Priority 4) → READ_FIRST
  2. ✅ Think tank report (Credibility 3, Bias 1, Priority 3) → VERIFY_CLAIMS
  3. ⚠️ Preprint (Credibility 3, Bias 4, Priority 5) → READ_FIRST but verify
  4. ⚠️ Wikipedia article (Credibility 3, Bias 4, Priority 2) → Background only
  5. ❌ Blog post (Credibility 2, Bias 2, Priority 1) → SKIP

**2. Edge Case Decision Tree**
- **Impact**: Agents stuck on ambiguous sources (preprints, Wikipedia, gray literature)
- **Solution**: Add flowchart:
  ```
  Is it peer-reviewed?
  ├─ Yes → Academic rubric
  ├─ No → Is it from recognized institution?
      ├─ Yes → Institutional rubric
      └─ No → General rubric
  ```

**3. Conflicting Score Resolution**
- **Impact**: High credibility + high bias → Unclear recommendation
- **Solution**: Add tie-breaking logic:
  - If Credibility ≥4 AND Bias ≤2 → VERIFY_CLAIMS (useful but biased)
  - If Credibility ≤2 AND Bias ≥4 → SKIP (not credible enough despite low bias)

**4. Visual Markers**
- **Impact**: Agents unsure what's required vs optional
- **Solution**: Add ✅ Required, ⚠️ Optional, 💡 Tip, 🚨 Warning throughout

**5. Borderline Score Policy**
- **Impact**: Score = 2.5 or 3.5 → ambiguous rounding
- **Solution**: Conservative rounding (round down for credibility, round up for bias/priority)

### Medium (Priority 2)

**6. Self-Consistency Check**
- Add step 3.5: "Re-score if total score variance >1 point, investigate discrepancy"

**7. Domain-Specific Calibration**
- Add appendix with citation thresholds by field (CS, Math, History, etc.)

**8. Batch Processing Guidance**
- Add section on scoring multiple sources efficiently

### Minor (Priority 3)

**9. Citation Extraction Automation**
- Link to tools (Google Scholar API, Semantic Scholar) for fetching citation counts

**10. Confidence Intervals**
- For borderline scores (2.5-3.5), report as "3±0.5" to indicate uncertainty

---

## Grading Rubric

| Dimension | Score | Max | Notes |
|-----------|-------|-----|-------|
| **Intent Clarity** | 9 | 10 | Clear purpose, minor gaps on edge cases |
| **Structural Organization** | 9 | 10 | Sequential workflow excellent, missing visual markers |
| **Context Sufficiency** | 6 | 10 | NO few-shot examples, NO edge case guidance |
| **Evidence-Based Techniques** | 7 | 10 | Strong program-of-thought, missing self-consistency and few-shot |
| **Failure Mode Coverage** | 6 | 10 | Basics covered, missing conflicting scores and borderline cases |
| **Formatting** | 7 | 10 | Clean structure, missing visual markers and tables |
| **Overall** | 44 | 60 | **73% = B+** |

---

## Recommended Improvements

### Priority 1 (CRITICAL - Implement Before v2)

1. **Add 5 Complete Few-Shot Examples** (examples/scoring-examples.md)
   - Academic paper (ideal score)
   - Think tank report (high bias)
   - Preprint (ambiguous credibility)
   - Wikipedia (edge case)
   - Blog post (low score)
   - Each example: Input → Calculation → Output → Explanation

2. **Edge Case Decision Tree** (SKILL.md Step 0.5)
   - Flowchart for classifying ambiguous sources
   - Rubric adjustments for preprints, Wikipedia, gray literature

3. **Conflicting Score Resolution** (SKILL.md Step 4)
   - Tie-breaking logic for High Credibility + High Bias
   - Trade-off matrix (credibility vs bias)

4. **Visual Markers Throughout** (SKILL.md all steps)
   - ✅ Required metadata (title, author, year, venue, type)
   - ⚠️ Optional metadata (citations, DOI, institution)
   - 💡 Tips (conservative rounding for borderline scores)
   - 🚨 Warnings (verify claims if bias <3)

5. **Borderline Score Policy** (SKILL.md Steps 1-3)
   - Explicit rounding rules (round down credibility, round up bias)

### Priority 2 (HIGH - Add if Time Permits)

6. **Self-Consistency Check** (SKILL.md Step 3.5)
   - Re-score sources with variance >1 point

7. **Domain-Specific Calibration** (references/calibration-guide.md)
   - Citation thresholds by field
   - Recency thresholds by discipline

8. **Batch Processing** (SKILL.md Section)
   - Checkpoint/resume mechanism
   - Parallel scoring guidance

### Priority 3 (NICE-TO-HAVE - Future Enhancement)

9. **Citation Extraction Tools** (references/automation-tools.md)
10. **Confidence Intervals** (Output format enhancement)

---

## Implementation Plan

**For v2 Optimized SOP**:
1. Implement ALL Priority 1 improvements (5 items)
2. Add Priority 2 if implementing quickly (est. +30 min)
3. Defer Priority 3 to future versions

**Estimated Improvement Impact**:
- Grade: B+ (73%) → A (88%) with Priority 1
- Grade: A (88%) → A+ (95%) with Priority 1+2

**Critical Path**: Few-shot examples (30% of impact) → Edge cases (25%) → Visual markers (20%) → Conflicting scores (15%) → Borderline policy (10%)

---

**Next**: Implement Priority 1 improvements in v2 SOP
