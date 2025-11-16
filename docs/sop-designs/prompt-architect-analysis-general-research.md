# Prompt Architect Analysis: General Research Workflow SOP

## Analysis Date
2025-01-06

## Artifact Analyzed
`general-research-workflow-v2-sop.md` - SOP for sequential agent workflow implementing Red's 6-phase research methodology

---

## 1. Intent and Clarity Assessment

### Strengths ✅
- Core objective is crystal clear: "Systematic methodology for general-purpose research"
- Success criteria explicitly stated (quantitative + qualitative metrics)
- Each step has clear objective and deliverable
- Agent roles are unambiguous

### Areas for Improvement ⚠️
- **Issue**: "When to Invoke" section lists trigger conditions but could be more precise about decision boundaries
- **Recommendation**: Add decision tree or flowchart showing "Use this SOP when X AND Y, but NOT when Z"

---

## 2. Structural Organization Analysis

### Strengths ✅
- Excellent hierarchical structure (6 steps → sub-phases → instructions)
- Quality Gates positioned at logical checkpoints
- Clear progression: Discovery → Validation → Analysis → Synthesis
- Table of contents via agent sequence table provides quick reference

### Areas for Improvement ⚠️
- **Issue**: Step 6 has 4 sub-phases (A-D) which is more complex than other steps
- **Recommendation**: Consider breaking Step 6 into two separate steps (Step 6: Pattern Recognition, Step 7: Synthesis & Validation) for consistency

---

## 3. Context Sufficiency Evaluation

### Strengths ✅
- Prerequisites clearly stated (research question defined, access to sources)
- Integration points show relationship to other skills
- Red's principles table explicitly connects theory to practice
- Duration estimates help with planning

### Areas for Improvement ⚠️
- **Issue**: Assumes agents understand "WorldCat", "Google Scholar", "JSTOR" without definition
- **Issue**: "Memory MCP tagging protocol" referenced but not explained
- **Recommendation**: Add brief glossary or inline definitions for technical terms
- **Recommendation**: Link to or embed Memory MCP tagging requirements

---

## 4. Technique Application Review

### Strengths ✅
- **Self-Consistency**: Step 6 Phase C includes validation by analyst agent (multiple perspectives)
- **Plan-and-Solve**: Clear separation of planning (Steps 1-4) from execution (Step 5) and verification (Step 6)
- **Evidence-Based**: Entire SOP built on Red's research-backed methodology

### Areas for Improvement ⚠️
- **Missing Technique**: Few-shot examples
- **Recommendation**: Add 1-2 concrete examples showing:
  - Sample citation extraction (Step 1)
  - Sample source classification with scores (Step 3)
  - Sample final report (Step 6)
- **Rationale**: Examples help agents understand expected output format without ambiguity

---

## 5. Failure Mode Detection

### Strengths ✅
- Quality Gates catch common failures (insufficient sources, low credibility)
- Error Handling section addresses gate failures explicitly
- "If source access blocked" contingency plan included

### Areas for Improvement ⚠️
- **Missing Failure Mode 1**: What if Wikipedia has NO article on topic?
- **Missing Failure Mode 2**: What if ALL sources are secondary (no primary sources exist)?
- **Missing Failure Mode 3**: What if sources are in non-English languages without translations?
- **Missing Failure Mode 4**: What if thesis cannot be formed because evidence is contradictory?

**Recommendations**:
1. Add "Gate 0" before Step 1: Verify Wikipedia article exists, else use alternative starting point
2. Add exception to Gate 3: If topic has no primary sources (e.g., ancient history), document this and proceed with ≥10 high-credibility secondary sources
3. Add language accessibility check in Step 2, flag non-English sources for translation or exclusion
4. Add Step 6 failure mode: If evidence too contradictory, output "Insufficient evidence for conclusive thesis" with summary of competing interpretations

---

## 6. Formatting and Accessibility

### Strengths ✅
- Excellent use of tables for quick reference
- Clear delimiters (--- for sections)
- Template examples use code blocks for visual clarity
- Consistent numbering and hierarchy

### Areas for Improvement ⚠️
- **Issue**: Some instruction blocks are dense (Step 5 template has 7 sub-sections)
- **Recommendation**: Add visual markers:
  - ✅ for required elements
  - ⚠️ for conditional elements
  - 💡 for tips/best practices

Example improvement:
```
KEY CLAIMS:
✅ Claim 1 (page X): "[exact quote or paraphrase]" [REQUIRED]
✅ Claim 2 (page Y): "[exact quote or paraphrase]" [REQUIRED]
⚠️ Claim 3 (if applicable): ...
```

---

## 7. Evidence-Based Technique Integration

### Current Techniques Used:
1. ✅ **Plan-and-Solve**: Steps 1-4 (plan) → Step 5 (execute) → Step 6 (solve + verify)
2. ✅ **Self-Consistency**: Step 6 Phase C analyst validates researcher's synthesis
3. ✅ **Chain-of-Thought**: Implicit in Step 5 note-taking structure (claims → evidence → contradictions)

### Missing Techniques That Would Improve SOP:

**A. Few-Shot Examples** (High Priority)
- **Where**: After Step 1, Step 3, Step 6
- **Why**: Agents need concrete examples of expected output format
- **How**: Add appendix with sample outputs:
  - Appendix A: Sample Wikipedia citation extraction (Step 1)
  - Appendix B: Sample source classification table (Step 3)
  - Appendix C: Sample final research report (Step 6)

**B. Program-of-Thought** (Medium Priority)
- **Where**: Step 3 credibility evaluation
- **Why**: Scoring can be subjective without explicit rubric
- **How**: Add scoring rubric with decision tree:
  ```
  Credibility Score Calculation:
  Start at 3 (neutral)
  +1 if peer-reviewed publication
  +1 if author has PhD/expert credentials
  -1 if blog or non-vetted source
  -1 if conflicts of interest detected
  Final score: [1-5]
  ```

---

## 8. Anti-Pattern Check

### Detected Anti-Patterns:

**1. Vague Instructions** ⚠️
- **Location**: Step 2 - "Brief relevance note (1 sentence: why this matters)"
- **Issue**: "Why this matters" is subjective and vague
- **Fix**: Replace with structured relevance criteria:
  - "Relevance Note: How does this source address [research question]? Does it provide primary evidence, expert analysis, or contextual background?"

**2. Over-Complexity** ⚠️
- **Location**: Step 5 note template has 7 sections
- **Issue**: Cognitive load may lead to incomplete notes
- **Fix**: Mark sections as REQUIRED vs OPTIONAL:
  - REQUIRED: Key Claims, Supporting Evidence, Quotable Passages
  - OPTIONAL: Contradictions (if applicable), Bias/Agenda (if detected), Cross-References (if relevant)

**3. Insufficient Edge Case Handling** (see Failure Mode section above)

### No Detected Anti-Patterns ✅:
- No contradictory requirements
- No assuming shared understanding (context is explicit)
- No cognitive biases in language

---

## 9. Task-Category Specific Optimization

**Task Category**: Analytical Research Workflow

**Recommended Optimizations**:

1. **Uncertainty Handling**:
   - Add explicit instruction in Step 6 Phase B: "If evidence is insufficient or contradictory, state 'Inconclusive' and explain why rather than forcing a thesis"

2. **Source Diversity Metrics**:
   - Add to Quality Gate 2: "Sources must represent ≥2 different geographic/cultural perspectives if topic spans multiple regions"

3. **Temporal Coverage**:
   - Add to Quality Gate 4: "If topic spans >50 years, sources must cover ≥3 distinct time periods"

---

## 10. Agent-Specific Considerations

**Agent Type Alignment**:
- ✅ `researcher`: Correctly assigned to discovery, analysis, note-taking tasks
- ✅ `analyst`: Correctly assigned to validation, classification, quality checks
- ✅ `coordinator`: Correctly assigned to synthesis and orchestration

**Potential Improvement**:
- Consider adding `optimizer` agent for Step 2 Source Discovery
- **Rationale**: Source discovery could benefit from optimization heuristics (prioritize accessible sources, optimize search queries)

---

## Optimized SOP Recommendations

### Priority 1: CRITICAL (Implement Immediately)

1. **Add Few-Shot Examples** (Appendices A, B, C)
   - Concrete citation extraction example
   - Concrete source classification with scores
   - Concrete final research report

2. **Add Missing Failure Modes**:
   - Gate 0: Wikipedia article existence check
   - Exception for topics with no primary sources
   - Language accessibility handling
   - Inconclusive thesis handling

3. **Add Program-of-Thought Scoring Rubric**:
   - Step 3: Explicit credibility/bias scoring calculation

### Priority 2: HIGH (Implement Soon)

4. **Restructure Step 6**:
   - Split into Step 6 (Pattern Recognition) and Step 7 (Synthesis & Validation)
   - Reduces complexity per step

5. **Add Visual Markers**:
   - ✅ Required elements
   - ⚠️ Conditional elements
   - 💡 Tips/best practices

6. **Add Glossary**:
   - Define: WorldCat, Google Scholar, primary vs secondary sources, historiography
   - Embed Memory MCP tagging requirements

### Priority 3: MEDIUM (Nice to Have)

7. **Add Source Diversity Metrics**:
   - Geographic/cultural perspective requirements
   - Temporal coverage requirements

8. **Add Uncertainty Language**:
   - Explicit "Inconclusive" option in Step 6

9. **Simplify Step 5 Template**:
   - Mark sections as REQUIRED vs OPTIONAL

---

## Overall Assessment

**Strengths**:
- Exceptionally clear structure and agent coordination
- Strong quality gates with explicit criteria
- Embeds research best practices (Red's methodology)
- Good error handling for common failures

**Weaknesses**:
- Missing few-shot examples (agents may misinterpret output format)
- Some edge cases not covered (no Wikipedia article, no primary sources, language barriers)
- Step 6 is more complex than other steps (should consider splitting)

**Grade**: B+ (Very Good, with room for targeted improvements)

**Recommendation**: Implement Priority 1 changes before building skill with skill-forge. SOP will be production-ready after those additions.

---

## Next Actions

1. Create few-shot examples (Appendices A, B, C)
2. Add failure modes and error handling
3. Add scoring rubric for Step 3
4. Consider splitting Step 6 into two steps
5. Once optimized → Build with `skill-forge`
