# Test 1: Basic Research (Level 1)

## Test Objective
Validate Level 1 (Basic Research) workflow with single-source validation and quick factual queries.

## Test Scenario
**Query**: "What is the current learning rate for GPT-4 training?"
**Expected Duration**: < 5 minutes
**Credibility Threshold**: > 70%

## Test Steps

### 1. Execute Basic Research
```bash
# Trigger researcher skill with Level 1 parameters
# This should use Gemini Search directly without heavy orchestration
```

**Expected Behavior**:
- Single Gemini Search query
- Quick factual answer extraction
- Source credibility validation (>70%)
- Concise summary delivery

### 2. Verify Output Quality

**Required Elements**:
- [x] Clear, direct answer to the query
- [x] Source citation with URL
- [x] Credibility score displayed
- [x] Completion time < 5 minutes

**Sample Expected Output**:
```markdown
## Answer
GPT-4 training uses adaptive learning rate scheduling with peak rates around 6e-4,
gradually decaying over the training period.

**Source**: OpenAI Technical Report (2023)
**Credibility**: 95% (Tier 1: Official documentation)
**URL**: https://openai.com/research/gpt-4

**Completion Time**: 2 minutes
```

### 3. Edge Cases

**Test 3.1**: Ambiguous Query
- Query: "What is GPT performance?"
- Expected: Request clarification or provide multiple interpretations

**Test 3.2**: No Credible Sources
- Query: "What will AGI look like in 2025?" (speculative)
- Expected: Flag low credibility, present with caveats

**Test 3.3**: Outdated Information
- Expected: Highlight publication dates, suggest newer sources if available

## Success Criteria

### Functional Requirements
- [ ] Query answered within 5 minutes
- [ ] Source credibility >= 70%
- [ ] Valid citation with URL
- [ ] Clear, concise summary (< 200 words)

### Quality Requirements
- [ ] Factual accuracy verified
- [ ] No hallucinations or unsupported claims
- [ ] Appropriate confidence level indicated
- [ ] Source tier correctly identified (Tier 1/2/3)

### Error Handling
- [ ] Graceful handling of no results
- [ ] Clear messaging for ambiguous queries
- [ ] Appropriate warnings for low credibility

## Test Results

**Date**: _____________________
**Tester**: _____________________

### Execution Results
- Query execution time: _____ minutes
- Source credibility: _____ %
- Source tier: _____
- Output word count: _____ words

### Pass/Fail Assessment
- [ ] PASS - All criteria met
- [ ] CONDITIONAL PASS - Minor issues (document below)
- [ ] FAIL - Critical issues (document below)

**Issues Found**:
```
(List any issues encountered)
```

**Recommendations**:
```
(Suggest improvements)
```

## Notes
- Level 1 should NOT trigger heavy scripts (research-orchestrator.py)
- Gemini Search should be primary tool
- Focus on speed and directness over comprehensiveness
- Credibility scoring should be automatic
