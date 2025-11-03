# Test 2: Multi-Source Research (Level 2)

## Test Objective
Validate Level 2 (Multi-Source Research) workflow with cross-reference analysis and comprehensive coverage.

## Test Scenario
**Query**: "What are the best practices for fine-tuning large language models?"
**Expected Duration**: 15-30 minutes
**Credibility Threshold**: > 85%
**Target Sources**: 3-5 sources

## Test Steps

### 1. Execute Multi-Source Research
```bash
# Should trigger research-orchestrator.py for parallel aggregation
python resources/scripts/research-orchestrator.py \
  --query "fine-tuning large language models best practices" \
  --sources 5 \
  --min-credibility 85 \
  --output test-multi-source-report.md \
  --json
```

**Expected Behavior**:
- Parallel API queries (Gemini Search, Semantic Scholar, ArXiv)
- 3-5 high-credibility sources aggregated
- Cross-reference comparison
- Consensus identification
- Markdown report generation

### 2. Verify Source Quality

**Required Checks**:
- [x] Minimum 3 sources retrieved
- [x] Average credibility >= 85%
- [x] Mix of source types (academic, documentation, industry)
- [x] Recent publications (within 3 years preferred)
- [x] No duplicate sources

**Sample Expected Sources**:
```json
{
  "sources": [
    {
      "title": "Scaling Instruction-Finetuned Language Models",
      "credibility_score": 95.0,
      "source_type": "academic",
      "publication_date": "2023",
      "citations": 450
    },
    {
      "title": "Fine-tuning Best Practices | OpenAI Docs",
      "credibility_score": 92.0,
      "source_type": "documentation",
      "publication_date": "2024"
    },
    {
      "title": "PEFT: Parameter-Efficient Fine-Tuning",
      "credibility_score": 88.0,
      "source_type": "academic",
      "publication_date": "2023",
      "citations": 230
    }
  ]
}
```

### 3. Verify Synthesis Quality

**Required Elements**:
- [x] Comparison of methodologies across sources
- [x] Consensus statements identified
- [x] Disagreements or contradictions noted
- [x] Coherent synthesis narrative
- [x] Evidence-based conclusions

**Sample Expected Synthesis**:
```markdown
## Consensus Findings

All 5 sources agree on these best practices:
1. **Use Low-Rank Adaptation (LoRA)** for efficiency (Sources: 1, 2, 3, 5)
2. **Start with small learning rates** (1e-5 to 3e-5) (Sources: 1, 2, 4)
3. **Monitor validation loss closely** to prevent overfitting (All sources)

## Divergent Opinions

- **Dataset Size**: Source 1 recommends 10k+ examples, Source 3 shows success with 1k
- **Epochs**: Range from 3 (Source 2) to 10 (Source 4)
```

### 4. Edge Cases

**Test 4.1**: Contradicting Sources
- Expected: Clear documentation of contradictions
- Expected: Evidence-based resolution or "further investigation needed"

**Test 4.2**: Insufficient High-Credibility Sources
- Scenario: Only 2 sources > 85% credibility found
- Expected: Lower threshold with warning or expand search

**Test 4.3**: Very Recent Topic (< 6 months old)
- Expected: Acknowledge limited research, include preprints with caveats

## Success Criteria

### Functional Requirements
- [ ] 3-5 sources retrieved and analyzed
- [ ] Average credibility >= 85%
- [ ] Execution time 15-30 minutes
- [ ] Markdown + JSON outputs generated
- [ ] No API errors or timeouts

### Quality Requirements
- [ ] Cross-source comparison present
- [ ] Consensus clearly identified
- [ ] Contradictions documented
- [ ] Synthesis is coherent and evidence-based
- [ ] Bibliography formatted correctly

### Content Requirements
- [ ] Best practices identified (at least 5)
- [ ] Evidence for each practice (source citations)
- [ ] Practical recommendations
- [ ] Confidence levels indicated

## Test Results

**Date**: _____________________
**Tester**: _____________________

### Execution Metrics
- Total execution time: _____ minutes
- Sources retrieved: _____
- Average credibility: _____ %
- Sources by tier:
  - Tier 1: _____
  - Tier 2: _____
  - Tier 3: _____

### Output Files
- [ ] Markdown report generated: `test-multi-source-report.md`
- [ ] JSON data generated: `test-multi-source-report.json`
- [ ] Log file present: `research-orchestrator.log`

### Quality Assessment
**Consensus Findings**:
- Number of consensus points: _____
- Evidence strength: _____ (strong/moderate/weak)

**Contradictions**:
- Number of contradictions found: _____
- Resolution approach: _____ (evidence-based/flagged for investigation)

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
(Suggest improvements to the workflow)
```

## Script Integration Test

### research-orchestrator.py Validation
- [ ] Script executed without errors
- [ ] Parallel API calls successful
- [ ] Credibility scoring accurate
- [ ] Deduplication working correctly
- [ ] Output formatting correct

**Command Used**:
```bash
(Paste actual command executed)
```

**Sample Output**:
```
(Paste relevant log output)
```

## Notes
- Level 2 should use research-orchestrator.py for efficiency
- Focus on cross-reference synthesis, not just aggregation
- Credibility threshold enforcement is critical
- Parallel execution should be faster than sequential
