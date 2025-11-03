# Test 2: Multi-Intent Request Handling

## Test Overview
Validates Intent Analyzer's capability to recognize and appropriately handle requests that genuinely contain multiple distinct intents. Tests ability to identify primary intent, acknowledge secondary intents, and structure responses that address all facets of multi-dimensional requests.

## Test Scenarios

### Scenario 1: Research + Implementation
**User Request**: "Research best practices for microservices authentication and implement JWT-based auth for our Node.js API"

**Expected Analysis**:
```json
{
  "primary_category": "technical",
  "confidence": 0.60,
  "categories": {
    "technical": 0.60,
    "analytical": 0.35,
    "learning": 0.05
  },
  "multi_intent": true,
  "ambiguous": false,
  "requires_clarification": false,
  "reasoning": "Multiple intent categories detected: technical, analytical. Moderate confidence (0.60) in technical intent - proceed with assumption acknowledgment. Request contains both research (analytical) and implementation (technical) components. Key signals: research (verb), best practices (phrase), implement (verb)"
}
```

**Expected Pattern Extraction**:
```json
{
  "constraints": [
    {
      "pattern": "technology_stack",
      "signal": "technology_constraint",
      "matches": [
        {"matched_text": "JWT", "position": 70},
        {"matched_text": "Node.js", "position": 93}
      ]
    }
  ],
  "expertise": [
    {
      "pattern": "expert_terminology",
      "signal": "expert_user",
      "matches": [
        {"matched_text": "microservices", "position": 23},
        {"matched_text": "JWT", "position": 70}
      ]
    }
  ]
}
```

**Expected Response Structure**:
1. **Phase 1 - Research**: Best practices analysis for microservices auth
2. **Phase 2 - Implementation**: JWT implementation for Node.js
3. Both phases addressed in coherent workflow

**Pass Criteria**:
- ✅ Recognizes dual intent (research + implementation)
- ✅ Technical as primary, analytical as significant secondary
- ✅ Response structure addresses both intents
- ✅ No clarification needed (both intents are clear)

---

### Scenario 2: Learning + Decision Making
**User Request**: "Explain the difference between SQL and NoSQL databases and help me decide which to use for a social media app"

**Expected Analysis**:
```json
{
  "primary_category": "learning",
  "confidence": 0.50,
  "categories": {
    "learning": 0.50,
    "decision": 0.45,
    "analytical": 0.05
  },
  "multi_intent": true,
  "ambiguous": false,
  "requires_clarification": false,
  "reasoning": "Multiple intent categories detected: learning, decision. Equal confidence (0.50 vs 0.45) - both intents equally important. Key signals: explain (verb), difference (noun), help me decide (phrase), which to use (phrase)"
}
```

**Expected Pattern Extraction**:
```json
{
  "constraints": [
    {
      "pattern": "explicit_tech",
      "signal": "technology_constraint",
      "matches": [
        {"matched_text": "SQL", "position": 26},
        {"matched_text": "NoSQL", "position": 34}
      ]
    }
  ],
  "meta": [],
  "expertise": []
}
```

**Expected Response Structure**:
1. **Educational Component**: SQL vs NoSQL comparison
2. **Decision Framework**: How to choose between them
3. **Specific Recommendation**: For social media app use case

**Pass Criteria**:
- ✅ Detects near-equal probability for learning + decision
- ✅ Response balances education with actionable recommendation
- ✅ Tailors decision advice to specific use case (social media)
- ✅ Proceeds without clarification (both intents clear)

---

### Scenario 3: Creation + Validation
**User Request**: "Write a Python function to validate email addresses and tell me if my implementation approach is correct"

**Expected Analysis**:
```json
{
  "primary_category": "technical",
  "confidence": 0.65,
  "categories": {
    "technical": 0.65,
    "analytical": 0.30,
    "learning": 0.05
  },
  "multi_intent": true,
  "ambiguous": false,
  "requires_clarification": false,
  "reasoning": "Multiple intent categories detected: technical, analytical. Moderate confidence (0.65) in technical intent - proceed with assumption acknowledgment. Request contains creation (technical) and validation (analytical) components. Key signals: write (verb), function (noun), validate (verb), tell me if (phrase)"
}
```

**Expected Clarification Questions**:
Despite no overall clarification required, should ask:
1. "Do you have an existing implementation you'd like me to review, or should I write one and explain the approach?"

This disambiguates whether user has code to review or wants new code + explanation.

**Expected Response Structure**:
**If user has implementation**:
1. Review their approach
2. Provide corrected/improved version
3. Explain validation best practices

**If user wants new implementation**:
1. Write function with best practices
2. Explain approach and design decisions
3. Discuss alternative approaches

**Pass Criteria**:
- ✅ Detects creation + validation dual intent
- ✅ Asks clarifying question to determine which comes first
- ✅ Adapts response based on clarification
- ✅ Moderate confidence appropriately triggers assumption acknowledgment

---

### Scenario 4: Teaching + Problem Solving
**User Request**: "I'm getting a 'CORS error' in my React app. Explain what CORS is and fix my issue"

**Expected Analysis**:
```json
{
  "primary_category": "problem_solving",
  "confidence": 0.55,
  "categories": {
    "problem_solving": 0.55,
    "learning": 0.40,
    "technical": 0.05
  },
  "multi_intent": true,
  "ambiguous": false,
  "requires_clarification": false,
  "reasoning": "Multiple intent categories detected: problem_solving, learning. Moderate confidence (0.55) in problem_solving intent - proceed with assumption acknowledgment. User has active problem but also wants to understand. Key signals: getting error (phrase), explain (verb), fix (verb)"
}
```

**Expected Pattern Extraction**:
```json
{
  "temporal": [
    {
      "pattern": "high_urgency",
      "signal": "high_urgency",
      "interpretation": "Implied urgency from active error/problem"
    }
  ],
  "constraints": [
    {
      "pattern": "technology_stack",
      "signal": "technology_constraint",
      "matches": [{"matched_text": "React", "position": 31}]
    }
  ],
  "meta": [],
  "expertise": []
}
```

**Expected Response Structure**:
1. **Quick Fix First**: Immediate solution to resolve error
2. **Explanation**: What CORS is and why error occurred
3. **Best Practices**: How to handle CORS properly going forward

**Pass Criteria**:
- ✅ Prioritizes problem solving (active error) over learning
- ✅ Includes educational component (CORS explanation)
- ✅ Solution-first structure (fix then explain)
- ✅ Recognizes implied urgency from error condition

---

### Scenario 5: Analysis + Planning + Implementation
**User Request**: "Evaluate different state management solutions for React, recommend the best one for our e-commerce app, and show me how to set it up"

**Expected Analysis**:
```json
{
  "primary_category": "analytical",
  "confidence": 0.45,
  "categories": {
    "analytical": 0.45,
    "decision": 0.35,
    "technical": 0.20
  },
  "multi_intent": true,
  "ambiguous": false,
  "requires_clarification": false,
  "reasoning": "Multiple intent categories detected: analytical, decision, technical. Low-moderate confidence (0.45) across three distinct phases. Key signals: evaluate (verb), recommend (verb), show me (phrase). Three-phase request: analysis → decision → implementation"
}
```

**Expected Pattern Extraction**:
```json
{
  "constraints": [
    {
      "pattern": "technology_stack",
      "signal": "technology_constraint",
      "matches": [{"matched_text": "React", "position": 46}]
    }
  ],
  "expertise": [
    {
      "pattern": "specific_question",
      "signal": "high_specificity",
      "matches": [{"matched_text": "state management solutions", "position": 19}]
    }
  ]
}
```

**Expected Response Structure**:
1. **Phase 1 - Analysis**: Compare Redux, MobX, Zustand, Context API, Recoil
2. **Phase 2 - Recommendation**: Best choice for e-commerce (likely Redux Toolkit or Zustand)
3. **Phase 3 - Setup Guide**: Implementation walkthrough with code examples

**Pass Criteria**:
- ✅ Recognizes three distinct phases (analyze, decide, implement)
- ✅ Structures response as coherent workflow
- ✅ Each phase builds on previous (comparison → recommendation → implementation)
- ✅ Tailors recommendation to specific use case (e-commerce)

---

### Scenario 6: Overlapping Intents - Creative + Analytical
**User Request**: "Generate 5 blog post ideas about AI and rank them by potential engagement"

**Expected Analysis**:
```json
{
  "primary_category": "creative",
  "confidence": 0.55,
  "categories": {
    "creative": 0.55,
    "analytical": 0.40,
    "decision": 0.05
  },
  "multi_intent": true,
  "ambiguous": false,
  "requires_clarification": false,
  "reasoning": "Multiple intent categories detected: creative, analytical. Moderate confidence (0.55) in creative intent - proceed with assumption acknowledgment. Request requires both generation (creative) and evaluation (analytical). Key signals: generate (verb), ideas (noun), rank (verb)"
}
```

**Expected Pattern Extraction**:
```json
{
  "constraints": [
    {
      "pattern": "explicit_format",
      "signal": "format_constraint",
      "matches": [{"matched_text": "5 blog post ideas", "position": 9}]
    }
  ],
  "audience": [],
  "meta": []
}
```

**Expected Response Structure**:
1. **Generate 5 Ideas**: Creative ideation phase
2. **Rank by Engagement**: Analytical evaluation with criteria
3. **Justification**: Why each ranking was assigned

**Pass Criteria**:
- ✅ Recognizes creative (generation) + analytical (ranking) combination
- ✅ Provides exactly 5 ideas (respects format constraint)
- ✅ Includes ranking with clear criteria
- ✅ Justifies rankings (not arbitrary ordering)

---

## Multi-Intent Complexity Matrix

| Complexity | Intent Count | Confidence Range | Clarification? | Example |
|------------|--------------|------------------|----------------|---------|
| **Simple** | 2 intents | 60-40 split | No | "Explain X and show example" |
| **Moderate** | 2-3 intents | 50-30-20 split | Sometimes | "Research, recommend, implement" |
| **Complex** | 3+ intents | Even distribution | Often | "Compare, analyze, decide, build, document" |
| **Sequential** | 2-3 intents | Clear ordering | No | "Fix bug then explain what caused it" |
| **Parallel** | 2-3 intents | No clear order | Sometimes | "Improve performance and add features" |

## Validation Checklist

For each multi-intent scenario:

### Intent Recognition
- [ ] All significant intents (>25% probability) identified
- [ ] Primary intent correctly determined
- [ ] Multi-intent flag set appropriately
- [ ] Intent relationships understood (sequential, parallel, nested)

### Response Strategy
- [ ] Response structure addresses all intents
- [ ] Intent ordering makes logical sense
- [ ] Primary intent receives appropriate focus
- [ ] Secondary intents not ignored

### Clarification Judgment
- [ ] Clear multi-intent doesn't trigger unnecessary clarification
- [ ] Ambiguous multi-intent triggers appropriate questions
- [ ] Questions disambiguate intent priorities when needed
- [ ] Proceeds confidently when all intents are clear

### Script Validation
- [ ] intent-classifier.py detects multi-intent correctly
- [ ] pattern-extractor.js identifies signals for each intent
- [ ] clarification-generator.py creates appropriate questions for ambiguous cases
- [ ] intent-validator.sh validates multi-intent analyses

## Success Metrics

- **Multi-Intent Detection**: 100% for requests with 2+ genuine intents
- **False Positive Rate**: <5% for single-intent requests incorrectly flagged as multi-intent
- **Response Completeness**: 100% of identified intents addressed in response
- **Structural Coherence**: Intents organized logically (sequential, priority-based, etc.)
- **Clarification Precision**: Only ask about intent priorities when genuinely ambiguous

## Edge Cases

### Edge Case 1: Implied Multi-Intent
**Request**: "Debug this code"
**Hidden Intents**: Fix bug (primary) + Explain what was wrong (implied learning)
**Strategy**: Address both even though learning not explicit

### Edge Case 2: Conflicting Intents
**Request**: "Keep it simple but make it production-ready"
**Conflict**: Simple vs Production-ready often in tension
**Strategy**: Clarify priority or find middle ground (simple architecture, production practices)

### Edge Case 3: Nested Intents
**Request**: "Teach me how to build an API by walking through creating one"
**Structure**: Learning (outer) contains Technical (inner)
**Strategy**: Recognize teaching as delivery mechanism for implementation

## Notes

Multi-intent requests are common in real-world usage. The key is distinguishing between:
1. **Genuine multi-intent**: Multiple distinct goals that all matter
2. **Nested intent**: One intent is the mechanism for achieving another
3. **Sequential intent**: Multiple phases of a single larger goal

Successful handling requires understanding these relationships and structuring responses accordingly.
