# Test 1: Ambiguous Request Analysis

## Test Overview
Validates Intent Analyzer's ability to detect and handle ambiguous user requests where multiple interpretations have significant probability. Tests disambiguation strategy, clarification question generation, and appropriate confidence thresholds.

## Test Scenarios

### Scenario 1: Help Request Ambiguity
**User Request**: "Help me with Python"

**Expected Analysis**:
```json
{
  "primary_category": "learning",
  "confidence": 0.35,
  "categories": {
    "learning": 0.35,
    "technical": 0.30,
    "problem_solving": 0.20,
    "decision": 0.10,
    "creative": 0.03,
    "analytical": 0.02
  },
  "multi_intent": true,
  "ambiguous": true,
  "requires_clarification": true,
  "reasoning": "Multiple intent categories detected: learning, technical. Low confidence (0.35) - clarification recommended"
}
```

**Expected Pattern Extraction**:
```json
{
  "temporal": [],
  "audience": [],
  "constraints": [
    {
      "pattern": "technology_stack",
      "signal": "technology_constraint",
      "matches": [{"matched_text": "Python", "position": 12}]
    }
  ],
  "expertise": [],
  "summary": {
    "has_constraints": true,
    "expertise_level": "intermediate"
  }
}
```

**Expected Clarification Questions**:
1. Disambiguation: "Are you looking to learn Python concepts or fix a specific Python problem?"
2. Context: "What will you use Python for?"
3. Experience: "What's your experience level with Python?"

**Pass Criteria**:
- ✅ Detects multiple high-probability intents (learning, technical)
- ✅ Confidence below 50% triggers clarification
- ✅ Generates appropriate disambiguation questions
- ✅ Identifies technology constraint (Python)

---

### Scenario 2: "Make it Better" Vagueness
**User Request**: "Make this code better"

**Expected Analysis**:
```json
{
  "primary_category": "problem_solving",
  "confidence": 0.40,
  "categories": {
    "problem_solving": 0.40,
    "analytical": 0.30,
    "technical": 0.25,
    "learning": 0.05
  },
  "multi_intent": true,
  "ambiguous": true,
  "requires_clarification": true,
  "reasoning": "Multiple intent categories detected: problem_solving, analytical, technical. Moderate confidence (0.40) - clarification recommended. Key signals: better (verb)"
}
```

**Expected Pattern Extraction**:
```json
{
  "temporal": [],
  "audience": [],
  "constraints": [
    {
      "pattern": "implicit_quality",
      "signal": "quality_constraint",
      "interpretation": "Implicit quality improvement requirement"
    }
  ],
  "meta": [],
  "expertise": [],
  "summary": {
    "has_constraints": true
  }
}
```

**Expected Clarification Questions**:
1. "What aspects do you want to improve (performance, readability, maintainability, security)?"
2. "Is this for production use or learning purposes?"
3. "Are there specific problems you've noticed with the code?"

**Pass Criteria**:
- ✅ Recognizes vague "better" as requiring clarification
- ✅ Generates questions to disambiguate improvement type
- ✅ Asks about context (production vs learning)
- ✅ Confidence appropriately low (<50%)

---

### Scenario 3: Dual Intent - Learning While Building
**User Request**: "I need to build a REST API and want to understand how authentication works"

**Expected Analysis**:
```json
{
  "primary_category": "technical",
  "confidence": 0.55,
  "categories": {
    "technical": 0.55,
    "learning": 0.40,
    "analytical": 0.05
  },
  "multi_intent": true,
  "ambiguous": false,
  "requires_clarification": false,
  "reasoning": "Multiple intent categories detected: technical, learning. Moderate confidence (0.55) in technical intent - proceed with assumption acknowledgment. Key signals: build (verb), need (phrase), understand (verb)"
}
```

**Expected Pattern Extraction**:
```json
{
  "temporal": [],
  "audience": [],
  "constraints": [
    {
      "pattern": "technology_stack",
      "signal": "technology_constraint",
      "matches": [{"matched_text": "REST API", "position": 15}]
    }
  ],
  "expertise": [
    {
      "pattern": "specific_question",
      "signal": "high_specificity",
      "matches": [{"matched_text": "REST API", "position": 15}]
    }
  ]
}
```

**Expected Response Strategy**:
- Acknowledge dual intent: "I'll help you build a REST API while explaining authentication concepts"
- Provide working code with educational commentary
- Balance implementation with explanation

**Pass Criteria**:
- ✅ Detects both technical and learning intents
- ✅ Moderate confidence allows proceeding with acknowledgment
- ✅ Response strategy balances implementation + education
- ✅ No clarification needed (intent is clear despite dual nature)

---

### Scenario 4: Contradictory Signals - Urgency vs Quality
**User Request**: "I need a quick but comprehensive analysis of these options"

**Expected Analysis**:
```json
{
  "primary_category": "analytical",
  "confidence": 0.70,
  "categories": {
    "analytical": 0.70,
    "decision": 0.25,
    "creative": 0.05
  },
  "multi_intent": false,
  "ambiguous": true,
  "requires_clarification": true,
  "reasoning": "High confidence (0.70) in analytical intent. Key signals: analysis (noun), options (noun). WARNING: Contradictory temporal signals detected - both urgency ('quick') and thoroughness ('comprehensive')"
}
```

**Expected Pattern Extraction**:
```json
{
  "temporal": [
    {
      "pattern": "high_urgency",
      "signal": "high_urgency",
      "matches": [{"matched_text": "quick", "position": 9}],
      "interpretation": "User values speed over completeness"
    },
    {
      "pattern": "comprehensive",
      "signal": "quality_over_speed",
      "matches": [{"matched_text": "comprehensive", "position": 19}],
      "interpretation": "User values thoroughness - invest time in comprehensive response"
    }
  ],
  "summary": {
    "urgency_level": "CONFLICTING"
  },
  "interpretation_guidance": [
    "CONFLICT DETECTED: User values speed over completeness",
    "CONFLICT DETECTED: User values thoroughness - invest time in comprehensive response"
  ]
}
```

**Expected Clarification Questions**:
1. "I see you want both speed and comprehensiveness. Which is more important - a quick overview or a thorough analysis?"
2. "Would you prefer a brief summary now that I can expand later, or a complete analysis that takes more time?"

**Pass Criteria**:
- ✅ Detects contradictory temporal signals
- ✅ Flags conflict in interpretation guidance
- ✅ Generates question to resolve priority conflict
- ✅ Validation script identifies contradictory signals

---

### Scenario 5: Expert vs Novice Signal Conflict
**User Request**: "Explain OAuth 2.0 in simple terms"

**Expected Analysis**:
```json
{
  "primary_category": "learning",
  "confidence": 0.85,
  "categories": {
    "learning": 0.85,
    "analytical": 0.10,
    "technical": 0.05
  },
  "multi_intent": false,
  "ambiguous": false,
  "requires_clarification": false,
  "reasoning": "High confidence (0.85) in learning intent - proceed with dominant interpretation. Key signals: explain (verb)"
}
```

**Expected Pattern Extraction**:
```json
{
  "audience": [
    {
      "pattern": "general_audience",
      "signal": "general_audience",
      "matches": [{"matched_text": "simple terms", "position": 19}],
      "interpretation": "General audience - prioritize clarity and accessibility"
    }
  ],
  "expertise": [
    {
      "pattern": "expert_terminology",
      "signal": "expert_user",
      "matches": [{"matched_text": "OAuth", "position": 8}],
      "interpretation": "Expert-level terminology - user has deep technical knowledge"
    }
  ],
  "summary": {
    "audience_type": "general",
    "expertise_level": "CONFLICTING"
  }
}
```

**Expected Response Strategy**:
- Recognize user knows OAuth exists (expert awareness) but wants accessible explanation
- Provide clear explanation without excessive simplification
- Skip very basic web concepts, focus on OAuth specifics

**Pass Criteria**:
- ✅ High confidence in learning intent
- ✅ Detects both expert terminology and general audience request
- ✅ Resolves apparent conflict: expert awareness + accessibility need
- ✅ Response appropriately calibrated (not overly simplified)

---

## Validation Checklist

For each test scenario, verify:

### Analysis Quality
- [ ] Intent categories accurately reflect possible interpretations
- [ ] Confidence scores calibrated appropriately (sum to ~1.0)
- [ ] Multi-intent detection works for genuinely ambiguous requests
- [ ] Ambiguous flag set correctly based on confidence thresholds

### Pattern Detection
- [ ] Temporal signals (urgency, timeline) extracted correctly
- [ ] Audience indicators detected from language cues
- [ ] Constraint markers identified (explicit and implicit)
- [ ] Expertise signals recognized from terminology
- [ ] Contradictory patterns flagged appropriately

### Clarification Strategy
- [ ] Clarification required when confidence < 50%
- [ ] Clarification optional for 50-80% confidence
- [ ] No clarification for > 80% confidence (unless conflicting signals)
- [ ] Questions target highest-priority ambiguities
- [ ] Question count limited to 3 maximum

### Script Integration
- [ ] intent-classifier.py produces valid JSON
- [ ] pattern-extractor.js produces valid JSON
- [ ] clarification-generator.py creates appropriate questions
- [ ] intent-validator.sh passes for valid analyses
- [ ] intent-validator.sh fails for incomplete/invalid analyses

## Expected Failures (Anti-Patterns)

### Test Case: Empty Request
**Input**: ""
**Expected**: Error - "Empty input text"

### Test Case: Missing Confidence
**Input**: `{"primary_category": "learning"}`
**Expected**: Validation failure - "Missing required field: confidence"

### Test Case: Invalid Confidence Range
**Input**: `{"confidence": 1.5}`
**Expected**: Validation failure - "Confidence score 1.5 outside valid range"

### Test Case: Probability Sum Mismatch
**Input**: `{"categories": {"creative": 0.3, "technical": 0.3}}`
**Expected**: Validation warning - "Category probabilities sum to 0.6 (expected ~1.0)"

## Success Metrics

- **Ambiguity Detection Rate**: 100% for genuinely ambiguous requests
- **False Positive Rate**: <10% for clear requests incorrectly flagged as ambiguous
- **Clarification Precision**: Questions address actual ambiguities, not spurious ones
- **Conflict Detection**: 100% for contradictory signal pairs
- **Validation Pass Rate**: 100% for well-formed analyses

## Notes

This test suite focuses on the most challenging aspect of intent analysis: handling ambiguity. Clear requests should proceed quickly without over-clarification. Ambiguous requests must be detected reliably and clarification questions should be strategic, not interrogatory.
