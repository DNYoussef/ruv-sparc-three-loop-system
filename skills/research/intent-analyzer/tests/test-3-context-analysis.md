# Test 3: Context Analysis and Constraint Detection

## Test Overview
Validates Intent Analyzer's ability to extract contextual signals (temporal, audience, expertise, constraints) from user requests and use this context to inform interpretation. Tests implicit constraint detection, expertise calibration, and appropriate response adaptation.

## Test Scenarios

### Scenario 1: Urgent Timeline Detection
**User Request**: "I need a quick Python script to parse JSON files ASAP - presentation is tomorrow morning"

**Expected Pattern Extraction**:
```json
{
  "temporal": [
    {
      "pattern": "urgency",
      "signal": "high_urgency",
      "matches": [
        {"matched_text": "quick", "position": 9},
        {"matched_text": "ASAP", "position": 48}
      ],
      "interpretation": "User values speed over completeness - prefer simple, direct solutions"
    },
    {
      "pattern": "timeline",
      "signal": "specific_timeline",
      "matches": [
        {"matched_text": "tomorrow morning", "position": 71}
      ],
      "interpretation": "Explicit deadline mentioned - prioritize meeting timeline"
    }
  ],
  "summary": {
    "urgency_level": "high"
  }
}
```

**Expected Response Adaptation**:
- Provide simple, working solution immediately
- Skip comprehensive error handling
- Avoid over-engineering
- Include quick usage example
- No lengthy explanations of alternatives

**Pass Criteria**:
- ✅ Detects multiple urgency signals ("quick", "ASAP")
- ✅ Recognizes explicit deadline ("tomorrow morning")
- ✅ Urgency level set to "high"
- ✅ Response prioritizes speed over completeness

---

### Scenario 2: Audience - Technical Expert
**User Request**: "How should I implement async/await error handling in my Node.js microservices? I'm using Kubernetes with distributed tracing"

**Expected Pattern Extraction**:
```json
{
  "audience": [
    {
      "pattern": "technical_expert",
      "signal": "expert_audience",
      "matches": [
        {"matched_text": "async/await", "position": 22},
        {"matched_text": "microservices", "position": 61},
        {"matched_text": "Kubernetes", "position": 88},
        {"matched_text": "distributed tracing", "position": 104}
      ],
      "interpretation": "Technical audience - precision and accuracy critical, can skip basics"
    }
  ],
  "expertise": [
    {
      "pattern": "expert_terminology",
      "signal": "expert_user",
      "matches": ["async/await", "microservices", "Kubernetes", "distributed tracing"]
    },
    {
      "pattern": "tool_awareness",
      "signal": "tool_familiarity",
      "matches": ["Node.js", "Kubernetes"]
    }
  ],
  "summary": {
    "audience_type": "technical_expert",
    "expertise_level": "expert"
  }
}
```

**Expected Response Adaptation**:
- Skip basic async/await explanation
- Focus on advanced patterns (error boundaries, circuit breakers)
- Reference distributed tracing integration
- Assume Kubernetes knowledge
- Use precise technical terminology

**Pass Criteria**:
- ✅ Detects 4+ expert-level terms
- ✅ Audience classified as "technical_expert"
- ✅ Expertise level set to "expert"
- ✅ Response skips basics, focuses on advanced topics

---

### Scenario 3: Audience - Beginner/General
**User Request**: "I'm new to programming and don't understand what an API is. Can you explain in simple terms?"

**Expected Pattern Extraction**:
```json
{
  "audience": [
    {
      "pattern": "general_audience",
      "signal": "general_audience",
      "matches": [
        {"matched_text": "new to programming", "position": 4},
        {"matched_text": "don't understand", "position": 27},
        {"matched_text": "simple terms", "position": 70}
      ],
      "interpretation": "General audience - prioritize clarity and accessibility over technical precision"
    }
  ],
  "expertise": [
    {
      "pattern": "general_question",
      "signal": "low_specificity",
      "matches": [{"matched_text": "what an API is", "position": 44}]
    }
  ],
  "summary": {
    "audience_type": "general",
    "expertise_level": "beginner"
  }
}
```

**Expected Response Adaptation**:
- Use analogies and simple language
- Avoid jargon (or define when necessary)
- Provide concrete examples
- Start with high-level concept before details
- Confirm understanding progressively

**Pass Criteria**:
- ✅ Detects beginner signals ("new to", "don't understand")
- ✅ Recognizes request for simplicity ("simple terms")
- ✅ Audience classified as "general"
- ✅ Expertise level set to "beginner"

---

### Scenario 4: Constraint - Production Environment
**User Request**: "I need a secure, production-ready authentication system for our enterprise SaaS application"

**Expected Pattern Extraction**:
```json
{
  "constraints": [
    {
      "pattern": "implicit_quality",
      "signal": "quality_constraint",
      "matches": [
        {"matched_text": "secure", "position": 9},
        {"matched_text": "production-ready", "position": 17},
        {"matched_text": "enterprise", "position": 58}
      ],
      "interpretation": "Implicit quality/reliability requirements - prioritize production-ready solutions"
    }
  ],
  "audience": [
    {
      "pattern": "formal_presentation",
      "signal": "formal_context",
      "matches": [
        {"matched_text": "enterprise", "position": 58},
        {"matched_text": "SaaS application", "position": 69}
      ]
    }
  ],
  "summary": {
    "has_constraints": true,
    "quality_requirements": ["security", "production", "enterprise-grade"]
  }
}
```

**Expected Response Adaptation**:
- Focus on battle-tested solutions (OAuth 2.0, OIDC)
- Address security best practices (token storage, refresh strategies)
- Include scalability considerations
- Mention compliance (GDPR, SOC 2)
- Avoid experimental or "toy" solutions

**Pass Criteria**:
- ✅ Detects quality signals ("secure", "production-ready", "enterprise")
- ✅ Quality constraint flag set
- ✅ Formal context recognized
- ✅ Response prioritizes proven, production-grade solutions

---

### Scenario 5: Constraint - Technology Stack
**User Request**: "Using React 18, TypeScript, and Zustand, how do I handle async state updates?"

**Expected Pattern Extraction**:
```json
{
  "constraints": [
    {
      "pattern": "technology_stack",
      "signal": "technology_constraint",
      "matches": [
        {"matched_text": "React 18", "position": 6},
        {"matched_text": "TypeScript", "position": 16},
        {"matched_text": "Zustand", "position": 32}
      ],
      "interpretation": "Specific technology requirements - must use stated tech stack"
    }
  ],
  "expertise": [
    {
      "pattern": "tool_awareness",
      "signal": "tool_familiarity",
      "matches": ["React 18", "TypeScript", "Zustand"]
    },
    {
      "pattern": "specific_question",
      "signal": "high_specificity",
      "matches": [{"matched_text": "async state updates", "position": 53}]
    }
  ],
  "summary": {
    "has_constraints": true,
    "technology_stack": ["React 18", "TypeScript", "Zustand"],
    "expertise_level": "advanced"
  }
}
```

**Expected Response Adaptation**:
- Solutions must use React 18, TypeScript, Zustand
- No generic React solutions
- Include TypeScript type annotations
- Zustand-specific patterns (not Redux, Context API)
- Assume familiarity with all three tools

**Pass Criteria**:
- ✅ Extracts all 3 technology constraints
- ✅ Technology constraint flag set
- ✅ Response uses ONLY specified technologies
- ✅ Expertise level calibrated to "advanced" (knows specific tools)

---

### Scenario 6: Constraint - Resource Limitations
**User Request**: "I need a free, open-source solution for image processing - can't use paid APIs or cloud services"

**Expected Pattern Extraction**:
```json
{
  "constraints": [
    {
      "pattern": "resource_limits",
      "signal": "resource_constraint",
      "matches": [
        {"matched_text": "free", "position": 9},
        {"matched_text": "open-source", "position": 15},
        {"matched_text": "can't use paid", "position": 61},
        {"matched_text": "can't use cloud services", "position": 80}
      ],
      "interpretation": "Resource limitations present - constrain solutions to available resources"
    }
  ],
  "summary": {
    "has_constraints": true,
    "resource_constraints": ["free", "open-source", "no_paid_apis", "no_cloud"],
    "deployment_constraint": "on-premise"
  }
}
```

**Expected Response Adaptation**:
- Only suggest free, open-source libraries (PIL/Pillow, OpenCV, ImageMagick)
- No AWS Rekognition, Google Vision, etc.
- Self-hosted/local solutions only
- Acknowledge resource constraints explicitly

**Pass Criteria**:
- ✅ Detects resource limitation signals
- ✅ Resource constraint flag set
- ✅ Solutions respect all stated constraints
- ✅ No paid or cloud-based suggestions

---

### Scenario 7: Meta-Request - Capability Query
**User Request**: "Can you help me build a mobile app? What can you actually do?"

**Expected Pattern Extraction**:
```json
{
  "meta": [
    {
      "pattern": "capability_query",
      "signal": "capability_question",
      "matches": [
        {"matched_text": "Can you help", "position": 0},
        {"matched_text": "What can you actually do", "position": 32}
      ],
      "interpretation": "User exploring capabilities - explain what is possible before proceeding"
    }
  ],
  "summary": {
    "is_meta_request": true
  }
}
```

**Expected Response Adaptation**:
- Explain mobile development capabilities first
- Clarify what IS possible (code generation, architecture advice, debugging)
- Clarify what ISN'T possible (can't compile/run apps directly)
- Then ask what specific help they need

**Pass Criteria**:
- ✅ Detects capability query pattern
- ✅ Meta-request flag set
- ✅ Response explains capabilities before diving into task
- ✅ Sets appropriate expectations

---

### Scenario 8: Meta-Request - Refinement
**User Request**: "That approach won't work for us. Can you suggest a different solution?"

**Expected Pattern Extraction**:
```json
{
  "meta": [
    {
      "pattern": "refinement_request",
      "signal": "refinement",
      "matches": [
        {"matched_text": "won't work", "position": 14},
        {"matched_text": "different solution", "position": 48}
      ],
      "interpretation": "User wants different approach - previous attempt did not meet needs"
    }
  ],
  "summary": {
    "is_meta_request": true,
    "previous_attempt_status": "rejected"
  }
}
```

**Expected Response Adaptation**:
- Acknowledge previous attempt didn't work
- Ask why it won't work (understand constraints)
- Propose distinctly different approach
- Avoid repeating similar patterns

**Pass Criteria**:
- ✅ Detects refinement request pattern
- ✅ Meta-request flag set
- ✅ Response asks for rejection reason
- ✅ New approach differs from previous

---

## Context Analysis Complexity Matrix

| Context Type | Signal Strength | Impact on Response | Clarification Need |
|--------------|----------------|--------------------|--------------------|
| **Temporal** | High urgency | Prefer simple/fast | Low |
| **Temporal** | Specific deadline | Prioritize meeting it | Low |
| **Temporal** | Quality over speed | Invest in thoroughness | Low |
| **Audience** | Expert | Skip basics, use jargon | Low |
| **Audience** | General/Beginner | Simplify, explain | Low |
| **Audience** | Formal | Polish, professionalism | Low |
| **Constraint** | Tech stack | Must use specified | Low |
| **Constraint** | Resources | Solutions within limits | Medium |
| **Constraint** | Quality | Production-grade only | Low |
| **Meta** | Capability query | Explain before proceeding | Low |
| **Meta** | Refinement | Understand why failed | High |
| **Meta** | Validation | Confirm understanding | Low |

## Validation Checklist

For each context analysis scenario:

### Signal Detection
- [ ] Temporal signals (urgency, timeline, quality) extracted
- [ ] Audience indicators (expertise, formality) identified
- [ ] Constraint markers (tech, resources, quality) detected
- [ ] Meta-request patterns recognized
- [ ] Expertise level calibrated from terminology

### Context Integration
- [ ] Multiple context signals synthesized coherently
- [ ] Contradictory signals flagged appropriately
- [ ] Context informs response strategy
- [ ] Implicit constraints surfaced

### Response Adaptation
- [ ] Response complexity matches expertise level
- [ ] Urgency reflected in solution approach
- [ ] All constraints respected in suggestions
- [ ] Audience type determines communication style
- [ ] Meta-requests handled appropriately

### Script Validation
- [ ] pattern-extractor.js detects all signal types
- [ ] Context summary accurately reflects signals
- [ ] Interpretation guidance aligns with detected patterns
- [ ] Conflicting patterns flagged in summary

## Success Metrics

- **Signal Detection Rate**: >95% for explicit signals
- **Implicit Constraint Detection**: >70% for commonly-implied constraints
- **Expertise Calibration Accuracy**: >85% correct level assignment
- **Context-Response Alignment**: 100% of constraints respected in suggestions
- **Meta-Request Handling**: 100% appropriate response strategy

## Edge Cases

### Edge Case 1: Conflicting Context
**Request**: "I need enterprise-grade security but can't use any paid services"
**Conflict**: Enterprise requirements vs free constraint
**Strategy**: Acknowledge tension, suggest best free options, note limitations

### Edge Case 2: Implied vs Explicit
**Request**: "Quick script for parsing logs"
**Explicit**: Quick (temporal)
**Implicit**: Production use? Learning? One-time task?
**Strategy**: Proceed with quick solution, note assumptions

### Edge Case 3: Context Evolution
**Request**: Initial: "Explain authentication", Follow-up: "I need to implement this for 1M users"
**Context Shift**: Learning → Production at scale
**Strategy**: Adapt from educational to production-focused

## Notes

Context analysis is critical for appropriate response calibration. The goal is extracting maximum relevant context from minimal explicit signals, while avoiding over-inference. When context is ambiguous or contradictory, clarification questions should target the specific contextual dimension (urgency? audience? constraints?) rather than the task itself.
