# Intent Analyzer - Advanced Intent Interpretation System

**Tier:** Gold
**Version:** 1.0.0
**Status:** Production Ready

## Overview

Intent Analyzer is a sophisticated system for deeply understanding user intent by going beyond surface-level requests to discover underlying goals, unstated constraints, and true needs. It applies cognitive science principles, probabilistic reasoning, and Socratic questioning to transform vague requests into well-understood goals.

## Quick Start

```bash
# Activate the skill (auto-loaded when needed)
# Intent Analyzer activates when:
# - User requests are ambiguous or could be interpreted multiple ways
# - Deeper understanding would significantly improve response quality
# - Multiple reasonable interpretations exist

# Example usage:
User: "Help me with Python"

# Intent Analyzer will:
# 1. Analyze possible interpretations (learning, debugging, coding, etc.)
# 2. Detect that clarification is needed (low confidence)
# 3. Ask strategic questions to disambiguate
# 4. Adapt response based on clarified intent
```

## When to Use

Apply Intent Analyzer when:
- User requests are **ambiguous** or could be interpreted multiple ways
- **Deeper understanding** of goals would significantly improve response quality
- The stated request might be a **proxy** for an unstated underlying need
- **Critical information** appears to be missing or assumed
- **Multiple reasonable interpretations** exist and choosing wrong would waste effort
- Helping users **clarify** complex or poorly-defined problems

## Features

### Core Capabilities

1. **Intent Classification**
   - Probabilistic categorization into 6 intent types
   - Multi-intent detection for complex requests
   - Confidence scoring and calibration

2. **Pattern Extraction**
   - Temporal signals (urgency, timelines, quality preferences)
   - Audience indicators (expertise level, formality, context)
   - Constraint markers (technology, resources, requirements)
   - Meta-request patterns (capability queries, refinements)

3. **Strategic Clarification**
   - Socratic questioning techniques
   - Disambiguation, constraint revelation, context gathering
   - Adaptive question generation (1-3 questions max)

4. **Context Analysis**
   - Expertise calibration from terminology
   - Implicit constraint detection
   - Contradictory signal identification

### Executable Tools

- **`intent-classifier.py`**: Classify requests into intent categories with confidence scores
- **`pattern-extractor.js`**: Extract linguistic patterns signaling specific intents
- **`clarification-generator.py`**: Generate strategic Socratic questions
- **`intent-validator.sh`**: Validate analysis completeness before proceeding

## File Structure

```
intent-analyzer/
├── SKILL.md                           # Core skill logic and workflows
├── README.md                          # This file
├── intent-analyzer-process.dot        # Workflow visualization (GraphViz)
├── references/                        # Additional documentation
├── resources/                         # Supporting resources
│   ├── readme.md                      # Resource overview
│   ├── scripts/                       # Executable tools
│   │   ├── intent-classifier.py       # Intent categorization
│   │   ├── pattern-extractor.js       # Pattern detection
│   │   ├── clarification-generator.py # Question generation
│   │   └── intent-validator.sh        # Analysis validation
│   └── templates/                     # Configuration and patterns
│       ├── intent-analysis-config.yaml    # Analysis parameters
│       ├── pattern-definitions.json       # Intent pattern library
│       └── clarification-templates.yaml   # Question templates
├── tests/                             # Test scenarios
│   ├── test-1-ambiguous-requests.md   # Ambiguity handling
│   ├── test-2-multi-intent.md         # Multi-intent requests
│   └── test-3-context-analysis.md     # Context extraction
└── examples/                          # Comprehensive examples
    ├── example-1-vague-request-analysis.md      # Vague request handling
    ├── example-2-multi-step-intent.md           # Multi-step workflows
    └── example-3-socratic-clarification.md      # Clarification strategies
```

## Usage Examples

### Example 1: Ambiguous Request

**User:** "Help me with Python"

**Intent Analyzer Process:**
```bash
# 1. Classify intent
$ python resources/scripts/intent-classifier.py "Help me with Python"
# Output: learning (35%), technical (30%), problem_solving (25%)
# → Multiple intents, low confidence → clarification needed

# 2. Extract patterns
$ node resources/scripts/pattern-extractor.js "Help me with Python"
# Output: technology_constraint (Python), no temporal/audience signals

# 3. Generate clarification questions
$ python resources/scripts/clarification-generator.py \
    --type disambiguation \
    --interpretations "learn Python,fix Python problem,write Python code"
# Output: Strategic questions to disambiguate

# 4. User clarifies: "I want to learn Python for data analysis"

# 5. Adapt response to clarified intent (learning + data analysis focus)
```

### Example 2: Multi-Intent Request

**User:** "Research best practices for microservices authentication and implement JWT-based auth for our Node.js API"

**Intent Analyzer Process:**
- Detects dual intent: Analytical (research) + Technical (implementation)
- No clarification needed (both intents are clear)
- Structures response in two phases: Research findings → Implementation guide

### Example 3: Context Analysis

**User:** "I need a quick Python script ASAP - presentation is tomorrow"

**Intent Analyzer Process:**
```bash
# Extract temporal signals
$ node resources/scripts/pattern-extractor.js "I need a quick Python script ASAP - presentation is tomorrow"
# Output: high_urgency (quick, ASAP), specific_timeline (tomorrow)

# Adapt response strategy:
# - Prioritize speed over completeness
# - Provide simple, working solution immediately
# - Skip comprehensive explanations
# - No over-engineering
```

## Configuration

Customize analysis behavior via `resources/templates/intent-analysis-config.yaml`:

```yaml
# Confidence thresholds
confidence:
  high_threshold: 0.80    # Proceed without clarification
  moderate_threshold: 0.50  # Proceed with acknowledgment
  low_threshold: 0.50     # Seek clarification

# Clarification strategy
clarification:
  max_questions_per_turn: 3
  question_strategy: adaptive  # adaptive | prioritized | comprehensive
```

## Testing

Run test scenarios to validate intent analysis:

```bash
# Test ambiguous request handling
# See: tests/test-1-ambiguous-requests.md

# Test multi-intent detection
# See: tests/test-2-multi-intent.md

# Test context extraction
# See: tests/test-3-context-analysis.md
```

## Performance Metrics

- **Intent Classification:** <100ms for typical requests
- **Pattern Extraction:** <50ms for typical requests
- **Clarification Generation:** <200ms with template lookup
- **Validation:** <50ms for standard analysis format

## Best Practices

1. **Don't over-clarify simple requests** - If confidence >80%, proceed directly
2. **Ask strategic questions** - 1-3 targeted questions beat 10 generic ones
3. **Progressive disclosure** - Ask basic intent first, details second
4. **Respect signals** - Use detected patterns to avoid redundant questions
5. **Acknowledge assumptions** - Make implicit interpretations explicit

## Integration Points

### With Other Skills

- **research-driven-planning**: Use intent analysis for requirement gathering
- **interactive-planner**: Combine with interactive questions for complex planning
- **sparc-methodology**: Apply intent analysis in Specification phase
- **pair-programming**: Calibrate pairing mode based on intent analysis

### With MCP Tools

- **Memory MCP**: Store successful interpretation patterns
- **Connascence Analyzer**: Detect code quality intent from terminology

## Troubleshooting

### Common Issues

**Issue:** Too many clarification questions asked
**Solution:** Check confidence thresholds in config, ensure questions are strategic

**Issue:** Missing obvious intent
**Solution:** Update pattern-definitions.json with domain-specific patterns

**Issue:** Contradictory signals detected
**Solution:** Review pattern weights in config, may need domain-specific rules

## Development

### Adding New Intent Categories

1. Update `pattern-definitions.json` with new patterns
2. Add classification logic to `intent-classifier.py`
3. Create clarification templates in `clarification-templates.yaml`
4. Test with representative examples

### Customizing for Domains

1. Add domain rules to `intent-analysis-config.yaml`
2. Extend pattern libraries with domain-specific signals
3. Create specialized clarification question sets

## Dependencies

**Python Scripts** (intent-classifier.py, clarification-generator.py):
- Python 3.8+
- Standard library only (no external dependencies)

**JavaScript Scripts** (pattern-extractor.js):
- Node.js 14+
- No external dependencies

**Shell Scripts** (intent-validator.sh):
- Bash 4.0+
- jq (for JSON processing)

## Version History

- **1.0.0** (2025-11-02): Initial Gold tier release
  - Core intent analysis with 6 categories
  - Pattern extraction for temporal, audience, constraint signals
  - Strategic clarification with Socratic questioning
  - Executable scripts for classification, extraction, generation, validation
  - Comprehensive test suite and examples

## Related Skills

- `interactive-planner` - Multi-select questions for requirements
- `research-driven-planning` - 5x pre-mortem planning
- `intent-analyzer` - THIS SKILL
- `sparc-methodology` - 5-phase SPARC workflow
- `pair-programming` - Adaptive pairing modes

## License

Part of the Claude Code Skills Library

## Support

For issues, questions, or feature requests:
1. Check examples in `examples/` directory
2. Review test scenarios in `tests/` directory
3. Consult SKILL.md for detailed workflow
4. Update pattern libraries for domain-specific needs

---

**Remember:** Intent Analyzer transforms request interpretation from surface-level reading to deep understanding. Use it thoughtfully—not every request needs deep analysis, but complex, ambiguous, or high-stakes requests benefit enormously from this systematic approach.
