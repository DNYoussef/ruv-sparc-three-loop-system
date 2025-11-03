# Intent Analyzer Resources

This directory contains supporting resources for the Intent Analyzer skill, organized to enable sophisticated intent interpretation through executable scripts, configuration templates, and reference materials.

## Directory Structure

```
resources/
├── readme.md                          # This file
├── scripts/                           # Executable tools for intent analysis
│   ├── intent-classifier.py           # Classify user intent into categories
│   ├── pattern-extractor.js           # Extract intent patterns from requests
│   ├── clarification-generator.py     # Generate Socratic clarifying questions
│   └── intent-validator.sh            # Validate analysis completeness
└── templates/                         # Configuration and pattern templates
    ├── intent-analysis-config.yaml    # Configuration for analysis parameters
    ├── pattern-definitions.json       # Structured intent pattern library
    └── clarification-templates.yaml   # Question templates for various scenarios
```

## Scripts Overview

### intent-classifier.py
Classifies user requests into intent categories using probabilistic analysis:
- Creative, Analytical, Technical, Learning, Decision, Problem-Solving
- Outputs confidence scores for each category
- Identifies multi-intent requests
- Flags ambiguous cases requiring clarification

**Usage**: `python intent-classifier.py "user request text"`

### pattern-extractor.js
Extracts linguistic patterns that signal specific intents:
- Temporal signals (urgency, timeframes)
- Audience indicators (formality, expertise level)
- Constraint markers (explicit/implicit limitations)
- Meta-request patterns (capability questions, refinement requests)

**Usage**: `node pattern-extractor.js "user request text"`

### clarification-generator.py
Generates strategic Socratic questions for ambiguous requests:
- Disambiguation questions (choose between interpretations)
- Constraint revelation questions (surface unstated requirements)
- Context gathering questions (build essential understanding)
- Assumption validation questions (verify implicit assumptions)

**Usage**: `python clarification-generator.py --type disambiguation --interpretations "A,B,C"`

### intent-validator.sh
Validates that intent analysis is complete before proceeding:
- Checks all required analysis dimensions are addressed
- Verifies confidence scores are calibrated
- Ensures critical assumptions are surfaced
- Confirms appropriate clarification strategy

**Usage**: `bash intent-validator.sh analysis-results.json`

## Templates Overview

### intent-analysis-config.yaml
Configuration parameters for intent analysis behavior:
- Confidence thresholds for clarification (high/moderate/low)
- Pattern matching sensitivity settings
- Domain-specific analysis rules
- Clarification question limits per turn

### pattern-definitions.json
Structured library of intent patterns:
- Request type patterns (creative, analytical, technical, etc.)
- Temporal patterns (urgency signals, timeline indicators)
- Audience patterns (expertise markers, formality cues)
- Constraint patterns (explicit requirements, implicit limitations)
- Meta-patterns (conversation management signals)

### clarification-templates.yaml
Question templates organized by scenario:
- Disambiguation question patterns
- Constraint revelation strategies
- Context gathering approaches
- Assumption validation techniques
- Domain-specific question libraries

## Usage Guidelines

**For Script Integration**:
1. Scripts are designed to be called from SKILL.md workflows
2. Each script outputs structured JSON for easy parsing
3. Scripts include error handling for malformed inputs
4. All scripts support verbose mode for debugging

**For Template Customization**:
1. Templates use YAML/JSON for easy modification
2. Add domain-specific patterns to pattern-definitions.json
3. Customize confidence thresholds in intent-analysis-config.yaml
4. Extend clarification templates for new scenarios

**For Progressive Disclosure**:
1. Core intent analysis logic lives in SKILL.md
2. Scripts handle deterministic operations (classification, pattern matching)
3. Templates provide reusable patterns without bloating SKILL.md
4. References contain detailed domain knowledge loaded as needed

## Integration with SKILL.md

The Intent Analyzer workflow in SKILL.md references these resources at key decision points:

**Phase 1 - Deep Analysis**:
- Calls `intent-classifier.py` to categorize request type
- Uses `pattern-extractor.js` to identify intent signals
- References `pattern-definitions.json` for pattern recognition

**Phase 2 - Decision Point**:
- Uses confidence thresholds from `intent-analysis-config.yaml`
- Validates analysis with `intent-validator.sh`

**Phase 3 - Socratic Clarification**:
- Generates questions using `clarification-generator.py`
- Selects templates from `clarification-templates.yaml`
- Limits questions per turn based on config settings

**Phase 4 - Interpretation Reconstruction**:
- Validates final interpretation completeness
- Documents assumptions using structured formats

## Extension Points

Add new intent categories by:
1. Updating `pattern-definitions.json` with new patterns
2. Adding classification logic to `intent-classifier.py`
3. Creating clarification templates in `clarification-templates.yaml`

Customize for specific domains by:
1. Adding domain rules to `intent-analysis-config.yaml`
2. Extending pattern libraries with domain-specific signals
3. Creating specialized clarification question sets

## Best Practices

1. **Keep scripts focused**: Each script has a single, well-defined purpose
2. **Maintain template libraries**: Regular updates improve pattern recognition
3. **Version configurations**: Track changes to analysis parameters
4. **Test edge cases**: Validate scripts with ambiguous/multi-intent requests
5. **Document customizations**: Record domain-specific additions

## Dependencies

**Python Scripts** (intent-classifier.py, clarification-generator.py):
- Python 3.8+
- Standard library only (no external dependencies for portability)
- JSON output for easy integration

**JavaScript Scripts** (pattern-extractor.js):
- Node.js 14+
- No external dependencies
- JSON output for easy integration

**Shell Scripts** (intent-validator.sh):
- Bash 4.0+
- jq for JSON processing
- Standard Unix utilities (grep, awk, sed)

## Performance Considerations

- Intent classification: <100ms for typical requests
- Pattern extraction: <50ms for typical requests
- Clarification generation: <200ms with template lookup
- Validation: <50ms for standard analysis format

## Troubleshooting

**Script not found**: Ensure scripts directory is in PATH or use absolute paths
**Permission denied**: Make scripts executable with `chmod +x script-name`
**JSON parse errors**: Validate input format matches script expectations
**Missing templates**: Verify template files exist and are valid YAML/JSON

## Future Enhancements

Planned additions:
- Machine learning-based intent classification (when training data available)
- Cross-session pattern learning (track successful interpretations)
- Multi-language support (extend pattern definitions)
- Real-time confidence calibration (learn from user corrections)
