# Micro-Skill Creator Resources

This directory contains reusable components, templates, and scripts for creating atomic, evidence-based micro-skills.

## Directory Structure

### scripts/
Executable utilities for micro-skill lifecycle management:
- **skill-generator.py** - Generate micro-skills from templates using evidence-based patterns
- **skill-validator.sh** - Validate micro-skill structure, contracts, and agent design
- **skill-optimizer.js** - Optimize skill content for clarity and performance
- **skill-packager.py** - Package micro-skills for distribution

### templates/
Reusable templates for different micro-skill types:
- **atomic-skill-template.yaml** - Base template for atomic micro-skills
- **micro-skill-frontmatter.json** - Standard YAML frontmatter structure
- **skill-metadata.yaml** - Metadata schema for skill discovery

## Usage Guidelines

### Creating a New Micro-Skill

1. **Choose Template**: Select appropriate template based on task type
2. **Generate Structure**: Run `skill-generator.py` with your specifications
3. **Design Agent**: Apply evidence-based prompting patterns (self-consistency, program-of-thought, plan-and-solve)
4. **Validate**: Run `skill-validator.sh` to check structure and contracts
5. **Optimize**: Use `skill-optimizer.js` for clarity and performance
6. **Package**: Create distribution package with `skill-packager.py`

### Evidence-Based Pattern Selection

**Self-Consistency** (for factual/extraction tasks):
- Extract information from multiple perspectives
- Cross-reference findings for accuracy
- Flag inconsistencies and ambiguities

**Program-of-Thought** (for analytical/logical tasks):
- Decompose problem systematically
- Show intermediate reasoning
- Validate logical consistency

**Plan-and-Solve** (for complex/multi-step tasks):
- Create comprehensive plan with dependencies
- Execute systematically with validation checkpoints
- Return complete solution with metadata

### Validation Checklist

Before packaging, ensure your micro-skill:
- [ ] Has single, well-defined responsibility
- [ ] Uses appropriate evidence-based pattern
- [ ] Includes specialist agent system prompt
- [ ] Defines explicit input/output contracts
- [ ] Handles failure modes gracefully
- [ ] Includes validation test cases
- [ ] Enables neural training integration
- [ ] Documents composition interfaces

### Best Practices

1. **Atomic Design**: One responsibility per skill
2. **Clean Contracts**: Explicit input/output specifications
3. **Failure Awareness**: Document and handle edge cases
4. **Composability**: Design for cascade workflows
5. **Quality Gates**: Systematic validation and testing
6. **Continuous Learning**: Enable neural training feedback

## Integration Points

### With Agent-Creator
Use agent-creator principles for designing specialist agents with domain expertise and evidence-based methodologies.

### With Cascade-Orchestrator
Design micro-skills with clean interfaces for sequential, parallel, conditional, and iterative composition.

### With Functionality-Audit
Apply systematic testing patterns to validate micro-skill execution in sandbox environments.

### With Slash-Command-Encoder
Create command bindings for direct /command access to micro-skills.

## Version Control

All templates and scripts follow semantic versioning:
- **Major**: Breaking changes to interfaces or structure
- **Minor**: New features maintaining backward compatibility
- **Patch**: Bug fixes and improvements

## Support

For issues, improvements, or new template requests, consult the main micro-skill-creator SKILL.md documentation.
