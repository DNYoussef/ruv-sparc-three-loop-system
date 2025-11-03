#!/usr/bin/env python3
"""
Micro-Skill Generator - Create atomic, evidence-based micro-skills from templates

This script generates production-ready micro-skills with:
- Evidence-based prompting patterns (self-consistency, program-of-thought, plan-and-solve)
- Specialist agent system prompts
- Explicit input/output contracts
- Validation test cases
- Neural training integration

Usage:
    python skill-generator.py --name <skill-name> --pattern <evidence-pattern> --domain <expertise>
    python skill-generator.py --config <config.yaml>
    python skill-generator.py --interactive

Version: 2.0.0
"""

import argparse
import json
import yaml
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Evidence-based pattern templates
EVIDENCE_PATTERNS = {
    "self-consistency": {
        "methodology": """1. Extract information from multiple perspectives
2. Cross-reference findings for consistency
3. Flag any inconsistencies or ambiguities
4. Provide confidence scores
5. Return validated results""",
        "use_case": "Factual extraction, data validation, information retrieval",
        "techniques": ["multi-angle analysis", "cross-validation", "confidence scoring"]
    },
    "program-of-thought": {
        "methodology": """1. Decompose problem into logical components
2. Work through each component systematically
3. Show intermediate reasoning
4. Validate logical consistency
5. Synthesize final analysis""",
        "use_case": "Analytical tasks, logical reasoning, systematic processing",
        "techniques": ["decomposition", "step-by-step reasoning", "validation"]
    },
    "plan-and-solve": {
        "methodology": """1. Create comprehensive plan with dependencies
2. Break into executable steps
3. Execute plan systematically
4. Validate completion at each step
5. Return complete solution""",
        "use_case": "Complex tasks, multi-step workflows, generation tasks",
        "techniques": ["planning", "systematic execution", "checkpoint validation"]
    }
}

# Skill templates by type
SKILL_TEMPLATES = {
    "extraction": {
        "agent_role": "extraction specialist",
        "pattern": "self-consistency",
        "input_schema": {"source_document": "string | file_path", "target_schema": "json_schema"},
        "output_schema": {"extracted_data": "object", "confidence_scores": "object", "ambiguities": "array[string]"}
    },
    "validation": {
        "agent_role": "validation specialist",
        "pattern": "program-of-thought",
        "input_schema": {"data": "object | array", "specification": "schema | rules_file"},
        "output_schema": {"validation_result": "object", "violations": "array", "suggested_fixes": "array"}
    },
    "generation": {
        "agent_role": "generation specialist",
        "pattern": "plan-and-solve",
        "input_schema": {"specification": "object | markdown", "templates": "array (optional)"},
        "output_schema": {"generated_artifact": "string | object", "generation_metadata": "object"}
    },
    "analysis": {
        "agent_role": "analysis specialist",
        "pattern": "program-of-thought",
        "input_schema": {"data": "object | array | file_path", "analysis_type": "string"},
        "output_schema": {"analysis_report": "object", "key_findings": "array", "recommendations": "array"}
    }
}


class MicroSkillGenerator:
    def __init__(self, output_dir: str = "./skills"):
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().isoformat()

    def generate_skill(self, config: Dict) -> Path:
        """Generate complete micro-skill from configuration"""
        skill_name = config['name']
        skill_dir = self.output_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Generate SKILL.md
        skill_content = self._generate_skill_md(config)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(skill_content)

        # Generate test cases if requested
        if config.get('generate_tests', True):
            tests_dir = skill_dir / "tests"
            tests_dir.mkdir(exist_ok=True)
            test_content = self._generate_test_cases(config)
            (tests_dir / f"test-{skill_name}.md").write_text(test_content)

        # Generate examples if requested
        if config.get('generate_examples', True):
            examples_dir = skill_dir / "examples"
            examples_dir.mkdir(exist_ok=True)
            example_content = self._generate_example(config)
            (examples_dir / f"example-{skill_name}.md").write_text(example_content)

        print(f"✓ Generated micro-skill: {skill_name}")
        print(f"  Location: {skill_dir}")
        print(f"  Pattern: {config.get('pattern', 'custom')}")
        print(f"  Agent: {config.get('agent_role', 'specialist')}")

        return skill_dir

    def _generate_skill_md(self, config: Dict) -> str:
        """Generate SKILL.md content"""
        pattern = config.get('pattern', 'plan-and-solve')
        evidence = EVIDENCE_PATTERNS.get(pattern, EVIDENCE_PATTERNS['plan-and-solve'])

        frontmatter = f"""---
name: {config['name']}
description: {config.get('description', 'Atomic micro-skill with evidence-based agent')}
tags: [{', '.join(config.get('tags', ['micro-skill', 'atomic', pattern]))}]
version: 1.0.0
---

# {config.get('title', config['name'].replace('-', ' ').title())}

## Purpose
{config.get('purpose', 'Single-responsibility micro-skill with specialist agent')}

## Specialist Agent

I am a {config.get('agent_role', 'specialist')} using {pattern} for {evidence['use_case']}.

### Methodology ({pattern.replace('-', ' ').title()} Pattern)
{evidence['methodology']}

### Expertise
{self._format_list(config.get('expertise', ['Core domain knowledge', 'Edge case handling', 'Quality standards']))}

### Failure Modes & Mitigations
{self._format_failure_modes(config.get('failure_modes', {}))}

## Input Contract

```yaml
{self._format_yaml(config.get('input_schema', {}))}
```

## Output Contract

```yaml
{self._format_yaml(config.get('output_schema', {}))}
```

## Validation Rules

{self._format_list(config.get('validation_rules', ['Input must match schema', 'Output must be complete', 'Quality standards met']))}

## Integration Points

### Cascades
{config.get('cascade_info', 'Can be composed in sequential, parallel, or conditional workflows')}

### Commands
{config.get('command_binding', 'Available via slash command: /' + config['name'])}

### Dependencies
{self._format_list(config.get('dependencies', ['None - fully atomic']))}

## Test Coverage

- Normal operation: {config.get('test_coverage', '✓')}
- Boundary conditions: {config.get('test_coverage', '✓')}
- Error cases: {config.get('test_coverage', '✓')}
- Edge cases: {config.get('test_coverage', '✓')}

## Neural Training

```yaml
training:
  pattern: {config.get('cognitive_pattern', 'convergent')}
  feedback_collection: true
  improvement_iteration: true
  success_tracking: true
```

---
*Generated by micro-skill-creator v2.0.0 on {self.timestamp}*
"""
        return frontmatter

    def _generate_test_cases(self, config: Dict) -> str:
        """Generate test case documentation"""
        return f"""# Test Cases: {config['name']}

## Test 1: Normal Operation
**Input**: Typical valid input matching schema
**Expected**: Successful execution with valid output
**Validation**: Output matches contract, no errors

## Test 2: Boundary Conditions
**Input**: Edge values (empty, max size, special characters)
**Expected**: Graceful handling with appropriate warnings
**Validation**: No crashes, clear error messages if invalid

## Test 3: Error Cases
**Input**: Invalid input violating schema
**Expected**: Error detection with helpful messages
**Validation**: Specific error identification, suggested fixes

## Test 4: Edge Cases
**Input**: Unusual but valid combinations
**Expected**: Correct handling with metadata
**Validation**: Output quality maintained, edge case documented

## Test 5: Performance
**Input**: Large or complex valid input
**Expected**: Completion within acceptable time
**Validation**: Performance metrics logged, no degradation

## Test 6: Composition
**Input**: Output from upstream skill
**Expected**: Seamless integration in cascade
**Validation**: Contract compatibility verified

---
*Generated test cases for {config['name']} v1.0.0*
"""

    def _generate_example(self, config: Dict) -> str:
        """Generate usage example"""
        return f"""# Example Usage: {config['name']}

## Scenario
{config.get('example_scenario', 'Common use case for this micro-skill')}

## Input

```yaml
{self._format_yaml(config.get('example_input', {'input': 'example data'}))}
```

## Execution

```bash
# Via Claude Code skill activation
"Use {config['name']} to process this data..."

# Via slash command (if configured)
/{config['name']} --input example.json --output result.json
```

## Expected Output

```yaml
{self._format_yaml(config.get('example_output', {'output': 'processed result', 'metadata': {}}))}
```

## Integration in Cascade

```yaml
workflow:
  - step: validate-input
  - step: {config['name']}  # This micro-skill
  - step: transform-output
  - step: generate-report
```

## Common Variations

### Variation 1: {config.get('variation_1', 'Alternative input format')}
### Variation 2: {config.get('variation_2', 'Error handling scenario')}
### Variation 3: {config.get('variation_3', 'Performance optimization')}

## Tips

- {config.get('tip_1', 'Always validate input schema first')}
- {config.get('tip_2', 'Check output confidence scores')}
- {config.get('tip_3', 'Handle failure modes explicitly')}

---
*Example for {config['name']} v1.0.0*
"""

    def _format_list(self, items: List[str]) -> str:
        """Format list items as markdown bullets"""
        return '\n'.join(f"- {item}" for item in items)

    def _format_failure_modes(self, modes: Dict) -> str:
        """Format failure modes dictionary"""
        if not modes:
            return "- No specific failure modes documented"
        return '\n'.join(f"- {k}: {v}" for k, v in modes.items())

    def _format_yaml(self, data: Dict) -> str:
        """Format dictionary as YAML with indentation"""
        return yaml.dump(data, default_flow_style=False, indent=2)

    def interactive_mode(self):
        """Interactive CLI for skill generation"""
        print("🔨 Micro-Skill Generator - Interactive Mode\n")

        config = {}
        config['name'] = input("Skill name (kebab-case): ").strip()
        config['description'] = input("One-sentence description: ").strip()

        print("\nEvidence-based patterns:")
        for i, (pattern, info) in enumerate(EVIDENCE_PATTERNS.items(), 1):
            print(f"{i}. {pattern}: {info['use_case']}")

        pattern_choice = int(input("Select pattern (1-3): ")) - 1
        config['pattern'] = list(EVIDENCE_PATTERNS.keys())[pattern_choice]

        config['agent_role'] = input("Agent role (e.g., 'extraction specialist'): ").strip()
        config['purpose'] = input("Skill purpose (one sentence): ").strip()

        print("\nGenerating micro-skill...")
        skill_dir = self.generate_skill(config)
        print(f"\n✓ Complete! Micro-skill created at: {skill_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate atomic micro-skills with evidence-based agents")
    parser.add_argument('--name', help='Skill name (kebab-case)')
    parser.add_argument('--pattern', choices=['self-consistency', 'program-of-thought', 'plan-and-solve'],
                        help='Evidence-based prompting pattern')
    parser.add_argument('--domain', help='Domain expertise')
    parser.add_argument('--config', help='Path to YAML configuration file')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--output', default='./skills', help='Output directory')

    args = parser.parse_args()

    generator = MicroSkillGenerator(args.output)

    if args.interactive:
        generator.interactive_mode()
    elif args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        generator.generate_skill(config)
    elif args.name and args.pattern:
        config = {
            'name': args.name,
            'pattern': args.pattern,
            'domain': args.domain or 'general'
        }
        generator.generate_skill(config)
    else:
        parser.print_help()
        print("\n💡 Tip: Use --interactive for guided skill creation")


if __name__ == '__main__':
    main()
