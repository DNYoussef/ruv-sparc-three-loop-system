#!/usr/bin/env python3
"""
Agent Specification Validator
Validates agent YAML specifications for completeness, correctness, and best practices
Usage: python validate_agent.py <agent-spec.yaml> [--json] [--strict]
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Evidence-based prompting requirements
REQUIRED_SECTIONS = {
    "metadata": ["name", "version", "category", "description"],
    "role": ["identity", "expertise", "responsibilities"],
    "capabilities": ["primary", "secondary"],
    "prompting": ["techniques", "examples"],
    "quality": ["success_criteria", "failure_modes"]
}

VALID_CATEGORIES = [
    "specialist", "coordinator", "hybrid", "research",
    "development", "testing", "documentation", "security"
]

VALID_PROMPTING_TECHNIQUES = [
    "chain-of-thought", "few-shot", "role-based", "plan-and-solve",
    "self-consistency", "program-of-thought", "least-to-most"
]

def validate_metadata(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate agent metadata section"""
    errors = []

    if "metadata" not in spec:
        return False, ["Missing 'metadata' section"]

    metadata = spec["metadata"]

    # Check required fields
    for field in REQUIRED_SECTIONS["metadata"]:
        if field not in metadata:
            errors.append(f"Missing metadata.{field}")
        elif not metadata[field]:
            errors.append(f"Empty metadata.{field}")

    # Validate name format (kebab-case)
    if "name" in metadata:
        name = metadata["name"]
        if not name.replace("-", "").replace("_", "").isalnum():
            errors.append("Agent name should use kebab-case or snake_case")

    # Validate category
    if "category" in metadata:
        if metadata["category"] not in VALID_CATEGORIES:
            errors.append(f"Invalid category. Must be one of: {', '.join(VALID_CATEGORIES)}")

    # Validate description length (evidence-based: 80-150 words)
    if "description" in metadata:
        word_count = len(metadata["description"].split())
        if word_count < 80:
            errors.append(f"Description too short ({word_count} words). Recommended: 80-150 words")
        elif word_count > 200:
            errors.append(f"Description too long ({word_count} words). Recommended: 80-150 words")

    # Check version format (semver)
    if "version" in metadata:
        version = str(metadata["version"])
        parts = version.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            errors.append("Version should follow semver format (e.g., 1.0.0)")

    return len(errors) == 0, errors

def validate_role(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate role definition (critical for agent performance)"""
    errors = []

    if "role" not in spec:
        return False, ["Missing 'role' section - critical for agent identity"]

    role = spec["role"]

    # Check required fields
    for field in REQUIRED_SECTIONS["role"]:
        if field not in role:
            errors.append(f"Missing role.{field}")

    # Validate identity clarity
    if "identity" in role:
        if len(role["identity"]) < 20:
            errors.append("Role identity too brief. Provide clear, specific agent persona")

    # Validate expertise
    if "expertise" in role:
        if isinstance(role["expertise"], list):
            if len(role["expertise"]) == 0:
                errors.append("Expertise list is empty")
            elif len(role["expertise"]) > 10:
                errors.append("Too many expertise areas. Focus on 3-7 core competencies")
        else:
            errors.append("Expertise should be a list of domain areas")

    # Validate responsibilities
    if "responsibilities" in role:
        if isinstance(role["responsibilities"], list):
            if len(role["responsibilities"]) == 0:
                errors.append("Responsibilities list is empty")
        else:
            errors.append("Responsibilities should be a list")

    return len(errors) == 0, errors

def validate_capabilities(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate capability definitions"""
    errors = []

    if "capabilities" not in spec:
        return False, ["Missing 'capabilities' section"]

    caps = spec["capabilities"]

    # Check primary capabilities
    if "primary" not in caps:
        errors.append("Missing capabilities.primary")
    elif not isinstance(caps["primary"], list) or len(caps["primary"]) == 0:
        errors.append("Primary capabilities must be a non-empty list")

    # Secondary capabilities (optional but recommended)
    if "secondary" in caps:
        if not isinstance(caps["secondary"], list):
            errors.append("Secondary capabilities must be a list")

    # Check for tools/integrations
    if "tools" in caps:
        if not isinstance(caps["tools"], list):
            errors.append("Capabilities.tools must be a list")

    return len(errors) == 0, errors

def validate_prompting(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate evidence-based prompting techniques"""
    errors = []

    if "prompting" not in spec:
        return False, ["Missing 'prompting' section - required for agent effectiveness"]

    prompting = spec["prompting"]

    # Check techniques
    if "techniques" not in prompting:
        errors.append("Missing prompting.techniques")
    else:
        techniques = prompting["techniques"]
        if not isinstance(techniques, list):
            errors.append("Prompting techniques must be a list")
        else:
            # Validate technique names
            for tech in techniques:
                if tech not in VALID_PROMPTING_TECHNIQUES:
                    errors.append(f"Unknown prompting technique: {tech}")

    # Check examples (few-shot learning)
    if "examples" not in prompting:
        errors.append("Missing prompting.examples - few-shot learning improves performance")
    else:
        examples = prompting["examples"]
        if not isinstance(examples, list):
            errors.append("Prompting examples must be a list")
        elif len(examples) < 2:
            errors.append("Provide at least 2-3 examples for effective few-shot learning")
        else:
            # Validate example structure
            for i, example in enumerate(examples):
                if not isinstance(example, dict):
                    errors.append(f"Example {i+1} must be a dictionary")
                elif "input" not in example or "output" not in example:
                    errors.append(f"Example {i+1} must have 'input' and 'output' fields")

    return len(errors) == 0, errors

def validate_quality(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate quality criteria and failure modes"""
    errors = []

    if "quality" not in spec:
        return False, ["Missing 'quality' section"]

    quality = spec["quality"]

    # Success criteria
    if "success_criteria" not in quality:
        errors.append("Missing quality.success_criteria")
    elif not isinstance(quality["success_criteria"], list):
        errors.append("Success criteria must be a list")

    # Failure modes (helps prevent common errors)
    if "failure_modes" not in quality:
        errors.append("Missing quality.failure_modes - helps prevent common errors")
    elif not isinstance(quality["failure_modes"], list):
        errors.append("Failure modes must be a list")

    # Performance metrics (optional but recommended)
    if "metrics" in quality:
        if not isinstance(quality["metrics"], dict):
            errors.append("Quality metrics must be a dictionary")

    return len(errors) == 0, errors

def validate_integration(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate integration configuration (optional)"""
    errors = []

    if "integration" not in spec:
        return True, []  # Optional section

    integration = spec["integration"]

    # Claude Code Task tool
    if "claude_code" in integration:
        cc = integration["claude_code"]
        if not isinstance(cc, dict):
            errors.append("Integration.claude_code must be a dictionary")
        elif "task_template" not in cc:
            errors.append("Missing integration.claude_code.task_template")

    # Memory MCP
    if "memory_mcp" in integration:
        mem = integration["memory_mcp"]
        if not isinstance(mem, dict):
            errors.append("Integration.memory_mcp must be a dictionary")

    # Hooks
    if "hooks" in integration:
        hooks = integration["hooks"]
        if not isinstance(hooks, dict):
            errors.append("Integration.hooks must be a dictionary")

    return len(errors) == 0, errors

def validate_yaml_syntax(file_path: Path) -> Tuple[bool, List[str]]:
    """Validate YAML syntax"""
    errors = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        return True, []
    except yaml.YAMLError as e:
        return False, [f"Invalid YAML syntax: {e}"]
    except Exception as e:
        return False, [f"Error reading file: {e}"]

def main():
    parser = argparse.ArgumentParser(description="Validate agent specification")
    parser.add_argument("agent_spec", help="Path to agent YAML specification")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--strict", action="store_true", help="Enable strict validation")

    args = parser.parse_args()
    spec_path = Path(args.agent_spec)

    if not spec_path.exists():
        print(f"Error: File not found: {spec_path}", file=sys.stderr)
        return 1

    # Load and validate YAML
    syntax_valid, syntax_errors = validate_yaml_syntax(spec_path)
    if not syntax_valid:
        if args.json:
            print(json.dumps({"passed": False, "errors": syntax_errors}, indent=2))
        else:
            print("YAML SYNTAX ERROR:")
            for error in syntax_errors:
                print(f"  • {error}")
        return 1

    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = yaml.safe_load(f)

    # Run all validations
    results = {
        "metadata": validate_metadata(spec),
        "role": validate_role(spec),
        "capabilities": validate_capabilities(spec),
        "prompting": validate_prompting(spec),
        "quality": validate_quality(spec),
        "integration": validate_integration(spec)
    }

    all_passed = all(passed for passed, _ in results.values())

    if args.json:
        output = {
            "passed": all_passed,
            "checks": {
                name: {"passed": passed, "errors": errors}
                for name, (passed, errors) in results.items()
            }
        }
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "="*70)
        print("AGENT SPECIFICATION VALIDATION REPORT")
        print("="*70 + "\n")

        for name, (passed, errors) in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{name.upper()}: {status}")
            if errors:
                for error in errors:
                    print(f"  • {error}")
            print()

        print("="*70)
        if all_passed:
            print("✓ All validations passed - Agent specification is ready!")
            return 0
        else:
            print("✗ Some validations failed - Review errors above")
            return 1

if __name__ == "__main__":
    sys.exit(main())
