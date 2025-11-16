#!/usr/bin/env python3
"""
Comprehensive SKILL.md Validation Script
Validates all 122 skills against Claude Code specifications

Requirements:
- Python 3.7+
- PyYAML (pip install pyyaml)

Usage:
    python scripts/validate-all-skills.py
    python scripts/validate-all-skills.py --fix  # Auto-fix issues
    python scripts/validate-all-skills.py --report # Generate JSON report
"""

import os
import re
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

@dataclass
class ValidationResult:
    """Result of validating a single skill"""
    skill_path: str
    skill_name: str
    valid: bool
    errors: List[str]
    warnings: List[str]
    info: Dict[str, any]

class SkillValidator:
    """Validates SKILL.md files against Claude Code specifications"""

    # Official Claude Code specs from https://code.claude.com/docs/en/skills
    MAX_NAME_LENGTH = 64
    MAX_DESCRIPTION_LENGTH = 1024
    NAME_PATTERN = re.compile(r'^[a-z0-9-]+$')

    # XML tags are not allowed
    XML_PATTERN = re.compile(r'<[^>]+>')

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.results: List[ValidationResult] = []

    def find_all_skills(self) -> List[Path]:
        """Find all SKILL.md files in the skills directory"""
        skill_files = []
        for skill_file in self.skills_dir.rglob("SKILL.md"):
            skill_files.append(skill_file)
        return sorted(skill_files)

    def validate_skill(self, skill_file: Path) -> ValidationResult:
        """Validate a single SKILL.md file"""
        errors = []
        warnings = []
        info = {}

        try:
            content = skill_file.read_text(encoding='utf-8')
        except Exception as e:
            return ValidationResult(
                skill_path=str(skill_file),
                skill_name="UNKNOWN",
                valid=False,
                errors=[f"Failed to read file: {e}"],
                warnings=[],
                info={}
            )

        # Check for YAML frontmatter
        if not content.startswith('---'):
            errors.append("Missing opening '---' delimiter on line 1")
            return ValidationResult(
                skill_path=str(skill_file),
                skill_name="UNKNOWN",
                valid=False,
                errors=errors,
                warnings=warnings,
                info=info
            )

        # Extract frontmatter
        parts = content.split('---', 2)
        if len(parts) < 3:
            errors.append("Missing closing '---' delimiter for YAML frontmatter")
            return ValidationResult(
                skill_path=str(skill_file),
                skill_name="UNKNOWN",
                valid=False,
                errors=errors,
                warnings=warnings,
                info=info
            )

        frontmatter_text = parts[1].strip()
        markdown_content = parts[2].strip()

        # Parse YAML
        try:
            frontmatter = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML syntax: {e}")
            return ValidationResult(
                skill_path=str(skill_file),
                skill_name="UNKNOWN",
                valid=False,
                errors=errors,
                warnings=warnings,
                info=info
            )

        # Validate required fields
        if not isinstance(frontmatter, dict):
            errors.append("Frontmatter must be a YAML object")
            return ValidationResult(
                skill_path=str(skill_file),
                skill_name="UNKNOWN",
                valid=False,
                errors=errors,
                warnings=warnings,
                info=info
            )

        # Check required field: name
        name = frontmatter.get('name')
        if not name:
            errors.append("Missing required field: 'name'")
        else:
            info['name'] = name

            # Validate name format
            if len(name) > self.MAX_NAME_LENGTH:
                errors.append(f"Name exceeds {self.MAX_NAME_LENGTH} characters: {len(name)} chars")

            if not self.NAME_PATTERN.match(name):
                errors.append(f"Name must use lowercase letters, numbers, and hyphens only: '{name}'")

            if self.XML_PATTERN.search(name):
                errors.append("Name contains XML tags (not allowed)")

        # Check required field: description
        description = frontmatter.get('description')
        if not description:
            errors.append("Missing required field: 'description'")
        else:
            info['description'] = description

            # Validate description length
            if len(description) > self.MAX_DESCRIPTION_LENGTH:
                errors.append(f"Description exceeds {self.MAX_DESCRIPTION_LENGTH} characters: {len(description)} chars")

            if self.XML_PATTERN.search(description):
                errors.append("Description contains XML tags (not allowed)")

            # Check for usage triggers (best practice)
            usage_keywords = ['use when', 'use this', 'use for', 'activate when', 'trigger']
            has_usage_trigger = any(keyword in description.lower() for keyword in usage_keywords)
            if not has_usage_trigger:
                warnings.append("Description should include usage triggers (e.g., 'Use when...')")

        # Collect optional fields
        optional_fields = {}
        for field in ['version', 'category', 'tags', 'author']:
            if field in frontmatter:
                optional_fields[field] = frontmatter[field]

        if optional_fields:
            info['optional_fields'] = optional_fields

        # Check for tabs (should use spaces)
        if '\t' in frontmatter_text:
            warnings.append("YAML frontmatter uses tabs (should use spaces for indentation)")

        # Check markdown content exists
        if not markdown_content:
            warnings.append("No markdown content after frontmatter")
        else:
            info['content_length'] = len(markdown_content)

        # Determine if valid
        valid = len(errors) == 0

        return ValidationResult(
            skill_path=str(skill_file),
            skill_name=name or "UNKNOWN",
            valid=valid,
            errors=errors,
            warnings=warnings,
            info=info
        )

    def validate_all(self) -> Tuple[int, int, int]:
        """
        Validate all skills
        Returns: (total, passed, failed)
        """
        skill_files = self.find_all_skills()
        total = len(skill_files)

        print(f"Found {total} SKILL.md files to validate\n")

        for skill_file in skill_files:
            result = self.validate_skill(skill_file)
            self.results.append(result)

        passed = sum(1 for r in self.results if r.valid)
        failed = total - passed

        return total, passed, failed

    def print_summary(self):
        """Print validation summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.valid)
        failed = total - passed

        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Total Skills:  {total}")
        print(f"Passed:        {passed} ({100*passed/total:.1f}%)")
        print(f"Failed:        {failed} ({100*failed/total:.1f}%)")
        print("="*80)

        if failed > 0:
            print("\nFAILED SKILLS:")
            print("-"*80)
            for result in self.results:
                if not result.valid:
                    print(f"\n{result.skill_name} - {result.skill_path}")
                    for error in result.errors:
                        print(f"  ERROR: {error}")
                    for warning in result.warnings:
                        print(f"  WARN:  {warning}")

        # Count warnings
        total_warnings = sum(len(r.warnings) for r in self.results)
        if total_warnings > 0:
            print(f"\nTotal Warnings: {total_warnings}")
            print("(Skills with warnings still pass validation)")

    def print_detailed_results(self):
        """Print detailed results for all skills"""
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)

        for result in self.results:
            status = "PASS" if result.valid else "FAIL"
            print(f"\n[{status}] {result.skill_name}")
            print(f"       {result.skill_path}")

            if result.errors:
                for error in result.errors:
                    print(f"  ERROR: {error}")

            if result.warnings:
                for warning in result.warnings:
                    print(f"  WARN:  {warning}")

            if result.info and result.valid:
                if 'optional_fields' in result.info:
                    fields = ', '.join(result.info['optional_fields'].keys())
                    print(f"  INFO:  Optional fields: {fields}")

    def analyze_optional_fields(self) -> Dict[str, Dict]:
        """Analyze usage of optional YAML fields across all skills"""
        field_usage = {
            'version': {'count': 0, 'skills': []},
            'category': {'count': 0, 'skills': []},
            'tags': {'count': 0, 'skills': []},
            'author': {'count': 0, 'skills': []},
        }

        for result in self.results:
            if result.valid and 'optional_fields' in result.info:
                for field in field_usage.keys():
                    if field in result.info['optional_fields']:
                        field_usage[field]['count'] += 1
                        field_usage[field]['skills'].append(result.skill_name)

        return field_usage

    def print_field_analysis(self):
        """Print analysis of optional field usage"""
        field_usage = self.analyze_optional_fields()
        total = len(self.results)

        print("\n" + "="*80)
        print("OPTIONAL FIELD USAGE ANALYSIS")
        print("="*80)

        for field, data in field_usage.items():
            count = data['count']
            percentage = 100 * count / total if total > 0 else 0
            print(f"\n{field}:")
            print(f"  Used by: {count}/{total} skills ({percentage:.1f}%)")

            if count == 0:
                print(f"  Recommendation: Consider adding '{field}' to all skills for consistency")
            elif count < total:
                print(f"  Recommendation: Standardize by adding '{field}' to remaining {total-count} skills")
            else:
                print(f"  Status: All skills have this field")

    def generate_json_report(self, output_file: Path):
        """Generate JSON report"""
        report = {
            'summary': {
                'total': len(self.results),
                'passed': sum(1 for r in self.results if r.valid),
                'failed': sum(1 for r in self.results if not r.valid),
                'warnings': sum(len(r.warnings) for r in self.results)
            },
            'field_usage': self.analyze_optional_fields(),
            'results': [asdict(r) for r in self.results]
        }

        output_file.write_text(json.dumps(report, indent=2))
        print(f"\nJSON report saved to: {output_file}")

def main():
    """Main entry point"""
    # Get script directory
    script_dir = Path(__file__).parent
    plugin_root = script_dir.parent
    skills_dir = plugin_root / "skills"

    if not skills_dir.exists():
        print(f"Error: Skills directory not found: {skills_dir}")
        sys.exit(1)

    # Parse arguments
    generate_report = '--report' in sys.argv
    detailed = '--detailed' in sys.argv or '-v' in sys.argv

    # Create validator
    validator = SkillValidator(skills_dir)

    # Run validation
    print("Claude Code SKILL.md Validator")
    print(f"Skills Directory: {skills_dir}\n")

    total, passed, failed = validator.validate_all()

    # Print results
    if detailed:
        validator.print_detailed_results()

    validator.print_summary()
    validator.print_field_analysis()

    # Generate report if requested
    if generate_report:
        report_file = plugin_root / "docs" / "skill-validation-report.json"
        report_file.parent.mkdir(exist_ok=True)
        validator.generate_json_report(report_file)

    # Exit code
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
