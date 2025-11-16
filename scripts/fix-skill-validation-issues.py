#!/usr/bin/env python3
"""
Auto-fix Script for SKILL.md Validation Issues

Automatically fixes common validation issues:
1. Converts names to kebab-case (lowercase with hyphens)
2. Adds missing YAML frontmatter
3. Standardizes optional fields (version, category, tags, author)

Usage:
    python scripts/fix-skill-validation-issues.py          # Dry-run mode
    python scripts/fix-skill-validation-issues.py --apply  # Apply fixes
"""

import os
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Standard optional fields to add to all skills
STANDARD_OPTIONAL_FIELDS = {
    'version': '1.0.0',
    'author': 'ruv',
    'tags': []  # Will be populated based on category
}

CATEGORY_TAG_MAPPINGS = {
    'delivery': ['delivery', 'development', 'workflow'],
    'foundry': ['foundry', 'creation', 'meta-tools'],
    'operations': ['operations', 'deployment', 'infrastructure'],
    'orchestration': ['orchestration', 'coordination', 'swarm'],
    'platforms': ['platforms', 'integration', 'tools'],
    'quality': ['quality', 'testing', 'validation'],
    'research': ['research', 'analysis', 'planning'],
    'security': ['security', 'compliance', 'safety'],
    'specialists': ['specialists', 'domain-expert']
}

class SkillFixer:
    """Fixes SKILL.md validation issues"""

    def __init__(self, skills_dir: Path, apply_fixes: bool = False):
        self.skills_dir = skills_dir
        self.apply_fixes = apply_fixes
        self.fixes_applied = []
        self.errors = []

    def to_kebab_case(self, text: str) -> str:
        """Convert text to kebab-case"""
        # Handle special cases first
        text = text.replace('&', 'and')
        text = text.replace(':', '')

        # Convert to lowercase and replace spaces with hyphens
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', '-', text)

        # Remove leading/trailing hyphens and multiple consecutive hyphens
        text = re.sub(r'-+', '-', text)
        text = text.strip('-')

        return text

    def infer_category(self, skill_path: Path) -> str:
        """Infer category from file path"""
        # Get the first directory after skills/
        parts = skill_path.parts
        skills_idx = parts.index('skills')
        if skills_idx + 1 < len(parts):
            return parts[skills_idx + 1]
        return 'general'

    def get_tags_for_category(self, category: str) -> List[str]:
        """Get standard tags for a category"""
        return CATEGORY_TAG_MAPPINGS.get(category, ['general'])

    def fix_skill_name(self, skill_file: Path) -> Optional[Dict]:
        """Fix skill name in YAML frontmatter"""
        try:
            content = skill_file.read_text(encoding='utf-8')
        except Exception as e:
            self.errors.append(f"Failed to read {skill_file}: {e}")
            return None

        # Check if file has YAML frontmatter
        if not content.startswith('---'):
            return self.add_missing_frontmatter(skill_file, content)

        # Extract frontmatter
        parts = content.split('---', 2)
        if len(parts) < 3:
            self.errors.append(f"Invalid frontmatter in {skill_file}")
            return None

        frontmatter_text = parts[1].strip()
        markdown_content = parts[2].strip()

        try:
            frontmatter = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML in {skill_file}: {e}")
            return None

        if not isinstance(frontmatter, dict):
            self.errors.append(f"Frontmatter is not a dict in {skill_file}")
            return None

        # Check if name needs fixing
        original_name = frontmatter.get('name', '')
        fixed_name = self.to_kebab_case(original_name)

        if original_name != fixed_name:
            frontmatter['name'] = fixed_name

            # Add standard optional fields if missing
            category = self.infer_category(skill_file)

            if 'version' not in frontmatter:
                frontmatter['version'] = STANDARD_OPTIONAL_FIELDS['version']

            if 'category' not in frontmatter:
                frontmatter['category'] = category

            if 'tags' not in frontmatter or not frontmatter['tags']:
                frontmatter['tags'] = self.get_tags_for_category(category)

            if 'author' not in frontmatter:
                frontmatter['author'] = STANDARD_OPTIONAL_FIELDS['author']

            # Rebuild file content
            new_frontmatter = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
            new_content = f"---\n{new_frontmatter}---\n\n{markdown_content}"

            return {
                'file': skill_file,
                'original_name': original_name,
                'fixed_name': fixed_name,
                'new_content': new_content,
                'added_fields': []
            }

        return None

    def add_missing_frontmatter(self, skill_file: Path, content: str) -> Optional[Dict]:
        """Add missing YAML frontmatter to a skill file"""
        # Try to infer name from file path or first heading
        name = skill_file.parent.name
        if not name or name == 'skills':
            # Try to get from first heading
            match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if match:
                name = match.group(1)
            else:
                name = "unnamed-skill"

        name = self.to_kebab_case(name)

        # Try to infer description from first paragraph
        description = "Skill description"
        lines = content.strip().split('\n')
        for line in lines[1:]:  # Skip first line (heading)
            line = line.strip()
            if line and not line.startswith('#'):
                description = line
                break

        # Build frontmatter
        category = self.infer_category(skill_file)
        frontmatter = {
            'name': name,
            'description': description,
            'version': STANDARD_OPTIONAL_FIELDS['version'],
            'category': category,
            'tags': self.get_tags_for_category(category),
            'author': STANDARD_OPTIONAL_FIELDS['author']
        }

        # Rebuild file content
        frontmatter_text = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        new_content = f"---\n{frontmatter_text}---\n\n{content}"

        return {
            'file': skill_file,
            'original_name': None,
            'fixed_name': name,
            'new_content': new_content,
            'added_fields': ['frontmatter']
        }

    def standardize_optional_fields(self, skill_file: Path) -> Optional[Dict]:
        """Add missing standard optional fields"""
        try:
            content = skill_file.read_text(encoding='utf-8')
        except Exception as e:
            self.errors.append(f"Failed to read {skill_file}: {e}")
            return None

        if not content.startswith('---'):
            return None  # Will be handled by add_missing_frontmatter

        parts = content.split('---', 2)
        if len(parts) < 3:
            return None

        frontmatter_text = parts[1].strip()
        markdown_content = parts[2].strip()

        try:
            frontmatter = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError:
            return None

        if not isinstance(frontmatter, dict):
            return None

        added_fields = []
        category = self.infer_category(skill_file)

        # Add missing fields
        if 'version' not in frontmatter:
            frontmatter['version'] = STANDARD_OPTIONAL_FIELDS['version']
            added_fields.append('version')

        if 'category' not in frontmatter:
            frontmatter['category'] = category
            added_fields.append('category')

        if 'tags' not in frontmatter or not frontmatter['tags']:
            frontmatter['tags'] = self.get_tags_for_category(category)
            added_fields.append('tags')

        if 'author' not in frontmatter:
            frontmatter['author'] = STANDARD_OPTIONAL_FIELDS['author']
            added_fields.append('author')

        if added_fields:
            # Rebuild file content
            new_frontmatter = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
            new_content = f"---\n{new_frontmatter}---\n\n{markdown_content}"

            return {
                'file': skill_file,
                'original_name': frontmatter.get('name'),
                'fixed_name': frontmatter.get('name'),
                'new_content': new_content,
                'added_fields': added_fields
            }

        return None

    def process_all_skills(self):
        """Process all SKILL.md files"""
        skill_files = list(self.skills_dir.rglob("SKILL.md"))
        total = len(skill_files)

        print(f"Processing {total} SKILL.md files...")
        print(f"Mode: {'APPLY FIXES' if self.apply_fixes else 'DRY RUN'}\n")

        for skill_file in skill_files:
            # Try to fix name issues
            fix = self.fix_skill_name(skill_file)
            if fix:
                self.fixes_applied.append(fix)
                if self.apply_fixes:
                    fix['file'].write_text(fix['new_content'], encoding='utf-8')
                    print(f"[FIXED] {fix['file'].relative_to(self.skills_dir)}")
                    print(f"        Name: '{fix['original_name']}' -> '{fix['fixed_name']}'")
                else:
                    print(f"[WOULD FIX] {fix['file'].relative_to(self.skills_dir)}")
                    print(f"            Name: '{fix['original_name']}' -> '{fix['fixed_name']}'")
                continue

            # Try to standardize optional fields
            fix = self.standardize_optional_fields(skill_file)
            if fix and fix['added_fields']:
                self.fixes_applied.append(fix)
                if self.apply_fixes:
                    fix['file'].write_text(fix['new_content'], encoding='utf-8')
                    print(f"[STANDARDIZED] {fix['file'].relative_to(self.skills_dir)}")
                    print(f"               Added fields: {', '.join(fix['added_fields'])}")
                else:
                    print(f"[WOULD STANDARDIZE] {fix['file'].relative_to(self.skills_dir)}")
                    print(f"                    Would add: {', '.join(fix['added_fields'])}")

    def print_summary(self):
        """Print summary of fixes"""
        print("\n" + "="*80)
        print("FIX SUMMARY")
        print("="*80)
        print(f"Total fixes: {len(self.fixes_applied)}")
        print(f"Errors: {len(self.errors)}")

        if self.apply_fixes:
            print("\nFixes have been APPLIED to files.")
            print("Run validation script to verify:")
            print("  python scripts/validate-all-skills.py")
        else:
            print("\nDRY RUN MODE - No files were modified.")
            print("To apply fixes, run:")
            print("  python scripts/fix-skill-validation-issues.py --apply")

        if self.errors:
            print("\nERRORS ENCOUNTERED:")
            for error in self.errors:
                print(f"  - {error}")

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
    apply_fixes = '--apply' in sys.argv

    # Create fixer
    fixer = SkillFixer(skills_dir, apply_fixes=apply_fixes)

    # Process all skills
    fixer.process_all_skills()

    # Print summary
    fixer.print_summary()

    # Exit code
    sys.exit(0 if len(fixer.errors) == 0 else 1)

if __name__ == "__main__":
    main()
