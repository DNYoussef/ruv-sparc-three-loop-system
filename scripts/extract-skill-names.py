#!/usr/bin/env python3
"""
Extract skill names and categories from all SKILL.md files
"""

import yaml
from pathlib import Path
from collections import defaultdict

def extract_skill_info(skill_file):
    """Extract name and category from SKILL.md frontmatter"""
    try:
        with open(skill_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract frontmatter
        if not content.startswith('---'):
            return None, None

        parts = content.split('---', 2)
        if len(parts) < 3:
            return None, None

        frontmatter = yaml.safe_load(parts[1].strip())
        name = frontmatter.get('name', '')
        category = frontmatter.get('category', 'uncategorized')

        return name, category
    except Exception as e:
        return None, None

def main():
    # Get plugin root
    script_dir = Path(__file__).parent
    plugin_root = script_dir.parent
    skills_dir = plugin_root / 'skills'

    # Find all SKILL.md files
    skill_files = list(skills_dir.rglob('SKILL.md'))

    # Extract skill info
    skills_by_category = defaultdict(list)
    all_skills = []

    for skill_file in skill_files:
        name, category = extract_skill_info(skill_file)
        if name:
            all_skills.append(name)
            skills_by_category[category].append(name)

    # Print organized by category
    print(f"# Complete Skill Inventory ({len(all_skills)} skills)")
    print()

    for category in sorted(skills_by_category.keys()):
        skills = sorted(skills_by_category[category])
        print(f"## {category.title()} ({len(skills)} skills)")
        for skill in skills:
            print(f"- {skill}")
        print()

    # Also save to file
    output_file = plugin_root / 'docs' / 'SKILLS-INVENTORY.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Complete Skill Inventory ({len(all_skills)} skills)\n\n")
        f.write("**Generated**: 2025-11-14\n")
        f.write(f"**Total Skills**: {len(all_skills)}\n")
        f.write(f"**Categories**: {len(skills_by_category)}\n\n")
        f.write("---\n\n")

        for category in sorted(skills_by_category.keys()):
            skills = sorted(skills_by_category[category])
            f.write(f"## {category.title()} ({len(skills)} skills)\n\n")
            for skill in skills:
                f.write(f"- `{skill}`\n")
            f.write("\n")

    print(f"\nInventory saved to: {output_file}")

if __name__ == '__main__':
    main()
