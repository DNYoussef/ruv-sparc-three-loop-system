#!/usr/bin/env python3
"""
MECE Gap Analysis Generator
Analyzes all skills against the universal MECE template
"""

import os
import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

REQUIRED_FILES = ['skill.md', 'README.md']
REQUIRED_DIRS = ['examples']
OPTIONAL_DIRS = ['references', 'resources', 'graphviz', 'tests']
OPTIONAL_SUBDIRS = {
    'resources': ['scripts', 'templates', 'assets']
}

QUALITY_TIERS = {
    'Bronze': {'files': 3, 'dirs': ['examples']},
    'Silver': {'files': 7, 'dirs': ['examples', 'references', 'graphviz']},
    'Gold': {'files': 12, 'dirs': ['examples', 'references', 'resources', 'graphviz', 'tests']},
    'Platinum': {'files': 20, 'dirs': ['examples', 'references', 'resources', 'graphviz', 'tests']}
}

class SkillAnalyzer:
    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self.skills = {}
        self.meta_skills = []
        self.gaps = defaultdict(list)

    def analyze_all(self):
        """Analyze all skills in the directory"""
        print(f"Analyzing skills in: {self.skills_dir}")

        for skill_path in self.skills_dir.iterdir():
            if not skill_path.is_dir() or skill_path.name.startswith(('_', '.')):
                continue

            analysis = self.analyze_skill(skill_path)
            if analysis:
                self.skills[skill_path.name] = analysis

                # Identify meta skills
                if self.is_meta_skill(skill_path.name, analysis):
                    self.meta_skills.append(skill_path.name)

        return self.generate_report()

    def analyze_skill(self, skill_path: Path) -> Dict:
        """Analyze a single skill"""
        # Find skill.md or SKILL.md
        skill_md = skill_path / 'skill.md'
        if not skill_md.exists():
            skill_md = skill_path / 'SKILL.md'
        if not skill_md.exists():
            return None

        analysis = {
            'path': str(skill_path),
            'has_skill_md': True,
            'has_readme': (skill_path / 'README.md').exists(),
            'has_examples': (skill_path / 'examples').is_dir() if (skill_path / 'examples').exists() else False,
            'has_references': (skill_path / 'references').is_dir() if (skill_path / 'references').exists() else False,
            'has_resources': (skill_path / 'resources').is_dir() if (skill_path / 'resources').exists() else False,
            'has_graphviz': (skill_path / 'graphviz').is_dir() if (skill_path / 'graphviz').exists() else False,
            'has_tests': (skill_path / 'tests').is_dir() if (skill_path / 'tests').exists() else False,
            'file_count': sum(1 for _ in skill_path.rglob('*') if _.is_file()),
            'missing_required': [],
            'missing_optional': [],
            'tier': 'Incomplete'
        }

        # Check required files
        if not analysis['has_readme']:
            analysis['missing_required'].append('README.md')
        if not analysis['has_examples']:
            analysis['missing_required'].append('examples/')

        # Check optional components
        if not analysis['has_references']:
            analysis['missing_optional'].append('references/')
        if not analysis['has_resources']:
            analysis['missing_optional'].append('resources/')
        if not analysis['has_graphviz']:
            analysis['missing_optional'].append('graphviz/')
        if not analysis['has_tests']:
            analysis['missing_optional'].append('tests/')

        # Determine tier
        analysis['tier'] = self.determine_tier(analysis)

        # Extract frontmatter if possible
        try:
            with open(skill_md, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        analysis['frontmatter'] = yaml.safe_load(parts[1])
        except Exception as e:
            analysis['frontmatter'] = None

        return analysis

    def determine_tier(self, analysis: Dict) -> str:
        """Determine quality tier"""
        has_all_required = len(analysis['missing_required']) == 0

        if not has_all_required:
            return 'Incomplete'
        elif analysis['file_count'] >= 20 and all([
            analysis['has_examples'],
            analysis['has_references'],
            analysis['has_resources'],
            analysis['has_graphviz'],
            analysis['has_tests']
        ]):
            return 'Platinum'
        elif analysis['file_count'] >= 12 and all([
            analysis['has_examples'],
            analysis['has_references'],
            analysis['has_resources']
        ]):
            return 'Gold'
        elif analysis['file_count'] >= 7 and all([
            analysis['has_examples'],
            analysis['has_references']
        ]):
            return 'Silver'
        elif analysis['file_count'] >= 3 and analysis['has_examples']:
            return 'Bronze'
        else:
            return 'Incomplete'

    def is_meta_skill(self, name: str, analysis: Dict) -> bool:
        """Check if skill is a meta skill (operates on other skills)"""
        meta_keywords = [
            'audit', 'test', 'verify', 'validate', 'quality',
            'forge', 'creator', 'generator', 'builder', 'analyzer'
        ]

        name_lower = name.lower()

        # Check name
        if any(keyword in name_lower for keyword in meta_keywords):
            return True

        # Check frontmatter description
        if analysis.get('frontmatter'):
            desc = str(analysis['frontmatter'].get('description', '')).lower()
            if 'skill' in desc and any(keyword in desc for keyword in meta_keywords):
                return True

        return False

    def generate_report(self) -> Dict:
        """Generate comprehensive MECE gap analysis report"""
        total_skills = len(self.skills)

        # Tier distribution
        tier_counts = defaultdict(int)
        for analysis in self.skills.values():
            tier_counts[analysis['tier']] += 1

        # Missing components count
        missing_readme = sum(1 for a in self.skills.values() if not a['has_readme'])
        missing_examples = sum(1 for a in self.skills.values() if not a['has_examples'])
        missing_references = sum(1 for a in self.skills.values() if not a['has_references'])
        missing_resources = sum(1 for a in self.skills.values() if not a['has_resources'])
        missing_graphviz = sum(1 for a in self.skills.values() if not a['has_graphviz'])
        missing_tests = sum(1 for a in self.skills.values() if not a['has_tests'])

        report = {
            'total_skills': total_skills,
            'meta_skills': sorted(self.meta_skills),
            'meta_skills_count': len(self.meta_skills),
            'tier_distribution': dict(tier_counts),
            'missing_components': {
                'README.md': missing_readme,
                'examples/': missing_examples,
                'references/': missing_references,
                'resources/': missing_resources,
                'graphviz/': missing_graphviz,
                'tests/': missing_tests
            },
            'completion_percentages': {
                'README.md': f"{((total_skills - missing_readme) / total_skills * 100):.1f}%",
                'examples/': f"{((total_skills - missing_examples) / total_skills * 100):.1f}%",
                'references/': f"{((total_skills - missing_references) / total_skills * 100):.1f}%",
                'resources/': f"{((total_skills - missing_resources) / total_skills * 100):.1f}%",
                'graphviz/': f"{((total_skills - missing_graphviz) / total_skills * 100):.1f}%",
                'tests/': f"{((total_skills - missing_tests) / total_skills * 100):.1f}%"
            },
            'enhancement_needed': {
                'to_bronze': sum(1 for a in self.skills.values() if a['tier'] == 'Incomplete'),
                'to_silver': sum(1 for a in self.skills.values() if a['tier'] in ['Incomplete', 'Bronze']),
                'to_gold': sum(1 for a in self.skills.values() if a['tier'] in ['Incomplete', 'Bronze', 'Silver']),
                'to_platinum': sum(1 for a in self.skills.values() if a['tier'] != 'Platinum')
            }
        }

        return report

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate MECE gap analysis for all skills')
    parser.add_argument('skills_dir', nargs='?', default='.', help='Skills directory path')
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    parser.add_argument('--detailed', action='store_true', help='Include per-skill details')

    args = parser.parse_args()

    analyzer = SkillAnalyzer(args.skills_dir)
    report = analyzer.analyze_all()

    if args.json:
        if args.detailed:
            report['skills'] = analyzer.skills
        print(json.dumps(report, indent=2))
    else:
        print("\n" + "="*80)
        print("MECE GAP ANALYSIS - All Skills vs Universal Template")
        print("="*80)
        print(f"\n[STATS] Total Skills Analyzed: {report['total_skills']}")
        print(f"[META] Meta Skills Identified: {report['meta_skills_count']}")

        print("\n[META SKILLS] Audit/Test/Improve:")
        for skill in report['meta_skills']:
            print(f"   - {skill}")

        print("\n[TIERS] Quality Tier Distribution:")
        for tier in ['Platinum', 'Gold', 'Silver', 'Bronze', 'Incomplete']:
            count = report['tier_distribution'].get(tier, 0)
            pct = (count / report['total_skills'] * 100) if report['total_skills'] > 0 else 0
            print(f"   {tier:12} {count:3} skills ({pct:5.1f}%)")

        print("\n[GAPS] Missing Components (MECE Analysis):")
        for component, count in report['missing_components'].items():
            pct = report['completion_percentages'][component]
            print(f"   {component:15} {count:3} skills missing ({100 - float(pct.rstrip('%')):5.1f}% incomplete)")

        print("\n[WORK] Enhancement Required:")
        print(f"   To Bronze:   {report['enhancement_needed']['to_bronze']:3} skills need basic structure")
        print(f"   To Silver:   {report['enhancement_needed']['to_silver']:3} skills need production quality")
        print(f"   To Gold:     {report['enhancement_needed']['to_gold']:3} skills need enterprise grade")
        print(f"   To Platinum: {report['enhancement_needed']['to_platinum']:3} skills need best-in-class")

        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("1. Enhance meta skills to Gold tier first")
        print("2. Run pilot enhancement on 10 sample skills")
        print("3. Process all skills in batches of 10")
        print("4. Target: Bronze tier for ALL skills (100%)")
        print("="*80 + "\n")

if __name__ == '__main__':
    main()
