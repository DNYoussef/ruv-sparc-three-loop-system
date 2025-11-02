#!/usr/bin/env python3
"""
Skill Audit Automation Script
Validates skill against MECE universal template standards
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

class SkillAuditor:
    def __init__(self, skill_path: Path):
        self.skill_path = Path(skill_path)
        self.skill_name = self.skill_path.name
        self.violations = []
        self.warnings = []
        self.passes = []

    def audit_structure(self) -> Dict:
        """Audit MECE structure compliance"""
        print(f"\n[AUDIT] Structural validation for {self.skill_name}...")

        results = {
            'category': 'structure',
            'checks': [],
            'score': 0.0
        }

        checks = [
            ('skill.md exists', self._check_skill_md()),
            ('README.md exists', self._check_readme()),
            ('examples/ directory exists', self._check_examples_dir()),
            ('At least 1 example present', self._check_min_examples()),
            ('No files in root (except required)', self._check_no_root_files()),
            ('Proper directory naming (lowercase)', self._check_dir_naming()),
            ('No orphaned files', self._check_no_orphans()),
        ]

        for check_name, (passed, message) in checks:
            results['checks'].append({
                'name': check_name,
                'passed': passed,
                'message': message
            })
            if passed:
                self.passes.append(f"[STRUCTURE] {check_name}")
            else:
                self.violations.append(f"[STRUCTURE] {check_name}: {message}")

        results['score'] = sum(1 for _, (passed, _) in checks if passed) / len(checks)
        return results

    def audit_content(self) -> Dict:
        """Audit content quality"""
        print(f"\n[AUDIT] Content validation...")

        results = {
            'category': 'content',
            'checks': [],
            'score': 0.0
        }

        checks = [
            ('skill.md has YAML frontmatter', self._check_yaml_frontmatter()),
            ('skill.md uses imperative voice', self._check_imperative_voice()),
            ('README.md has overview', self._check_readme_overview()),
            ('README.md has quick start', self._check_readme_quickstart()),
            ('Examples are concrete (not abstract)', self._check_concrete_examples()),
            ('File naming conventions followed', self._check_file_naming()),
        ]

        for check_name, (passed, message) in checks:
            results['checks'].append({
                'name': check_name,
                'passed': passed,
                'message': message
            })
            if passed:
                self.passes.append(f"[CONTENT] {check_name}")
            else:
                self.violations.append(f"[CONTENT] {check_name}: {message}")

        results['score'] = sum(1 for _, (passed, _) in checks if passed) / len(checks)
        return results

    def audit_quality_tier(self) -> Dict:
        """Determine quality tier achievement"""
        print(f"\n[AUDIT] Quality tier validation...")

        has_skill_md = (self.skill_path / 'skill.md').exists() or (self.skill_path / 'SKILL.md').exists()
        has_readme = (self.skill_path / 'README.md').exists()
        has_examples = (self.skill_path / 'examples').is_dir()
        has_references = (self.skill_path / 'references').is_dir()
        has_resources = (self.skill_path / 'resources').is_dir()
        has_graphviz = (self.skill_path / 'graphviz').is_dir()
        has_tests = (self.skill_path / 'tests').is_dir()

        file_count = sum(1 for _ in self.skill_path.rglob('*') if _.is_file())

        tier = 'Incomplete'
        if has_skill_md and has_readme and has_examples:
            if file_count >= 20 and all([has_examples, has_references, has_resources, has_graphviz, has_tests]):
                tier = 'Platinum'
            elif file_count >= 12 and all([has_examples, has_references, has_resources]):
                tier = 'Gold'
            elif file_count >= 7 and all([has_examples, has_references]):
                tier = 'Silver'
            elif file_count >= 3:
                tier = 'Bronze'

        results = {
            'category': 'quality_tier',
            'tier': tier,
            'file_count': file_count,
            'components': {
                'skill.md': has_skill_md,
                'README.md': has_readme,
                'examples/': has_examples,
                'references/': has_references,
                'resources/': has_resources,
                'graphviz/': has_graphviz,
                'tests/': has_tests
            },
            'score': self._tier_to_score(tier)
        }

        return results

    def _tier_to_score(self, tier: str) -> float:
        """Convert tier to numeric score"""
        tier_scores = {
            'Platinum': 1.0,
            'Gold': 0.85,
            'Silver': 0.70,
            'Bronze': 0.60,
            'Incomplete': 0.0
        }
        return tier_scores.get(tier, 0.0)

    def _check_skill_md(self) -> Tuple[bool, str]:
        """Check skill.md exists"""
        skill_md = self.skill_path / 'skill.md'
        if not skill_md.exists():
            skill_md = self.skill_path / 'SKILL.md'

        if skill_md.exists():
            return (True, "skill.md found")
        return (False, "skill.md missing (required)")

    def _check_readme(self) -> Tuple[bool, str]:
        """Check README.md exists"""
        readme = self.skill_path / 'README.md'
        if readme.exists():
            return (True, "README.md found")
        return (False, "README.md missing (required)")

    def _check_examples_dir(self) -> Tuple[bool, str]:
        """Check examples/ directory exists"""
        examples = self.skill_path / 'examples'
        if examples.is_dir():
            return (True, "examples/ directory found")
        return (False, "examples/ directory missing (required)")

    def _check_min_examples(self) -> Tuple[bool, str]:
        """Check at least 1 example exists"""
        examples = self.skill_path / 'examples'
        if not examples.is_dir():
            return (False, "examples/ directory missing")

        example_files = list(examples.glob('*.md'))
        if len(example_files) >= 1:
            return (True, f"{len(example_files)} example(s) found")
        return (False, "No examples found (minimum 1 required)")

    def _check_no_root_files(self) -> Tuple[bool, str]:
        """Check no unnecessary files in root"""
        allowed_root = {'skill.md', 'SKILL.md', 'README.md', '.gitkeep'}
        root_files = [f.name for f in self.skill_path.iterdir() if f.is_file()]

        extra_files = [f for f in root_files if f not in allowed_root]
        if not extra_files:
            return (True, "No orphaned root files")
        return (False, f"Files should be categorized: {', '.join(extra_files)}")

    def _check_dir_naming(self) -> Tuple[bool, str]:
        """Check directory naming conventions"""
        dirs = [d for d in self.skill_path.iterdir() if d.is_dir()]

        # Standard MECE template directories
        valid_dirs = {'examples', 'references', 'resources', 'graphviz', 'tests', 'tmp'}

        # Acceptable project-specific directories (pre-existing, valid structure)
        acceptable_project_dirs = {
            'docs', 'scripts', 'templates', 'integrations', 'patterns',
            'wcag-accessibility', 'aws-specialist', 'kubernetes-specialist',
            'gcp-specialist', 'python-specialist', 'typescript-specialist',
            'react-specialist', 'sql-database-specialist', 'docker-containerization',
            'terraform-iac', 'opentelemetry-observability'
        }

        # Combine standard + project-specific
        all_valid_dirs = valid_dirs | acceptable_project_dirs

        invalid_dirs = [d.name for d in dirs if d.name not in all_valid_dirs and not d.name.startswith('.')]
        if not invalid_dirs:
            return (True, "Directory naming conventions followed")
        return (False, f"Invalid directory names: {', '.join(invalid_dirs)}")

    def _check_no_orphans(self) -> Tuple[bool, str]:
        """Check for orphaned files"""
        # Files that should be in specific directories
        problematic_extensions = {
            '.dot': 'graphviz/',
            '.py': 'resources/scripts/',
            '.sh': 'resources/scripts/',
            '.yaml': 'resources/templates/',
            '.json': 'resources/templates/'
        }

        orphans = []
        for file in self.skill_path.rglob('*'):
            if file.is_file():
                ext = file.suffix
                if ext in problematic_extensions:
                    # Check if file is in correct location
                    expected_parent = problematic_extensions[ext].rstrip('/')
                    if expected_parent not in str(file.parent):
                        orphans.append(f"{file.relative_to(self.skill_path)} → should be in {problematic_extensions[ext]}")

        if not orphans:
            return (True, "No orphaned files detected")
        return (False, f"Orphaned files: {'; '.join(orphans)}")

    def _check_yaml_frontmatter(self) -> Tuple[bool, str]:
        """Check skill.md has YAML frontmatter"""
        skill_md = self.skill_path / 'skill.md'
        if not skill_md.exists():
            skill_md = self.skill_path / 'SKILL.md'

        if not skill_md.exists():
            return (False, "skill.md not found")

        try:
            with open(skill_md, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        frontmatter = yaml.safe_load(parts[1])
                        if isinstance(frontmatter, dict):
                            return (True, "Valid YAML frontmatter found")
            return (False, "YAML frontmatter missing or invalid")
        except Exception as e:
            return (False, f"Error reading frontmatter: {str(e)}")

    def _check_imperative_voice(self) -> Tuple[bool, str]:
        """Check skill.md uses imperative voice"""
        skill_md = self.skill_path / 'skill.md'
        if not skill_md.exists():
            skill_md = self.skill_path / 'SKILL.md'

        if not skill_md.exists():
            return (False, "skill.md not found")

        try:
            with open(skill_md, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                # Skip frontmatter
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        content = parts[2]

                # Check for imperative verbs (sample)
                imperative_verbs = ['use', 'create', 'build', 'implement', 'analyze',
                                   'generate', 'execute', 'validate', 'ensure', 'follow']

                found_imperatives = sum(1 for verb in imperative_verbs if verb in content)
                if found_imperatives >= 3:
                    return (True, f"Imperative voice detected ({found_imperatives} verbs)")
                return (False, "Limited imperative voice usage (add more action verbs)")
        except Exception as e:
            return (False, f"Error reading skill.md: {str(e)}")

    def _check_readme_overview(self) -> Tuple[bool, str]:
        """Check README has overview section"""
        readme = self.skill_path / 'README.md'
        if not readme.exists():
            return (False, "README.md not found")

        try:
            with open(readme, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                if 'overview' in content or '# ' in content:
                    return (True, "Overview section found")
                return (False, "Add overview section to README.md")
        except Exception as e:
            return (False, f"Error reading README: {str(e)}")

    def _check_readme_quickstart(self) -> Tuple[bool, str]:
        """Check README has quick start"""
        readme = self.skill_path / 'README.md'
        if not readme.exists():
            return (False, "README.md not found")

        try:
            with open(readme, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                if 'quick start' in content or 'getting started' in content or 'usage' in content:
                    return (True, "Quick start section found")
                return (False, "Add quick start section to README.md")
        except Exception as e:
            return (False, f"Error reading README: {str(e)}")

    def _check_concrete_examples(self) -> Tuple[bool, str]:
        """Check examples are concrete"""
        examples = self.skill_path / 'examples'
        if not examples.is_dir():
            return (False, "examples/ directory not found")

        example_files = list(examples.glob('*.md'))
        if not example_files:
            return (False, "No examples found")

        # Check first example for concrete patterns
        try:
            with open(example_files[0], 'r', encoding='utf-8') as f:
                content = f.read().lower()
                concrete_indicators = ['scenario', 'step', 'example', '```', 'output']
                found = sum(1 for indicator in concrete_indicators if indicator in content)

                if found >= 3:
                    return (True, f"Examples appear concrete ({found} indicators)")
                return (False, "Examples should be more concrete (add scenarios, steps, code)")
        except Exception as e:
            return (False, f"Error reading examples: {str(e)}")

    def _check_file_naming(self) -> Tuple[bool, str]:
        """Check file naming conventions"""
        violations = []

        for file in self.skill_path.rglob('*'):
            if file.is_file():
                name = file.name
                # Check for spaces (should be hyphens)
                if ' ' in name:
                    violations.append(f"{file.relative_to(self.skill_path)}: contains spaces")

                # Check for uppercase in non-README files
                if name not in ['README.md', 'SKILL.md'] and name != name.lower():
                    violations.append(f"{file.relative_to(self.skill_path)}: should be lowercase")

        if not violations:
            return (True, "File naming conventions followed")
        return (False, f"Naming issues: {'; '.join(violations[:3])}")  # Show first 3

    def generate_report(self, structure_results: Dict, content_results: Dict,
                       tier_results: Dict) -> Dict:
        """Generate comprehensive audit report"""
        overall_score = (
            structure_results['score'] * 0.4 +
            content_results['score'] * 0.3 +
            tier_results['score'] * 0.3
        )

        report = {
            'skill_name': self.skill_name,
            'audit_timestamp': datetime.now().isoformat(),
            'overall_score': round(overall_score, 3),
            'quality_tier': tier_results['tier'],
            'decision': 'GO' if overall_score >= 0.85 else 'NO-GO',
            'categories': {
                'structure': structure_results,
                'content': content_results,
                'quality_tier': tier_results
            },
            'summary': {
                'total_checks': len(self.passes) + len(self.violations),
                'passed': len(self.passes),
                'failed': len(self.violations),
                'violations': self.violations,
                'passes': self.passes
            }
        }

        return report

    def save_report(self, report: Dict) -> Path:
        """Save audit report"""
        output_dir = Path('_pipeline-automation/audits')
        output_dir.mkdir(parents=True, exist_ok=True)

        report_file = output_dir / f'{self.skill_name}-audit.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"\n[SAVE] Audit report saved to {report_file}")
        return report_file

    def print_summary(self, report: Dict):
        """Print audit summary"""
        print(f"\n{'='*80}")
        print(f"AUDIT REPORT: {self.skill_name}")
        print(f"{'='*80}")
        print(f"\n[RESULT] Overall Score: {report['overall_score']:.1%}")
        print(f"[RESULT] Quality Tier: {report['quality_tier']}")
        print(f"[RESULT] Decision: {report['decision']}")

        print(f"\n[SCORES] Category Breakdown:")
        for category, results in report['categories'].items():
            if 'score' in results:
                print(f"   {category.replace('_', ' ').title():20} {results['score']:.1%}")

        print(f"\n[CHECKS] Summary:")
        print(f"   Passed:  {report['summary']['passed']:3} checks")
        print(f"   Failed:  {report['summary']['failed']:3} checks")

        if report['summary']['violations']:
            print(f"\n[VIOLATIONS] Issues Found:")
            for violation in report['summary']['violations'][:5]:  # Show first 5
                print(f"   - {violation}")
            if len(report['summary']['violations']) > 5:
                print(f"   ... and {len(report['summary']['violations']) - 5} more")

        print(f"\n{'='*80}")
        if report['decision'] == 'GO':
            print("STATUS: APPROVED - Skill meets quality standards")
        else:
            print("STATUS: REJECTED - Skill needs improvement")
            print(f"Required Score: 85% | Current: {report['overall_score']:.1%}")
        print(f"{'='*80}\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Audit skill against MECE universal template standards')
    parser.add_argument('skill_path', help='Path to skill directory')
    parser.add_argument('--json', action='store_true', help='Output JSON only (no summary)')

    args = parser.parse_args()

    auditor = SkillAuditor(args.skill_path)

    # Run audits
    structure_results = auditor.audit_structure()
    content_results = auditor.audit_content()
    tier_results = auditor.audit_quality_tier()

    # Generate report
    report = auditor.generate_report(structure_results, content_results, tier_results)

    # Save report
    auditor.save_report(report)

    # Print summary
    if not args.json:
        auditor.print_summary(report)
    else:
        print(json.dumps(report, indent=2))

    # Exit with appropriate code
    sys.exit(0 if report['decision'] == 'GO' else 1)

if __name__ == '__main__':
    main()
