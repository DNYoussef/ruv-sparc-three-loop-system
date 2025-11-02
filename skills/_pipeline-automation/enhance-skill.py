#!/usr/bin/env python3
"""
Skill Enhancement Automation Script
Enhances a single skill to MECE universal template standards
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class SkillEnhancer:
    def __init__(self, skill_path: Path, target_tier: str = 'Bronze'):
        self.skill_path = Path(skill_path)
        self.skill_name = self.skill_path.name
        self.target_tier = target_tier
        self.enhancements = []

    def pre_enhancement_check(self) -> Dict:
        """
        Validate skill structure before enhancement begins.
        Returns validation status and issues found.
        """
        print(f"\n[VALIDATE] Running pre-enhancement checks on {self.skill_name}...")

        issues = []
        warnings = []

        # CRITICAL: Check if skill.md exists
        skill_md = self.skill_path / 'skill.md'
        skill_md_alt = self.skill_path / 'SKILL.md'

        if not skill_md.exists() and not skill_md_alt.exists():
            issues.append({
                'severity': 'CRITICAL',
                'category': 'missing_file',
                'message': 'skill.md (or SKILL.md) does not exist - REQUIRED for enhancement',
                'fix': 'Create skill.md before enhancement OR abort'
            })

        # Check for invalid directory names
        valid_dirs = {'examples', 'references', 'resources', 'graphviz', 'tests', '.claude-flow', '__pycache__'}
        for item in self.skill_path.iterdir():
            if item.is_dir() and item.name not in valid_dirs:
                # Check if it's a legacy skill directory (contains "when-" or "use-")
                if 'when-' in item.name or 'use-' in item.name:
                    issues.append({
                        'severity': 'MODERATE',
                        'category': 'invalid_directory',
                        'message': f'Invalid directory name: {item.name} (legacy structure)',
                        'fix': f'Remove or migrate content from {item.name}'
                    })

        # Identify orphaned files (files in root except allowed)
        allowed_root_files = {'skill.md', 'SKILL.md', 'README.md', '.gitignore'}
        orphaned_files = []

        for item in self.skill_path.iterdir():
            if item.is_file() and item.name not in allowed_root_files:
                orphaned_files.append({
                    'file': item.name,
                    'suggested_location': self._suggest_file_location(item.name)
                })

        if orphaned_files:
            warnings.append({
                'severity': 'MODERATE',
                'category': 'orphaned_files',
                'message': f'Found {len(orphaned_files)} orphaned file(s) in root',
                'files': orphaned_files
            })

        # Check for naming convention violations
        naming_issues = []
        for file_path in self.skill_path.rglob('*'):
            if file_path.is_file():
                # Check for uppercase in markdown files (except SKILL.md, README.md in root)
                if file_path.suffix == '.md':
                    if file_path.parent != self.skill_path:  # Not in root
                        if file_path.name != file_path.name.lower():
                            naming_issues.append({
                                'file': str(file_path.relative_to(self.skill_path)),
                                'issue': 'Uppercase letters in filename',
                                'expected': file_path.name.lower()
                            })

        if naming_issues:
            warnings.append({
                'severity': 'MINOR',
                'category': 'naming_conventions',
                'message': f'Found {len(naming_issues)} naming convention violation(s)',
                'files': naming_issues
            })

        # Determine if enhancement can proceed
        can_proceed = len([i for i in issues if i['severity'] == 'CRITICAL']) == 0

        validation = {
            'can_proceed': can_proceed,
            'critical_issues': len([i for i in issues if i['severity'] == 'CRITICAL']),
            'total_issues': len(issues),
            'total_warnings': len(warnings),
            'issues': issues,
            'warnings': warnings
        }

        # Print validation results
        if not can_proceed:
            print(f"[VALIDATE] [X] CRITICAL ISSUES FOUND - Cannot proceed with enhancement")
            for issue in issues:
                if issue['severity'] == 'CRITICAL':
                    print(f"  [X] {issue['message']}")
                    print(f"     Fix: {issue['fix']}")
        else:
            print(f"[VALIDATE] [OK] Pre-enhancement checks passed")
            if warnings:
                print(f"[VALIDATE] [!] Found {len(warnings)} warning(s) - will be cleaned during enhancement")

        return validation

    def _suggest_file_location(self, filename: str) -> str:
        """Suggest correct location for orphaned file"""
        if filename.endswith('.dot'):
            return 'graphviz/'
        elif filename.endswith('.py') or filename.endswith('.sh'):
            return 'resources/scripts/'
        elif filename.endswith('.json') or filename.endswith('.yaml') or filename.endswith('.yml'):
            return 'resources/templates/'
        elif 'enhancement' in filename.lower() or 'summary' in filename.lower():
            return 'references/ or DELETE (build artifact)'
        elif filename.endswith('.md'):
            return 'references/ or examples/'
        else:
            return 'resources/assets/'

    def analyze_current_state(self) -> Dict:
        """Analyze current skill structure"""
        print(f"\n[ANALYZE] Analyzing {self.skill_name}...")

        state = {
            'has_skill_md': False,
            'has_readme': False,
            'has_examples': False,
            'has_references': False,
            'has_resources': False,
            'has_graphviz': False,
            'has_tests': False,
            'file_count': 0,
            'current_tier': 'Incomplete',
            'missing_for_target': []
        }

        # Check existing components
        skill_md = self.skill_path / 'skill.md'
        if not skill_md.exists():
            skill_md = self.skill_path / 'SKILL.md'
        state['has_skill_md'] = skill_md.exists()

        state['has_readme'] = (self.skill_path / 'README.md').exists()
        state['has_examples'] = (self.skill_path / 'examples').is_dir() if (self.skill_path / 'examples').exists() else False
        state['has_references'] = (self.skill_path / 'references').is_dir() if (self.skill_path / 'references').exists() else False
        state['has_resources'] = (self.skill_path / 'resources').is_dir() if (self.skill_path / 'resources').exists() else False
        state['has_graphviz'] = (self.skill_path / 'graphviz').is_dir() if (self.skill_path / 'graphviz').exists() else False
        state['has_tests'] = (self.skill_path / 'tests').is_dir() if (self.skill_path / 'tests').exists() else False
        state['file_count'] = sum(1 for _ in self.skill_path.rglob('*') if _.is_file())

        # Determine current tier
        state['current_tier'] = self._determine_tier(state)

        # Calculate missing components for target tier
        state['missing_for_target'] = self._calculate_missing(state)

        return state

    def _determine_tier(self, state: Dict) -> str:
        """Determine quality tier based on MECE standards"""
        has_required = state['has_skill_md'] and state['has_readme'] and state['has_examples']

        if not has_required:
            return 'Incomplete'
        elif state['file_count'] >= 20 and all([
            state['has_examples'],
            state['has_references'],
            state['has_resources'],
            state['has_graphviz'],
            state['has_tests']
        ]):
            return 'Platinum'
        elif state['file_count'] >= 12 and all([
            state['has_examples'],
            state['has_references'],
            state['has_resources']
        ]):
            return 'Gold'
        elif state['file_count'] >= 7 and all([
            state['has_examples'],
            state['has_references']
        ]):
            return 'Silver'
        elif state['file_count'] >= 3 and state['has_examples']:
            return 'Bronze'
        else:
            return 'Incomplete'

    def _calculate_missing(self, state: Dict) -> List[str]:
        """Calculate what's missing for target tier"""
        missing = []

        # Required for all tiers
        if not state['has_skill_md']:
            missing.append('skill.md')
        if not state['has_readme']:
            missing.append('README.md')
        if not state['has_examples']:
            missing.append('examples/')

        # Target tier requirements
        if self.target_tier in ['Silver', 'Gold', 'Platinum']:
            if not state['has_references']:
                missing.append('references/')
            if not state['has_graphviz']:
                missing.append('graphviz/')

        if self.target_tier in ['Gold', 'Platinum']:
            if not state['has_resources']:
                missing.append('resources/scripts/')
                missing.append('resources/templates/')
            if not state['has_tests']:
                missing.append('tests/')

        return missing

    def create_enhancement_plan(self, state: Dict) -> Dict:
        """Create enhancement plan based on analysis"""
        print(f"\n[PLAN] Creating enhancement plan for {self.target_tier} tier...")

        plan = {
            'skill_name': self.skill_name,
            'current_tier': state['current_tier'],
            'target_tier': self.target_tier,
            'missing_components': state['missing_for_target'],
            'tasks': [],
            'estimated_time_hours': 0
        }

        # Generate tasks for missing components
        for component in state['missing_for_target']:
            if component == 'README.md':
                plan['tasks'].append({
                    'type': 'create_readme',
                    'agent': 'technical-writer',
                    'estimated_hours': 0.5
                })
            elif component == 'examples/':
                plan['tasks'].append({
                    'type': 'create_examples',
                    'agent': 'researcher',
                    'estimated_hours': 1.0,
                    'count': 3 if self.target_tier in ['Gold', 'Platinum'] else 1
                })
            elif component == 'references/':
                plan['tasks'].append({
                    'type': 'create_references',
                    'agent': 'technical-writer',
                    'estimated_hours': 0.5
                })
            elif component == 'graphviz/':
                plan['tasks'].append({
                    'type': 'create_graphviz',
                    'agent': 'architect',
                    'estimated_hours': 0.5
                })
            elif 'resources/' in component:
                plan['tasks'].append({
                    'type': 'create_resources',
                    'agent': 'coder',
                    'estimated_hours': 1.0
                })
            elif component == 'tests/':
                plan['tasks'].append({
                    'type': 'create_tests',
                    'agent': 'tester',
                    'estimated_hours': 0.5
                })

        plan['estimated_time_hours'] = sum(task['estimated_hours'] for task in plan['tasks'])

        return plan

    def generate_instructions(self, plan: Dict) -> str:
        """Generate agent instructions for enhancement"""
        print(f"\n[GENERATE] Creating agent instructions...")

        # Read skill.md to understand purpose
        skill_md = self.skill_path / 'skill.md'
        if not skill_md.exists():
            skill_md = self.skill_path / 'SKILL.md'

        skill_content = ""
        skill_purpose = ""

        if skill_md.exists():
            with open(skill_md, 'r', encoding='utf-8') as f:
                skill_content = f.read()
                # Extract first 200 chars for context
                skill_purpose = skill_content[:200].replace('\n', ' ')

        instructions = f"""# Enhancement Instructions for {self.skill_name}

**Target Tier**: {plan['target_tier']}
**Current Tier**: {plan['current_tier']}
**Estimated Time**: {plan['estimated_time_hours']} hours

## ⚠️ CRITICAL: Preserve Existing Files

**BEFORE starting any work**:
1. [!] DO NOT modify or delete existing skill.md/SKILL.md
2. [!] DO NOT modify or delete existing README.md (unless improving it)
3. [!] CREATE new files in proper MECE directories only
4. [!] FOLLOW file naming conventions (lowercase for all .md files except SKILL.md, README.md in root)

## Skill Purpose
{skill_purpose}...

## Tasks to Complete

"""

        for i, task in enumerate(plan['tasks'], 1):
            instructions += f"### Task {i}: {task['type']} (Agent: {task['agent']})\n\n"

            if task['type'] == 'create_readme':
                instructions += f"""Create README.md following MECE universal template:

**Location**: `{self.skill_name}/README.md`

**Required Sections**:
1. Title and one-line description
2. Quick Start (2-3 steps)
3. When to Use This Skill
4. Structure Overview
5. Examples (link to examples/)
6. Quality Tier: {plan['target_tier']}

**Template**: Reference skill-forge/README.md as example

"""

            elif task['type'] == 'create_examples':
                count = task.get('count', 1)
                instructions += f"""Create {count} example(s) in examples/ directory:

**Location**: `{self.skill_name}/examples/`

**Requirements**:
- example-1-basic.md: Simple, straightforward use case
"""
                if count >= 2:
                    instructions += "- example-2-advanced.md: Complex scenario with agent integration\n"
                if count >= 3:
                    instructions += "- example-3-edge-case.md: Edge case or specialized usage\n"

                instructions += """
**Format**: Step-by-step walkthrough with code samples

**Template**: Reference skill-forge/examples/ as model

"""

            elif task['type'] == 'create_references':
                instructions += f"""Create reference documentation in references/ directory:

**Location**: `{self.skill_name}/references/`

**Suggested Files**:
- best-practices.md: Guidelines and recommendations
- troubleshooting.md: Common issues and solutions
- related-skills.md: Links to complementary skills

**Content**: Abstract concepts, background knowledge, design decisions

"""

            elif task['type'] == 'create_graphviz':
                instructions += f"""Create GraphViz process diagram(s):

**Location**: `{self.skill_name}/graphviz/`

**Suggested Diagrams**:
- workflow.dot: Main process flow
- orchestration.dot (if multi-agent): Agent coordination pattern

**Format**: GraphViz DOT language with clear labels

**Template**: Reference skill-forge/graphviz/ as model

"""

            elif task['type'] == 'create_resources':
                instructions += f"""Create executable resources:

**Location**: `{self.skill_name}/resources/`

**Suggested Structure**:
- scripts/: Python/Shell scripts for deterministic operations
- templates/: YAML/JSON boilerplate files
- assets/: Images, configs, data files

**Quality**: Production-ready, well-documented code

"""

            elif task['type'] == 'create_tests':
                instructions += f"""Create test cases:

**Location**: `{self.skill_name}/tests/`

**Suggested Tests**:
- test-basic.md: Basic functionality validation
- test-edge-cases.md: Edge case scenarios
- test-integration.md (if applicable): Integration testing

**Format**: Markdown with expected outcomes

"""

        instructions += f"""
## Coordination Protocol

**BEFORE starting**:
```bash
npx claude-flow@alpha hooks pre-task --description "Enhance {self.skill_name} to {plan['target_tier']} tier"
```

**DURING work**:
```bash
npx claude-flow@alpha hooks post-edit --file "{{file}}" --memory-key "skill-enhancement-pipeline/{self.skill_name}/{{component}}"
```

**AFTER completion**:
```bash
npx claude-flow@alpha hooks post-task --task-id "enhance-{self.skill_name}"
npx claude-flow@alpha memory store --key "skill-enhancement-pipeline/{self.skill_name}/status" --value "enhanced-to-{plan['target_tier']}"
```

## Quality Checklist

- [ ] All missing components created
- [ ] MECE structure validated
- [ ] File naming conventions followed
- [ ] Examples are concrete and actionable
- [ ] References provide value-added context
- [ ] GraphViz diagrams are clear
- [ ] Resources are production-ready
- [ ] Tests cover key scenarios

## Success Criteria

**Tier Achievement**: {plan['target_tier']}
- File count: {self._get_tier_file_count(plan['target_tier'])}+ files
- All required directories present
- Quality validation passes (≥85%)

"""

        return instructions

    def _get_tier_file_count(self, tier: str) -> int:
        """Get minimum file count for tier"""
        tier_counts = {
            'Bronze': 3,
            'Silver': 7,
            'Gold': 12,
            'Platinum': 20
        }
        return tier_counts.get(tier, 3)

    def save_enhancement_plan(self, plan: Dict, instructions: str) -> Path:
        """Save enhancement plan and instructions"""
        output_dir = Path('_pipeline-automation/enhancements')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save plan as JSON
        plan_file = output_dir / f'{self.skill_name}-plan.json'
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2)

        # Save instructions as Markdown
        instructions_file = output_dir / f'{self.skill_name}-instructions.md'
        with open(instructions_file, 'w', encoding='utf-8') as f:
            f.write(instructions)

        print(f"\n[SAVE] Plan saved to {plan_file}")
        print(f"[SAVE] Instructions saved to {instructions_file}")

        return instructions_file

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Enhance a skill to MECE universal template standards')
    parser.add_argument('skill_path', help='Path to skill directory')
    parser.add_argument('--tier', choices=['Bronze', 'Silver', 'Gold', 'Platinum'],
                       default='Bronze', help='Target quality tier')
    parser.add_argument('--execute', action='store_true',
                       help='Execute enhancement (requires Claude Code Task tool)')

    args = parser.parse_args()

    enhancer = SkillEnhancer(args.skill_path, args.tier)

    # Phase 0: Pre-Enhancement Validation
    validation = enhancer.pre_enhancement_check()

    if not validation['can_proceed']:
        print(f"\n{'='*80}")
        print("[X] PRE-ENHANCEMENT VALIDATION FAILED")
        print(f"{'='*80}")
        print(f"\nCritical issues found: {validation['critical_issues']}")
        print(f"Total issues: {validation['total_issues']}")
        print(f"\nFix the critical issues above before enhancement can proceed.")
        print(f"{'='*80}\n")
        sys.exit(1)

    # Phase 1: Analyze
    state = enhancer.analyze_current_state()
    print(f"\n[RESULT] Current tier: {state['current_tier']}")
    print(f"[RESULT] Target tier: {args.tier}")
    print(f"[RESULT] Missing components: {', '.join(state['missing_for_target'])}")

    # Phase 2: Plan
    plan = enhancer.create_enhancement_plan(state)
    print(f"\n[RESULT] Enhancement tasks: {len(plan['tasks'])}")
    print(f"[RESULT] Estimated time: {plan['estimated_time_hours']} hours")

    # Phase 3: Generate Instructions
    instructions = enhancer.generate_instructions(plan)

    # Phase 4: Save
    instructions_file = enhancer.save_enhancement_plan(plan, instructions)

    print(f"\n{'='*80}")
    print("ENHANCEMENT PLAN READY")
    print(f"{'='*80}")
    print(f"\nNext Steps:")
    print(f"1. Review instructions: {instructions_file}")
    print(f"2. Use Claude Code Task tool to spawn enhancement agents")
    print(f"3. Run audit-skill.py for validation")
    print(f"{'='*80}\n")

    if args.execute:
        print("\n[EXECUTE] --execute flag requires Claude Code Task tool integration")
        print("[EXECUTE] Use the generated instructions with Task tool manually")

if __name__ == '__main__':
    main()
