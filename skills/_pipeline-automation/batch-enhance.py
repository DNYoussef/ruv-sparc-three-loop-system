#!/usr/bin/env python3
"""
Batch Enhancement Script
Processes multiple skills for enhancement in a single run
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict

class BatchEnhancer:
    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self.automation_dir = Path(__file__).parent

    def find_skills_needing_enhancement(self) -> List[str]:
        """Find all skills that need enhancement (missing README.md)"""
        skills = []
        for item in self.skills_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                readme = item / 'README.md'
                if not readme.exists():
                    skills.append(item.name)
        return sorted(skills)

    def generate_enhancement_plans(self, skills: List[str], target_tier: str = 'Silver') -> Dict:
        """Generate enhancement plans for a batch of skills"""
        print(f"\n[BATCH] Generating enhancement plans for {len(skills)} skills...")

        results = {
            'total': len(skills),
            'successful': 0,
            'failed': 0,
            'plans': {}
        }

        for skill in skills:
            try:
                skill_path = self.skills_dir / skill

                # Run enhance-skill.py
                cmd = [
                    sys.executable,
                    str(self.automation_dir / 'enhance-skill.py'),
                    str(skill_path),
                    '--tier', target_tier
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    results['successful'] += 1
                    results['plans'][skill] = 'success'
                    print(f"  [OK] {skill}")
                else:
                    results['failed'] += 1
                    results['plans'][skill] = 'failed'
                    print(f"  [X] {skill}: {result.stderr[:100]}")

            except Exception as e:
                results['failed'] += 1
                results['plans'][skill] = f'error: {str(e)}'
                print(f"  [X] {skill}: {str(e)}")

        return results

    def audit_batch(self, skills: List[str]) -> Dict:
        """Audit a batch of skills"""
        print(f"\n[BATCH] Auditing {len(skills)} skills...")

        results = {
            'total': len(skills),
            'passed': 0,
            'failed': 0,
            'scores': {},
            'decisions': {}
        }

        for skill in skills:
            try:
                skill_path = self.skills_dir / skill

                # Run audit-skill.py
                cmd = [
                    sys.executable,
                    str(self.automation_dir / 'audit-skill.py'),
                    str(skill_path)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                # Read audit JSON
                audit_file = self.automation_dir / 'audits' / f'{skill}-audit.json'
                if audit_file.exists():
                    with open(audit_file, 'r') as f:
                        audit_data = json.load(f)
                        score = audit_data.get('overall_score', 0) * 100
                        decision = audit_data.get('decision', 'NO-GO')

                        results['scores'][skill] = score
                        results['decisions'][skill] = decision

                        if decision == 'GO':
                            results['passed'] += 1
                            print(f"  [OK] {skill}: {score:.1f}% - {decision}")
                        else:
                            results['failed'] += 1
                            print(f"  [X] {skill}: {score:.1f}% - {decision}")
                else:
                    results['failed'] += 1
                    results['scores'][skill] = 0
                    results['decisions'][skill] = 'ERROR'
                    print(f"  [X] {skill}: Audit file not found")

            except Exception as e:
                results['failed'] += 1
                results['scores'][skill] = 0
                results['decisions'][skill] = 'ERROR'
                print(f"  [X] {skill}: {str(e)}")

        return results

    def cleanup_batch(self, skills: List[str]) -> Dict:
        """Run cleanup on a batch of skills"""
        print(f"\n[BATCH] Cleaning up {len(skills)} skills...")

        results = {
            'total': len(skills),
            'successful': 0,
            'failed': 0,
            'actions': {}
        }

        for skill in skills:
            try:
                skill_path = self.skills_dir / skill

                # Run cleanup-skill.py
                cmd = [
                    sys.executable,
                    str(self.automation_dir / 'cleanup-skill.py'),
                    str(skill_path)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    results['successful'] += 1
                    results['actions'][skill] = 'cleaned'
                    print(f"  [OK] {skill}")
                else:
                    results['failed'] += 1
                    results['actions'][skill] = 'failed'
                    print(f"  [X] {skill}")

            except Exception as e:
                results['failed'] += 1
                results['actions'][skill] = f'error: {str(e)}'
                print(f"  [X] {skill}: {str(e)}")

        return results

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch enhance multiple skills')
    parser.add_argument('--batch-size', type=int, default=15,
                       help='Number of skills to process in batch')
    parser.add_argument('--batch-num', type=int, default=1,
                       help='Which batch to process (1-based)')
    parser.add_argument('--tier', default='Silver',
                       choices=['Bronze', 'Silver', 'Gold', 'Platinum'],
                       help='Target quality tier')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list skills needing enhancement')
    parser.add_argument('--audit-only', action='store_true',
                       help='Only audit existing skills')
    parser.add_argument('--cleanup-only', action='store_true',
                       help='Only run cleanup on skills')

    args = parser.parse_args()

    # Initialize batch enhancer
    skills_dir = Path(__file__).parent.parent
    enhancer = BatchEnhancer(skills_dir)

    # Find skills needing enhancement
    all_skills = enhancer.find_skills_needing_enhancement()

    if args.list_only:
        print(f"\nSkills needing enhancement: {len(all_skills)}")
        for i, skill in enumerate(all_skills, 1):
            print(f"{i}. {skill}")
        return

    # Calculate batch range
    start_idx = (args.batch_num - 1) * args.batch_size
    end_idx = start_idx + args.batch_size
    batch_skills = all_skills[start_idx:end_idx]

    print(f"\n{'='*80}")
    print(f"BATCH {args.batch_num} PROCESSING")
    print(f"{'='*80}")
    print(f"Skills in batch: {len(batch_skills)}")
    print(f"Target tier: {args.tier}")
    print(f"Skills: {', '.join(batch_skills[:5])}{'...' if len(batch_skills) > 5 else ''}")
    print(f"{'='*80}\n")

    if args.audit_only:
        audit_results = enhancer.audit_batch(batch_skills)
        print(f"\n[RESULT] Audit complete: {audit_results['passed']}/{audit_results['total']} passed")
        return

    if args.cleanup_only:
        cleanup_results = enhancer.cleanup_batch(batch_skills)
        print(f"\n[RESULT] Cleanup complete: {cleanup_results['successful']}/{cleanup_results['total']} succeeded")
        return

    # Full batch processing
    # Step 1: Generate enhancement plans
    plan_results = enhancer.generate_enhancement_plans(batch_skills, args.tier)
    print(f"\n[RESULT] Plans generated: {plan_results['successful']}/{plan_results['total']}")

    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print("1. Review enhancement plans in _pipeline-automation/enhancements/")
    print("2. Create skill.md files for skills that need them (use skill-forge template)")
    print("3. Use Claude Code Task tool to spawn agents for parallel enhancement")
    print("4. Run: python batch-enhance.py --batch-num {args.batch_num} --cleanup-only")
    print("5. Run: python batch-enhance.py --batch-num {args.batch_num} --audit-only")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
