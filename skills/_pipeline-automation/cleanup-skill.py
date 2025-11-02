#!/usr/bin/env python3
"""
Skill Cleanup Automation Script
Post-enhancement cleanup to fix orphaned files, naming conventions, and structure issues
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List

class SkillCleanup:
    def __init__(self, skill_path: Path):
        self.skill_path = Path(skill_path)
        self.skill_name = self.skill_path.name
        self.cleanup_actions = []

    def analyze_cleanup_needs(self) -> Dict:
        """Analyze what cleanup is needed"""
        print(f"\n[ANALYZE] Analyzing cleanup needs for {self.skill_name}...")

        cleanup_plan = {
            'orphaned_files': [],
            'naming_violations': [],
            'invalid_directories': [],
            'build_artifacts': [],
            'total_actions': 0
        }

        # Find orphaned files in root
        allowed_root_files = {'skill.md', 'SKILL.md', 'README.md', '.gitignore'}
        for item in self.skill_path.iterdir():
            if item.is_file() and item.name not in allowed_root_files:
                suggested_location = self._suggest_file_location(item.name)
                cleanup_plan['orphaned_files'].append({
                    'file': item.name,
                    'current': str(item),
                    'suggested': suggested_location,
                    'action': 'move'
                })

        # Find naming convention violations
        for file_path in self.skill_path.rglob('*'):
            if file_path.is_file() and file_path.suffix == '.md':
                # Skip root README.md and SKILL.md
                if file_path.parent != self.skill_path:
                    if file_path.name != file_path.name.lower():
                        new_name = file_path.name.lower()
                        cleanup_plan['naming_violations'].append({
                            'file': str(file_path.relative_to(self.skill_path)),
                            'current_name': file_path.name,
                            'new_name': new_name,
                            'action': 'rename'
                        })

        # Find invalid directories
        valid_dirs = {'examples', 'references', 'resources', 'graphviz', 'tests', '.claude-flow', '__pycache__'}
        for item in self.skill_path.iterdir():
            if item.is_dir() and item.name not in valid_dirs:
                if 'when-' in item.name or 'use-' in item.name:
                    cleanup_plan['invalid_directories'].append({
                        'directory': item.name,
                        'path': str(item),
                        'action': 'remove',
                        'reason': 'Legacy skill directory structure'
                    })

        # Find build artifacts
        build_artifact_dirs = {'.claude-flow', '__pycache__', '.pytest_cache', 'node_modules'}
        for artifact_dir in build_artifact_dirs:
            artifact_path = self.skill_path / artifact_dir
            if artifact_path.exists() and artifact_path.is_dir():
                cleanup_plan['build_artifacts'].append({
                    'directory': artifact_dir,
                    'path': str(artifact_path),
                    'action': 'remove'
                })

        cleanup_plan['total_actions'] = (
            len(cleanup_plan['orphaned_files']) +
            len(cleanup_plan['naming_violations']) +
            len(cleanup_plan['invalid_directories']) +
            len(cleanup_plan['build_artifacts'])
        )

        return cleanup_plan

    def _suggest_file_location(self, filename: str) -> str:
        """Suggest correct location for orphaned file"""
        if filename.endswith('.dot'):
            return 'graphviz/'
        elif filename.endswith('.py') or filename.endswith('.sh'):
            return 'resources/scripts/'
        elif filename.endswith('.json') or filename.endswith('.yaml') or filename.endswith('.yml'):
            return 'resources/templates/'
        elif 'enhancement' in filename.lower() or 'summary' in filename.lower():
            return 'DELETE (build artifact)'
        elif filename.endswith('.md'):
            return 'references/'
        else:
            return 'resources/assets/'

    def execute_cleanup(self, cleanup_plan: Dict, dry_run: bool = False) -> Dict:
        """Execute cleanup actions"""
        print(f"\n[CLEANUP] {'DRY RUN - ' if dry_run else ''}Executing cleanup actions...")

        results = {
            'files_moved': 0,
            'files_renamed': 0,
            'directories_removed': 0,
            'artifacts_removed': 0,
            'errors': []
        }

        # Move orphaned files
        for orphaned in cleanup_plan['orphaned_files']:
            try:
                source = Path(orphaned['current'])
                suggested = orphaned['suggested']

                if suggested == 'DELETE (build artifact)':
                    print(f"  [DEL] DELETE: {orphaned['file']}")
                    if not dry_run:
                        source.unlink()
                        results['files_moved'] += 1
                else:
                    target_dir = self.skill_path / suggested
                    target_file = target_dir / orphaned['file']

                    print(f"  [MOVE] {orphaned['file']} -> {suggested}")
                    if not dry_run:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(source), str(target_file))
                        results['files_moved'] += 1

            except Exception as e:
                results['errors'].append(f"Failed to move {orphaned['file']}: {str(e)}")

        # Fix naming violations
        for violation in cleanup_plan['naming_violations']:
            try:
                file_path = self.skill_path / violation['file']
                new_path = file_path.parent / violation['new_name']

                print(f"  [REN] RENAME: {violation['file']} -> {violation['new_name']}")
                if not dry_run:
                    file_path.rename(new_path)
                    results['files_renamed'] += 1

            except Exception as e:
                results['errors'].append(f"Failed to rename {violation['file']}: {str(e)}")

        # Remove invalid directories
        for invalid_dir in cleanup_plan['invalid_directories']:
            try:
                dir_path = Path(invalid_dir['path'])

                print(f"  [DEL] REMOVE DIR: {invalid_dir['directory']} ({invalid_dir['reason']})")
                if not dry_run:
                    shutil.rmtree(dir_path)
                    results['directories_removed'] += 1

            except Exception as e:
                results['errors'].append(f"Failed to remove {invalid_dir['directory']}: {str(e)}")

        # Remove build artifacts
        for artifact in cleanup_plan['build_artifacts']:
            try:
                artifact_path = Path(artifact['path'])

                print(f"  [DEL] REMOVE ARTIFACT: {artifact['directory']}")
                if not dry_run:
                    shutil.rmtree(artifact_path)
                    results['artifacts_removed'] += 1

            except Exception as e:
                results['errors'].append(f"Failed to remove {artifact['directory']}: {str(e)}")

        return results

    def validate_final_structure(self) -> Dict:
        """Validate final skill structure after cleanup"""
        print(f"\n[VALIDATE] Validating final structure...")

        validation = {
            'structure_valid': True,
            'issues': []
        }

        # Check for files in root (should only be allowed files)
        allowed_root_files = {'skill.md', 'SKILL.md', 'README.md', '.gitignore'}
        for item in self.skill_path.iterdir():
            if item.is_file() and item.name not in allowed_root_files:
                validation['structure_valid'] = False
                validation['issues'].append(f"Orphaned file still in root: {item.name}")

        # Check for invalid directories
        valid_dirs = {'examples', 'references', 'resources', 'graphviz', 'tests'}
        for item in self.skill_path.iterdir():
            if item.is_dir() and item.name not in valid_dirs and not item.name.startswith('.'):
                validation['structure_valid'] = False
                validation['issues'].append(f"Invalid directory: {item.name}")

        # Check naming conventions
        for file_path in self.skill_path.rglob('*.md'):
            if file_path.parent != self.skill_path:  # Not in root
                if file_path.name != file_path.name.lower():
                    validation['structure_valid'] = False
                    validation['issues'].append(f"Naming violation: {file_path.relative_to(self.skill_path)}")

        if validation['structure_valid']:
            print(f"[VALIDATE] [OK] Structure validation passed")
        else:
            print(f"[VALIDATE] [X] Structure validation failed - {len(validation['issues'])} issue(s)")

        return validation

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Cleanup skill after enhancement')
    parser.add_argument('skill_path', help='Path to skill directory')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be cleaned without making changes')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip final validation')

    args = parser.parse_args()

    cleanup = SkillCleanup(args.skill_path)

    # Phase 1: Analyze
    cleanup_plan = cleanup.analyze_cleanup_needs()
    print(f"\n[RESULT] Cleanup actions needed: {cleanup_plan['total_actions']}")
    print(f"[RESULT]   - Orphaned files: {len(cleanup_plan['orphaned_files'])}")
    print(f"[RESULT]   - Naming violations: {len(cleanup_plan['naming_violations'])}")
    print(f"[RESULT]   - Invalid directories: {len(cleanup_plan['invalid_directories'])}")
    print(f"[RESULT]   - Build artifacts: {len(cleanup_plan['build_artifacts'])}")

    if cleanup_plan['total_actions'] == 0:
        print(f"\n{'='*80}")
        print("[OK] NO CLEANUP NEEDED - Structure is already clean")
        print(f"{'='*80}\n")
        return

    # Phase 2: Execute
    results = cleanup.execute_cleanup(cleanup_plan, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\n[RESULT] Cleanup completed:")
        print(f"[RESULT]   - Files moved/deleted: {results['files_moved']}")
        print(f"[RESULT]   - Files renamed: {results['files_renamed']}")
        print(f"[RESULT]   - Directories removed: {results['directories_removed']}")
        print(f"[RESULT]   - Artifacts removed: {results['artifacts_removed']}")

        if results['errors']:
            print(f"[RESULT]   - Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"    [X] {error}")

        # Phase 3: Validate (unless skipped)
        if not args.no_validate:
            validation = cleanup.validate_final_structure()

            print(f"\n{'='*80}")
            if validation['structure_valid']:
                print("[OK] CLEANUP SUCCESSFUL - Structure validated")
            else:
                print("[!] CLEANUP COMPLETED WITH ISSUES")
                for issue in validation['issues']:
                    print(f"  [X] {issue}")
            print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("DRY RUN COMPLETED - No changes made")
        print(f"Run without --dry-run to execute cleanup")
        print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
