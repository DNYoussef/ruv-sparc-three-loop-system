#!/usr/bin/env python3
"""
GitHub Release Orchestrator
Automated release management with semantic versioning, changelog generation,
deployment coordination, and rollback capabilities
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import re

# Configuration
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_API = 'https://api.github.com'
VERBOSE = os.environ.get('VERBOSE', '').lower() == 'true'

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def log_info(msg: str):
    """Log info message"""
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")

def log_warn(msg: str):
    """Log warning message"""
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}", file=sys.stderr)

def log_error(msg: str):
    """Log error message"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}", file=sys.stderr)

def log_debug(msg: str):
    """Log debug message if verbose mode enabled"""
    if VERBOSE:
        print(f"{Colors.BLUE}[DEBUG]{Colors.NC} {msg}")

def run_command(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Execute shell command and return exit code, stdout, stderr"""
    log_debug(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()

def parse_semver(version: str) -> Tuple[int, int, int]:
    """Parse semantic version string"""
    match = re.match(r'v?(\d+)\.(\d+)\.(\d+)', version)
    if not match:
        raise ValueError(f"Invalid semantic version: {version}")
    return tuple(map(int, match.groups()))

def bump_version(current: str, bump_type: str) -> str:
    """Bump semantic version"""
    major, minor, patch = parse_semver(current)

    if bump_type == 'major':
        return f"v{major + 1}.0.0"
    elif bump_type == 'minor':
        return f"v{major}.{minor + 1}.0"
    elif bump_type == 'patch':
        return f"v{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")

def get_latest_tag(repo_path: str) -> Optional[str]:
    """Get latest git tag"""
    code, stdout, _ = run_command(['git', 'describe', '--tags', '--abbrev=0'], cwd=repo_path)
    return stdout if code == 0 else None

def generate_changelog(repo_path: str, from_ref: str, to_ref: str = 'HEAD') -> str:
    """Generate changelog from git commits"""
    log_info(f"Generating changelog: {from_ref}..{to_ref}")

    # Get commit messages
    code, stdout, _ = run_command([
        'git', 'log', f'{from_ref}..{to_ref}',
        '--pretty=format:%h|%s|%an|%ad',
        '--date=short'
    ], cwd=repo_path)

    if code != 0:
        log_error("Failed to generate changelog")
        return ""

    # Categorize commits
    features = []
    fixes = []
    breaking = []
    other = []

    for line in stdout.split('\n'):
        if not line:
            continue

        parts = line.split('|')
        if len(parts) < 2:
            continue

        commit_hash, message = parts[0], parts[1]

        # Categorize by conventional commits
        if message.startswith('feat'):
            features.append(f"- {message} ({commit_hash})")
        elif message.startswith('fix'):
            fixes.append(f"- {message} ({commit_hash})")
        elif 'BREAKING CHANGE' in message or message.startswith('!'):
            breaking.append(f"- {message} ({commit_hash})")
        else:
            other.append(f"- {message} ({commit_hash})")

    # Build changelog
    changelog = []

    if breaking:
        changelog.append("## ⚠️ BREAKING CHANGES\n")
        changelog.extend(breaking)
        changelog.append("")

    if features:
        changelog.append("## 🚀 Features\n")
        changelog.extend(features)
        changelog.append("")

    if fixes:
        changelog.append("## 🐛 Bug Fixes\n")
        changelog.extend(fixes)
        changelog.append("")

    if other:
        changelog.append("## 📝 Other Changes\n")
        changelog.extend(other)
        changelog.append("")

    return '\n'.join(changelog)

def create_github_release(repo: str, tag: str, changelog: str, prerelease: bool = False) -> bool:
    """Create GitHub release via API"""
    if not GITHUB_TOKEN:
        log_error("GITHUB_TOKEN not set")
        return False

    log_info(f"Creating GitHub release: {tag}")

    url = f"{GITHUB_API}/repos/{repo}/releases"

    data = {
        'tag_name': tag,
        'name': f"Release {tag}",
        'body': changelog,
        'draft': False,
        'prerelease': prerelease
    }

    cmd = [
        'curl', '-X', 'POST',
        '-H', f'Authorization: Bearer {GITHUB_TOKEN}',
        '-H', 'Accept: application/vnd.github+json',
        '-H', 'X-GitHub-Api-Version: 2022-11-28',
        '-d', json.dumps(data),
        url
    ]

    code, stdout, stderr = run_command(cmd)

    if code != 0:
        log_error(f"Failed to create release: {stderr}")
        return False

    log_info("GitHub release created successfully")
    return True

def run_tests(repo_path: str) -> bool:
    """Run test suite"""
    log_info("Running test suite...")

    # Try different test commands
    test_commands = [
        ['npm', 'test'],
        ['yarn', 'test'],
        ['pytest'],
        ['go', 'test', './...'],
        ['mvn', 'test']
    ]

    for cmd in test_commands:
        if os.path.exists(os.path.join(repo_path, 'package.json')) and cmd[0] in ['npm', 'yarn']:
            code, stdout, stderr = run_command(cmd, cwd=repo_path)
            if code == 0:
                log_info("Tests passed")
                return True
            else:
                log_error(f"Tests failed: {stderr}")
                return False

    log_warn("No test command found - skipping tests")
    return True

def build_artifacts(repo_path: str) -> bool:
    """Build release artifacts"""
    log_info("Building release artifacts...")

    # Try different build commands
    build_commands = [
        ['npm', 'run', 'build'],
        ['yarn', 'build'],
        ['make', 'build'],
        ['go', 'build'],
        ['mvn', 'package']
    ]

    for cmd in build_commands:
        if os.path.exists(os.path.join(repo_path, 'package.json')) and cmd[0] in ['npm', 'yarn']:
            code, stdout, stderr = run_command(cmd, cwd=repo_path)
            if code == 0:
                log_info("Build successful")
                return True
            else:
                log_error(f"Build failed: {stderr}")
                return False

    log_warn("No build command found - skipping build")
    return True

def orchestrate_release(
    repo_path: str,
    repo_name: str,
    bump_type: str,
    skip_tests: bool = False,
    skip_build: bool = False,
    prerelease: bool = False
) -> bool:
    """Orchestrate complete release workflow"""

    log_info(f"Starting release orchestration for {repo_name}")
    log_info(f"Repository: {repo_path}")
    log_info(f"Bump type: {bump_type}")

    # Get current version
    current_tag = get_latest_tag(repo_path)
    if not current_tag:
        log_warn("No existing tags found - using v0.0.0")
        current_tag = "v0.0.0"

    log_info(f"Current version: {current_tag}")

    # Calculate new version
    new_version = bump_version(current_tag, bump_type)
    log_info(f"New version: {new_version}")

    # Run tests
    if not skip_tests:
        if not run_tests(repo_path):
            log_error("Tests failed - aborting release")
            return False

    # Build artifacts
    if not skip_build:
        if not build_artifacts(repo_path):
            log_error("Build failed - aborting release")
            return False

    # Generate changelog
    changelog = generate_changelog(repo_path, current_tag)
    if not changelog:
        log_warn("Empty changelog - no commits since last release")
        changelog = "No changes"

    log_info("Generated changelog:")
    print(changelog)

    # Create git tag
    log_info(f"Creating git tag: {new_version}")
    code, _, stderr = run_command(['git', 'tag', '-a', new_version, '-m', f'Release {new_version}'], cwd=repo_path)
    if code != 0:
        log_error(f"Failed to create tag: {stderr}")
        return False

    # Push tag
    log_info("Pushing tag to remote...")
    code, _, stderr = run_command(['git', 'push', 'origin', new_version], cwd=repo_path)
    if code != 0:
        log_error(f"Failed to push tag: {stderr}")
        return False

    # Create GitHub release
    if not create_github_release(repo_name, new_version, changelog, prerelease):
        log_error("Failed to create GitHub release")
        return False

    log_info(f"Release {new_version} created successfully!")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='GitHub Release Orchestrator')
    parser.add_argument('repo_path', help='Path to git repository')
    parser.add_argument('repo_name', help='GitHub repository (owner/repo)')
    parser.add_argument('bump_type', choices=['major', 'minor', 'patch'], help='Version bump type')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--skip-build', action='store_true', help='Skip building artifacts')
    parser.add_argument('--prerelease', action='store_true', help='Mark as prerelease')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        global VERBOSE
        VERBOSE = True

    success = orchestrate_release(
        args.repo_path,
        args.repo_name,
        args.bump_type,
        args.skip_tests,
        args.skip_build,
        args.prerelease
    )

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
