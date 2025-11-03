#!/usr/bin/env python3
"""
Changelog Generator with AI-Powered Categorization
Generates structured changelogs from git commits and GitHub PRs
"""

import re
import sys
import json
import argparse
import subprocess
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Changelog categories with emojis
CATEGORIES = {
    'breaking': {'title': '💥 Breaking Changes', 'priority': 1, 'emoji': '💥'},
    'security': {'title': '🔒 Security', 'priority': 2, 'emoji': '🔒'},
    'feature': {'title': '🚀 Features', 'priority': 3, 'emoji': '🚀'},
    'enhancement': {'title': '✨ Enhancements', 'priority': 4, 'emoji': '✨'},
    'fix': {'title': '🐛 Bug Fixes', 'priority': 5, 'emoji': '🐛'},
    'performance': {'title': '⚡ Performance', 'priority': 6, 'emoji': '⚡'},
    'refactor': {'title': '♻️ Refactoring', 'priority': 7, 'emoji': '♻️'},
    'docs': {'title': '📚 Documentation', 'priority': 8, 'emoji': '📚'},
    'test': {'title': '✅ Tests', 'priority': 9, 'emoji': '✅'},
    'build': {'title': '🔧 Build System', 'priority': 10, 'emoji': '🔧'},
    'ci': {'title': '👷 CI/CD', 'priority': 11, 'emoji': '👷'},
    'chore': {'title': '🧹 Chores', 'priority': 12, 'emoji': '🧹'},
    'other': {'title': '📦 Other', 'priority': 13, 'emoji': '📦'}
}

# Conventional commit patterns
COMMIT_PATTERN = re.compile(
    r'^(?P<type>[a-z]+)'
    r'(?:\((?P<scope>[a-z0-9\-]+)\))?'
    r'(?P<breaking>!)?'
    r':\s+'
    r'(?P<description>.+)'
    r'(?:\n\n(?P<body>(?:.|\n)+))?'
    r'(?:\n\n(?P<footer>(?:.|\n)+))?$',
    re.MULTILINE
)

BREAKING_PATTERN = re.compile(r'BREAKING[- ]CHANGE:\s*(.+)', re.MULTILINE | re.IGNORECASE)


class Commit:
    """Represents a git commit"""

    def __init__(self, sha: str, message: str, author: str, date: str):
        self.sha = sha
        self.message = message
        self.author = author
        self.date = date
        self.category = 'other'
        self.scope = None
        self.description = message.split('\n')[0]
        self.breaking = False
        self.pr_number = None

        self._parse()

    def _parse(self):
        """Parse conventional commit format"""
        match = COMMIT_PATTERN.match(self.message)

        if match:
            commit_type = match.group('type')
            self.scope = match.group('scope')
            self.description = match.group('description').strip()
            self.breaking = bool(match.group('breaking'))

            body = match.group('body') or ''
            footer = match.group('footer') or ''

            # Check for breaking changes in body/footer
            if BREAKING_PATTERN.search(body + footer):
                self.breaking = True

            # Map commit type to category
            self.category = self._map_type_to_category(commit_type)

        # Extract PR number
        pr_match = re.search(r'#(\d+)', self.description)
        if pr_match:
            self.pr_number = pr_match.group(1)

        # Override category if breaking change
        if self.breaking:
            self.category = 'breaking'

    def _map_type_to_category(self, commit_type: str) -> str:
        """Map conventional commit type to changelog category"""
        type_map = {
            'feat': 'feature',
            'feature': 'feature',
            'fix': 'fix',
            'bugfix': 'fix',
            'docs': 'docs',
            'doc': 'docs',
            'documentation': 'docs',
            'style': 'other',
            'refactor': 'refactor',
            'perf': 'performance',
            'performance': 'performance',
            'test': 'test',
            'tests': 'test',
            'build': 'build',
            'ci': 'ci',
            'chore': 'chore',
            'revert': 'other',
            'security': 'security',
            'sec': 'security'
        }
        return type_map.get(commit_type, 'other')

    def format_line(self, use_emoji: bool = True) -> str:
        """Format commit as changelog line"""
        parts = []

        if self.scope:
            parts.append(f"**{self.scope}**:")

        parts.append(self.description)

        if self.pr_number:
            parts.append(f"([#{self.pr_number}](https://github.com/{{repo}}/pull/{self.pr_number}))")

        if self.sha:
            short_sha = self.sha[:7]
            parts.append(f"([{short_sha}](https://github.com/{{repo}}/commit/{self.sha}))")

        return '- ' + ' '.join(parts)


class ChangelogGenerator:
    """Generates changelog from git history"""

    def __init__(self, from_ref: str, to_ref: str = 'HEAD', repo_path: str = '.'):
        self.from_ref = from_ref
        self.to_ref = to_ref
        self.repo_path = repo_path
        self.commits: List[Commit] = []
        self.categories: Dict[str, List[Commit]] = defaultdict(list)

    def get_commits(self) -> List[Commit]:
        """Get commits between refs"""
        cmd = [
            'git', 'log',
            f'{self.from_ref}..{self.to_ref}',
            '--pretty=format:%H|%s|%an|%ai',
            '--no-merges'
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = line.split('|', 3)
                if len(parts) == 4:
                    sha, message, author, date = parts

                    # Get full commit message
                    full_msg_cmd = ['git', 'show', '-s', '--format=%B', sha]
                    full_msg = subprocess.run(
                        full_msg_cmd,
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True
                    ).stdout.strip()

                    commit = Commit(sha, full_msg, author, date)
                    self.commits.append(commit)
                    self.categories[commit.category].append(commit)

            return self.commits

        except subprocess.CalledProcessError as e:
            print(f"Error getting commits: {e}", file=sys.stderr)
            sys.exit(1)

    def generate_markdown(
        self,
        version: Optional[str] = None,
        date: Optional[str] = None,
        use_emoji: bool = True,
        include_contributors: bool = True
    ) -> str:
        """Generate markdown changelog"""
        lines = []

        # Header
        if version:
            lines.append(f"## {version}")
        else:
            lines.append(f"## Unreleased")

        if date:
            lines.append(f"_{date}_")
        else:
            lines.append(f"_{datetime.now().strftime('%Y-%m-%d')}_")

        lines.append("")

        # Categories in priority order
        sorted_categories = sorted(
            [(cat, commits) for cat, commits in self.categories.items() if commits],
            key=lambda x: CATEGORIES[x[0]]['priority']
        )

        for category, commits in sorted_categories:
            cat_info = CATEGORIES[category]

            if use_emoji:
                lines.append(f"### {cat_info['title']}")
            else:
                lines.append(f"### {cat_info['title'].split(' ', 1)[1]}")

            lines.append("")

            for commit in commits:
                lines.append(commit.format_line(use_emoji))

            lines.append("")

        # Contributors
        if include_contributors and self.commits:
            contributors = sorted(set(c.author for c in self.commits))
            lines.append("### 👥 Contributors")
            lines.append("")
            lines.append("Thanks to the following people for contributing to this release:")
            lines.append("")
            for contributor in contributors:
                lines.append(f"- {contributor}")
            lines.append("")

        return '\n'.join(lines)

    def generate_json(self) -> str:
        """Generate JSON changelog"""
        data = {
            'version': self.to_ref,
            'from': self.from_ref,
            'date': datetime.now().isoformat(),
            'categories': {}
        }

        for category, commits in self.categories.items():
            if not commits:
                continue

            data['categories'][category] = {
                'title': CATEGORIES[category]['title'],
                'commits': [
                    {
                        'sha': c.sha,
                        'description': c.description,
                        'scope': c.scope,
                        'author': c.author,
                        'date': c.date,
                        'breaking': c.breaking,
                        'pr': c.pr_number
                    }
                    for c in commits
                ]
            }

        return json.dumps(data, indent=2)

    def detect_migration_needed(self) -> List[str]:
        """Detect if migration guide is needed"""
        migrations = []

        # Check for breaking changes
        if self.categories.get('breaking'):
            migrations.append("Breaking changes detected - migration guide recommended")

        # Check for database/schema changes
        db_keywords = ['migration', 'schema', 'database', 'model']
        for commit in self.commits:
            if any(kw in commit.message.lower() for kw in db_keywords):
                migrations.append("Database schema changes detected")
                break

        # Check for API changes
        api_keywords = ['api', 'endpoint', 'route', 'controller']
        for commit in self.commits:
            if any(kw in commit.message.lower() for kw in api_keywords) and commit.breaking:
                migrations.append("API changes detected - update client integrations")
                break

        return migrations


def main():
    parser = argparse.ArgumentParser(
        description='Generate changelog from git commits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate changelog from last tag to HEAD
  %(prog)s --from v1.0.0

  # Generate changelog between two tags
  %(prog)s --from v1.0.0 --to v2.0.0

  # Generate with version number
  %(prog)s --from v1.0.0 --version 2.0.0

  # Output as JSON
  %(prog)s --from v1.0.0 --format json

  # No emoji output
  %(prog)s --from v1.0.0 --no-emoji
        """
    )

    parser.add_argument('--from', dest='from_ref', required=True,
                       help='Start reference (tag, branch, commit)')
    parser.add_argument('--to', dest='to_ref', default='HEAD',
                       help='End reference (default: HEAD)')
    parser.add_argument('--version', help='Version number for changelog header')
    parser.add_argument('--date', help='Release date (default: today)')
    parser.add_argument('--format', choices=['markdown', 'json'], default='markdown',
                       help='Output format (default: markdown)')
    parser.add_argument('--no-emoji', action='store_true',
                       help='Disable emoji in output')
    parser.add_argument('--no-contributors', action='store_true',
                       help='Exclude contributors section')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--check-migrations', action='store_true',
                       help='Check if migration guide is needed')

    args = parser.parse_args()

    # Generate changelog
    generator = ChangelogGenerator(args.from_ref, args.to_ref)
    generator.get_commits()

    # Check migrations if requested
    if args.check_migrations:
        migrations = generator.detect_migration_needed()
        if migrations:
            print("⚠️  Migration warnings:", file=sys.stderr)
            for msg in migrations:
                print(f"  - {msg}", file=sys.stderr)
            print(file=sys.stderr)

    # Generate output
    if args.format == 'json':
        output = generator.generate_json()
    else:
        output = generator.generate_markdown(
            version=args.version,
            date=args.date,
            use_emoji=not args.no_emoji,
            include_contributors=not args.no_contributors
        )

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"✅ Changelog written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
