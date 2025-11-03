# Example: Automated Changelog Generation

This example demonstrates intelligent changelog generation from commits and pull requests with AI-powered categorization.

## Scenario

You need to generate professional changelogs that:
- Automatically categorize changes by type
- Include PR links and contributor attribution
- Detect breaking changes and migration needs
- Support multiple output formats (Markdown, JSON, HTML)
- Generate migration guides when needed

## Project Setup

```
my-project/
├── .github/
│   ├── workflows/
│   │   └── changelog.yml
│   └── scripts/
│       └── generate-release-notes.sh
├── resources/
│   ├── changelog-generator.py
│   └── changelog-template.md
├── CHANGELOG.md
├── MIGRATION.md
└── package.json
```

## Step 1: Enhanced Changelog Generator Script

```bash
#!/bin/bash
# .github/scripts/generate-release-notes.sh

set -euo pipefail

# Configuration
FROM_TAG="${FROM_TAG:-}"
TO_TAG="${TO_TAG:-HEAD}"
VERSION="${VERSION:-}"
OUTPUT="${OUTPUT:-CHANGELOG.md}"
FORMAT="${FORMAT:-markdown}"

# Get last tag if not specified
if [[ -z "$FROM_TAG" ]]; then
    FROM_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
fi

if [[ -z "$FROM_TAG" ]]; then
    echo "❌ No previous tags found. Cannot generate changelog."
    exit 1
fi

echo "📝 Generating changelog from $FROM_TAG to $TO_TAG..."

# Generate changelog
python resources/changelog-generator.py \
    --from "$FROM_TAG" \
    --to "$TO_TAG" \
    --version "$VERSION" \
    --format "$FORMAT" \
    --output "$OUTPUT" \
    --check-migrations

echo "✅ Changelog generated: $OUTPUT"

# Check if migration guide is needed
if grep -q "migration guide recommended" "$OUTPUT" 2>/dev/null; then
    echo "⚠️  Breaking changes detected!"
    echo "📋 Generating migration guide..."

    # Generate migration guide
    cat > MIGRATION.md << 'EOF'
# Migration Guide

## Upgrading from VERSION_OLD to VERSION_NEW

### Breaking Changes

BREAKING_CHANGES_LIST

### Migration Steps

1. **Update Dependencies**
   ```bash
   npm install package@VERSION_NEW
   ```

2. **Update Code**
   Review breaking changes above and update your code accordingly.

3. **Test**
   Run your test suite to verify everything works:
   ```bash
   npm test
   ```

4. **Deploy**
   Deploy to staging first, then production.

### Need Help?

- Check our [documentation](https://docs.example.com)
- Open an [issue](https://github.com/org/repo/issues)
- Join our [community](https://discord.gg/example)
EOF

    # Extract breaking changes from changelog
    BREAKING=$(python resources/changelog-generator.py \
        --from "$FROM_TAG" \
        --to "$TO_TAG" \
        --format json | jq -r '.categories.breaking.commits[].description')

    # Update migration guide with actual breaking changes
    sed -i "s/VERSION_OLD/$FROM_TAG/g" MIGRATION.md
    sed -i "s/VERSION_NEW/$VERSION/g" MIGRATION.md
    sed -i "s/BREAKING_CHANGES_LIST/$BREAKING/g" MIGRATION.md

    echo "✅ Migration guide generated: MIGRATION.md"
fi
```

## Step 2: Advanced Changelog Configuration

```python
#!/usr/bin/env python3
# scripts/smart-changelog.py

"""
Smart Changelog Generator with AI Categorization
Extends base changelog generator with intelligent features
"""

import re
import sys
from typing import List, Dict, Set
from collections import defaultdict
from changelog_generator import ChangelogGenerator, Commit, CATEGORIES

class SmartChangelogGenerator(ChangelogGenerator):
    """Enhanced changelog generator with smart categorization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contributor_stats: Dict[str, int] = defaultdict(int)
        self.scope_stats: Dict[str, int] = defaultdict(int)
        self.breaking_details: List[Dict] = []

    def analyze_commit_patterns(self):
        """Analyze commit patterns for insights"""
        for commit in self.commits:
            # Track contributor activity
            self.contributor_stats[commit.author] += 1

            # Track scope usage
            if commit.scope:
                self.scope_stats[commit.scope] += 1

            # Collect breaking change details
            if commit.breaking:
                self.breaking_details.append({
                    'commit': commit.sha[:7],
                    'description': commit.description,
                    'scope': commit.scope,
                    'author': commit.author
                })

    def detect_feature_flags(self) -> List[str]:
        """Detect feature flag mentions in commits"""
        flags = set()
        flag_pattern = re.compile(r'feature[_-]flag[:\s]+([a-z0-9_-]+)', re.IGNORECASE)

        for commit in self.commits:
            matches = flag_pattern.findall(commit.message)
            flags.update(matches)

        return sorted(flags)

    def detect_deprecations(self) -> List[Dict]:
        """Detect deprecation warnings"""
        deprecations = []
        deprecation_keywords = ['deprecate', 'deprecated', 'deprecation']

        for commit in self.commits:
            message_lower = commit.message.lower()

            for keyword in deprecation_keywords:
                if keyword in message_lower:
                    deprecations.append({
                        'commit': commit.sha[:7],
                        'description': commit.description,
                        'category': commit.category
                    })
                    break

        return deprecations

    def generate_migration_guide(self) -> str:
        """Generate detailed migration guide"""
        if not self.breaking_details:
            return ""

        lines = [
            "# Migration Guide",
            "",
            "## Breaking Changes",
            ""
        ]

        for detail in self.breaking_details:
            lines.append(f"### {detail['description']}")
            lines.append("")

            if detail['scope']:
                lines.append(f"**Affected scope**: `{detail['scope']}`")
                lines.append("")

            lines.append(f"**Commit**: {detail['commit']}")
            lines.append(f"**Author**: {detail['author']}")
            lines.append("")

            # Add migration steps placeholder
            lines.append("**Migration steps**:")
            lines.append("1. Review the breaking change")
            lines.append("2. Update your code accordingly")
            lines.append("3. Test thoroughly")
            lines.append("")

        return '\n'.join(lines)

    def generate_statistics(self) -> str:
        """Generate release statistics"""
        lines = [
            "## 📊 Release Statistics",
            "",
            f"- **Total Commits**: {len(self.commits)}",
            f"- **Contributors**: {len(self.contributor_stats)}",
            f"- **Categories**: {len([c for c in self.categories.values() if c])}",
            ""
        ]

        # Top contributors
        if self.contributor_stats:
            lines.append("### Top Contributors")
            lines.append("")

            top_contributors = sorted(
                self.contributor_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            for contributor, count in top_contributors:
                lines.append(f"- {contributor}: {count} commits")

            lines.append("")

        # Most active scopes
        if self.scope_stats:
            lines.append("### Most Active Areas")
            lines.append("")

            top_scopes = sorted(
                self.scope_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            for scope, count in top_scopes:
                lines.append(f"- `{scope}`: {count} changes")

            lines.append("")

        return '\n'.join(lines)

    def generate_enhanced_markdown(
        self,
        version: str = None,
        include_stats: bool = True,
        include_migration: bool = True,
        include_deprecations: bool = True
    ) -> str:
        """Generate enhanced markdown with additional insights"""

        # Analyze patterns first
        self.analyze_commit_patterns()

        # Generate base changelog
        base_changelog = self.generate_markdown(
            version=version,
            use_emoji=True,
            include_contributors=True
        )

        # Add enhancements
        sections = [base_changelog]

        # Add feature flags section
        feature_flags = self.detect_feature_flags()
        if feature_flags:
            sections.append("\n### 🚩 Feature Flags\n")
            sections.append("The following feature flags are mentioned in this release:\n")
            for flag in feature_flags:
                sections.append(f"- `{flag}`\n")

        # Add deprecations section
        if include_deprecations:
            deprecations = self.detect_deprecations()
            if deprecations:
                sections.append("\n### ⚠️ Deprecations\n")
                sections.append("The following deprecations were introduced:\n")
                for dep in deprecations:
                    sections.append(f"- {dep['description']} ({dep['commit']})\n")

        # Add statistics
        if include_stats:
            sections.append("\n" + self.generate_statistics())

        # Add migration guide
        if include_migration and self.breaking_details:
            sections.append("\n" + self.generate_migration_guide())

        return ''.join(sections)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Smart Changelog Generator')
    parser.add_argument('--from', dest='from_ref', required=True)
    parser.add_argument('--to', dest='to_ref', default='HEAD')
    parser.add_argument('--version', help='Version number')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--include-stats', action='store_true', default=True)
    parser.add_argument('--include-migration', action='store_true', default=True)
    parser.add_argument('--include-deprecations', action='store_true', default=True)

    args = parser.parse_args()

    # Generate enhanced changelog
    generator = SmartChangelogGenerator(args.from_ref, args.to_ref)
    generator.get_commits()

    output = generator.generate_enhanced_markdown(
        version=args.version,
        include_stats=args.include_stats,
        include_migration=args.include_migration,
        include_deprecations=args.include_deprecations
    )

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"✅ Enhanced changelog written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
```

## Step 3: GitHub Actions Integration

```yaml
# .github/workflows/changelog.yml
name: Generate Changelog

on:
  push:
    tags:
      - 'v*'

  workflow_dispatch:
    inputs:
      from_tag:
        description: 'From tag'
        required: true
      to_tag:
        description: 'To tag or branch'
        required: false
        default: 'HEAD'
      version:
        description: 'Release version'
        required: true

jobs:
  changelog:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Extract Version
        id: version
        run: |
          if [[ "${{ github.event_name }}" == "push" ]]; then
            VERSION="${{ github.ref_name }}"
            VERSION="${VERSION#v}"
            echo "version=$VERSION" >> $GITHUB_OUTPUT

            # Get previous tag
            PREV_TAG=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "")
            echo "from_tag=$PREV_TAG" >> $GITHUB_OUTPUT
            echo "to_tag=${{ github.ref_name }}" >> $GITHUB_OUTPUT
          else
            echo "version=${{ inputs.version }}" >> $GITHUB_OUTPUT
            echo "from_tag=${{ inputs.from_tag }}" >> $GITHUB_OUTPUT
            echo "to_tag=${{ inputs.to_tag }}" >> $GITHUB_OUTPUT
          fi

      - name: Generate Changelog
        run: |
          chmod +x .github/scripts/generate-release-notes.sh

          FROM_TAG="${{ steps.version.outputs.from_tag }}" \
          TO_TAG="${{ steps.version.outputs.to_tag }}" \
          VERSION="${{ steps.version.outputs.version }}" \
          ./.github/scripts/generate-release-notes.sh

      - name: Generate Enhanced Changelog
        run: |
          python scripts/smart-changelog.py \
            --from "${{ steps.version.outputs.from_tag }}" \
            --to "${{ steps.version.outputs.to_tag }}" \
            --version "${{ steps.version.outputs.version }}" \
            --output CHANGELOG_ENHANCED.md

      - name: Update Main Changelog
        run: |
          # Prepend new changelog to existing
          if [[ -f CHANGELOG.md ]]; then
            cat CHANGELOG_ENHANCED.md CHANGELOG.md > CHANGELOG_NEW.md
            mv CHANGELOG_NEW.md CHANGELOG.md
          else
            mv CHANGELOG_ENHANCED.md CHANGELOG.md
          fi

      - name: Commit Changelog
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

          git add CHANGELOG.md MIGRATION.md

          if git diff --cached --quiet; then
            echo "No changes to commit"
          else
            git commit -m "docs: update changelog for ${{ steps.version.outputs.version }}

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

            git push origin HEAD:main
          fi

      - name: Create Release with Changelog
        if: github.event_name == 'push'
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh release create "${{ github.ref_name }}" \
            --title "Release ${{ steps.version.outputs.version }}" \
            --notes-file CHANGELOG_ENHANCED.md \
            --verify-tag

      - name: Post Changelog to Discussion
        if: github.event_name == 'push'
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          DISCUSSION_CATEGORY=$(gh api repos/${{ github.repository }}/discussions/categories \
            --jq '.[] | select(.slug=="announcements") | .id')

          gh api repos/${{ github.repository }}/discussions \
            --method POST \
            -f title="📝 Changelog for ${{ steps.version.outputs.version }}" \
            -f body="$(cat CHANGELOG_ENHANCED.md)" \
            -f category_id="$DISCUSSION_CATEGORY"
```

## Example Output

### Generated Changelog

```markdown
## 2.0.0
_2024-01-15_

### 💥 Breaking Changes

- **api**: Removed deprecated v1 endpoints ([#123](https://github.com/org/repo/pull/123))
  - Migration: Update API calls to use v2 endpoints
  - See [MIGRATION.md](./MIGRATION.md) for details

### 🚀 Features

- **auth**: Implemented OAuth2 authentication ([#118](https://github.com/org/repo/pull/118))
- **database**: Added PostgreSQL support ([#120](https://github.com/org/repo/pull/120))
- **monitoring**: Integrated OpenTelemetry tracing ([#122](https://github.com/org/repo/pull/122))

### 🐛 Bug Fixes

- **api**: Fixed race condition in concurrent requests ([#124](https://github.com/org/repo/pull/124))
- **database**: Resolved connection pool exhaustion ([#126](https://github.com/org/repo/pull/126))

### ⚠️ Deprecations

The following deprecations were introduced:

- Deprecated `oldApi.call()` method (abc1234)
- Legacy auth module will be removed in v3.0.0 (def5678)

### 🚩 Feature Flags

The following feature flags are mentioned in this release:

- `enable_oauth2`
- `use_postgres`
- `enable_tracing`

## 📊 Release Statistics

- **Total Commits**: 25
- **Contributors**: 8
- **Categories**: 6

### Top Contributors

- Alice: 10 commits
- Bob: 5 commits
- Charlie: 4 commits
- Diana: 3 commits
- Eve: 2 commits

### Most Active Areas

- `api`: 8 changes
- `auth`: 5 changes
- `database`: 4 changes
- `monitoring`: 3 changes
- `docs`: 2 changes

### 👥 Contributors

Thanks to the following people for contributing to this release:

- Alice
- Bob
- Charlie
- Diana
- Eve
- Frank
- Grace
- Henry
```

## Usage Examples

### 1. Generate for Specific Range

```bash
python scripts/smart-changelog.py \
  --from v1.0.0 \
  --to v2.0.0 \
  --version 2.0.0 \
  --output CHANGELOG_NEW.md
```

### 2. Generate with All Features

```bash
python scripts/smart-changelog.py \
  --from v1.0.0 \
  --version 2.0.0 \
  --include-stats \
  --include-migration \
  --include-deprecations
```

### 3. CI/CD Integration

```bash
FROM_TAG=v1.9.0 \
TO_TAG=HEAD \
VERSION=2.0.0 \
./.github/scripts/generate-release-notes.sh
```

## Best Practices

1. **Conventional Commits**: Use conventional commit format for accurate categorization
2. **PR Links**: Reference PR numbers in commit messages
3. **Scopes**: Use scopes to organize changes by area
4. **Breaking Changes**: Always document breaking changes with migration steps
5. **Feature Flags**: Mention feature flags in commits for tracking
6. **Deprecations**: Clearly mark deprecations with timeline
7. **Automation**: Generate changelogs on every release

## Summary

This example demonstrates:
- ✅ Intelligent commit categorization
- ✅ Breaking change detection
- ✅ Migration guide generation
- ✅ Feature flag tracking
- ✅ Deprecation warnings
- ✅ Contributor statistics
- ✅ GitHub integration
- ✅ Multi-format output
