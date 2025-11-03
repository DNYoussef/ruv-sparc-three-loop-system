# GitHub Release Management Skill

Comprehensive GitHub release orchestration with AI swarm coordination for automated versioning, testing, deployment, and rollback management.

## Quick Start

```bash
# Simple semantic version release
./resources/semantic-versioning.sh --bump minor

# Generate changelog from commits
python ./resources/changelog-generator.py --from v1.0.0 --to HEAD

# Automated release with multi-platform builds
node ./resources/release-automation.js --version 2.0.0 --platforms all

# Package release assets
python ./resources/asset-packager.py --version 2.0.0 --sign
```

## Gold Tier Features

### Resources
- **semantic-versioning.sh**: Intelligent version bump analysis and tag creation
- **changelog-generator.py**: AI-powered changelog generation with categorization
- **release-automation.js**: End-to-end release orchestration workflow
- **asset-packager.py**: Multi-platform build artifact packaging and signing

### Templates
- **release-notes.md**: Structured release notes with sections for features, fixes, breaking changes
- **version-config.yaml**: Centralized version configuration for multi-package releases
- **changelog-template.md**: Categorized changelog template with emoji support

### Examples
- **semantic-release**: Automated semantic versioning workflow (200+ lines)
- **automated-changelogs**: Intelligent changelog generation from commits and PRs (250+ lines)
- **multi-platform-releases**: Cross-platform build and deployment pipeline (300+ lines)

### Tests
- **semantic-versioning.test.js**: Version analysis and bump logic validation
- **changelog-generator.test.py**: Changelog parsing and categorization tests
- **release-automation.test.js**: End-to-end release workflow testing

## Documentation

See [SKILL.md](./SKILL.md) for comprehensive documentation with progressive disclosure levels.

## Usage Patterns

### Quick Release
```bash
# Analyze commits and suggest version
./resources/semantic-versioning.sh --analyze

# Generate release notes
python ./resources/changelog-generator.py --categorize

# Execute release
node ./resources/release-automation.js --auto
```

### Enterprise Release
```bash
# Multi-package coordinated release
node ./resources/release-automation.js \
  --config version-config.yaml \
  --packages "frontend,backend,cli" \
  --staged-rollout \
  --monitoring
```

## Testing

```bash
# Run all tests
npm test

# Test semantic versioning
npm test semantic-versioning.test.js

# Test changelog generation
python -m pytest changelog-generator.test.py

# Test release automation
npm test release-automation.test.js
```

## License

Part of the SPARC Three-Loop System - Claude Flow Team
