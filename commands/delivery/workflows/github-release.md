---
name: github-release
category: deployment
version: 1.0.0
---

# /github-release

Automated GitHub release creation with semantic versioning and changelog generation.

## Usage
```bash
/github-release [version] [options]
```

## Parameters
- `version` - Release version (auto-incremented if not specified)
- `--type` - Release type: major|minor|patch|pre (default: patch)
- `--prerelease` - Mark as pre-release (default: false)
- `--draft` - Create as draft (default: false)
- `--changelog-auto` - Auto-generate changelog (default: true)
- `--tag-prefix` - Tag prefix (default: v)
- `--assets` - Attach assets: path1,path2,... (optional)
- `--target` - Target branch/commit (default: main)
- `--notify` - Send release notifications (default: true)

## What It Does

**Comprehensive Release Automation**:
1. 🏷️ Calculate next semantic version
2. 📝 Generate changelog from commits
3. 🔍 Validate release readiness
4. 🏗️ Build release artifacts
5. 📦 Create GitHub release
6. 🚀 Publish release notes
7. 📢 Send notifications
8. 🔖 Tag repository

**Changelog Generation**:
- Parse conventional commits
- Group by type (feat, fix, docs, etc.)
- List contributors
- Link PRs and issues
- Generate release notes

## Examples

```bash
# Auto-increment patch version
/github-release

# Specific version release
/github-release v1.2.3

# Major version bump
/github-release --type major

# Pre-release
/github-release v2.0.0-beta.1 --prerelease true

# Draft release with assets
/github-release --draft true --assets "dist/myapp.zip,dist/myapp.tar.gz"

# Release without changelog auto-generation
/github-release v1.5.0 --changelog-auto false

# Release on specific branch
/github-release --target develop --type minor
```

## Output

```
🚀 GitHub Release Creation Started

Repository: myorg/myapp
Current Version: v1.2.2
Next Version: v1.3.0 (minor bump)
Target: main (commit: abc1234)

Pre-Release Validation:
  ✅ All tests passing
  ✅ No merge conflicts
  ✅ Branch up to date with remote
  ✅ CI/CD pipeline succeeded
  ✅ No uncommitted changes
  ✅ Semantic version valid

Generating Changelog:
  📝 Analyzing 42 commits since v1.2.2...
  ✅ Parsed 38 conventional commits
  ✅ Found 3 breaking changes
  ✅ Identified 12 contributors

Changelog Preview:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v1.3.0 (2025-11-01)

## 🚀 Features
- Add dark mode toggle (#234) @johndoe
- Implement real-time notifications (#245) @janedoe
- Support multi-language (#251) @contributor

## 🐛 Bug Fixes
- Fix memory leak in WebSocket handler (#238) @dev1
- Resolve authentication timeout issue (#242) @dev2
- Correct timezone handling (#249) @dev3

## 📚 Documentation
- Update API documentation (#240) @techwriter
- Add deployment guide (#246) @devops

## ⚡ Performance
- Optimize database queries (40% faster) (#241)
- Reduce bundle size by 25% (#247)

## 💥 BREAKING CHANGES
- Remove deprecated API endpoints (#243)
- Change authentication flow (#248)
- Update minimum Node.js version to 18 (#250)

## 👥 Contributors
@johndoe, @janedoe, @contributor, @dev1, @dev2,
@dev3, @techwriter, @devops, and 4 others

**Full Changelog**: v1.2.2...v1.3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Building Release Artifacts:
  🔨 Running build script...
  ✅ Build successful (23.4s)
  📦 Creating distribution packages...
     - myapp-v1.3.0-linux-x64.tar.gz (12.4 MB)
     - myapp-v1.3.0-darwin-x64.tar.gz (11.8 MB)
     - myapp-v1.3.0-win-x64.zip (13.2 MB)
  ✅ Artifacts created (3 files, 37.4 MB total)

Creating Git Tag:
  🏷️ Tag: v1.3.0
  📝 Message: "Release v1.3.0"
  ✅ Tag created locally
  ✅ Tag pushed to origin

Creating GitHub Release:
  🚀 Release: v1.3.0
  📝 Title: "myapp v1.3.0 - Dark Mode & Real-time Features"
  ✅ Release created
  ✅ Changelog attached

Uploading Assets:
  📤 myapp-v1.3.0-linux-x64.tar.gz... ✅ (4.2s)
  📤 myapp-v1.3.0-darwin-x64.tar.gz... ✅ (4.0s)
  📤 myapp-v1.3.0-win-x64.zip... ✅ (4.5s)

Publishing Release:
  ✅ Release published (draft → public)
  🌐 URL: https://github.com/myorg/myapp/releases/tag/v1.3.0

Post-Release Actions:
  📢 Slack notification sent (#releases channel)
  📧 Email notification sent (subscribers: 1,234)
  🐦 Twitter announcement posted
  📊 GitHub Discussions thread created

Release Summary:
  Version: v1.3.0
  Type: Minor Release
  Commits: 42
  Contributors: 12
  Breaking Changes: 3
  Assets: 3 files (37.4 MB)
  Time: 45.3s

Next Steps:
  - Monitor deployment: /monitoring-configure
  - Deploy to staging: /k8s-deploy --environment staging
  - Create announcement: docs/announcements/v1.3.0.md

✅ GitHub Release Complete!
🌐 https://github.com/myorg/myapp/releases/tag/v1.3.0
```

## Chains With

```bash
# Test → release → deploy
/regression-test && /github-release && /k8s-deploy

# Build → release → notify
/docker-build --tag v1.0.0 && /github-release v1.0.0

# Full release pipeline
/workflow:cicd --stage release

# Release with deployment
/github-release --type minor && \
/docker-build && \
/k8s-deploy --environment production
```

## See Also
- `/workflow:cicd` - Full CI/CD pipeline
- `/k8s-deploy` - Kubernetes deployment
- `/docker-build` - Container image build
- `/monitoring-configure` - Post-release monitoring
- `/pr-enhance` - PR enhancement before release
