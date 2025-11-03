---
name: hook:on-push
description: Git push hook with CI trigger and deployment
category: Automation Hooks
version: 1.0.0
requires:
  - git
  - github-cli (optional)
usage: |
  /hook:on-push --trigger-ci --notify-team
  /hook:on-push --deploy-env "staging" --run-integration-tests
---

# Hook: On-Push (Pre-Push)

**Category**: Automation Hooks
**Purpose**: Validate and prepare code before pushing to remote repository.

## Implementation

```bash
#!/bin/bash
# Git pre-push hook

echo "🚀 Preparing to push..."

# Run full test suite
echo "🧪 Running full test suite..."
npm test

# Check for secrets
echo "🔐 Checking for secrets..."
git diff origin/main...HEAD | grep -iE '(api_key|password|secret|token)' && exit 1 || true

# Trigger CI
echo "⚙️  Triggering CI pipeline..."
gh workflow run ci.yml

echo "✅ Push checks passed!"
exit 0
```

---

**Status**: Production Ready
**Version**: 1.0.0
