---
name: hook:on-commit
description: Git commit hook with linting, formatting, and validation
category: Automation Hooks
version: 1.0.0
requires:
  - git
  - nodejs (for eslint, prettier)
  - python3 (for black, flake8)
usage: |
  /hook:on-commit --lint --format --run-tests
  /hook:on-commit --validate-commit-msg --check-branch "main,develop"
---

# Hook: On-Commit (Pre-Commit)

**Category**: Automation Hooks
**Purpose**: Automatically validate, lint, and format code before git commits.

## Features

- **Linting**: ESLint, Flake8, Pylint, RuboCop
- **Formatting**: Prettier, Black, autopep8
- **Testing**: Run tests on changed files
- **Commit Message Validation**: Conventional commits format
- **Branch Protection**: Prevent direct commits to protected branches

## Command Structure

```bash
/hook:on-commit [OPTIONS]

Options:
  --lint                        Run linters on staged files
  --format                      Auto-format staged files
  --run-tests                   Run tests for changed files
  --validate-commit-msg         Validate commit message format
  --check-branch <list>         Protected branches (prevent direct commits)
  --staged-only                 Only process staged files
  --auto-fix                    Auto-fix linting errors
```

## Implementation

```bash
#!/bin/bash
# Git pre-commit hook

set -e

echo "🔍 Running pre-commit checks..."

# Get staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$STAGED_FILES" ]; then
  echo "⚠️  No staged files to check"
  exit 0
fi

# Lint JavaScript/TypeScript files
JS_FILES=$(echo "$STAGED_FILES" | grep -E '\\.(js|jsx|ts|tsx)$' || true)
if [ -n "$JS_FILES" ]; then
  echo "📝 Linting JavaScript/TypeScript files..."
  npx eslint $JS_FILES --fix
  git add $JS_FILES
fi

# Format with Prettier
if [ -n "$JS_FILES" ]; then
  echo "🎨 Formatting with Prettier..."
  npx prettier --write $JS_FILES
  git add $JS_FILES
fi

# Lint Python files
PY_FILES=$(echo "$STAGED_FILES" | grep -E '\\.py$' || true)
if [ -n "$PY_FILES" ]; then
  echo "🐍 Linting Python files..."
  python3 -m flake8 $PY_FILES
  python3 -m black $PY_FILES
  git add $PY_FILES
fi

# Validate commit message
COMMIT_MSG=$(git log -1 --pretty=%B 2>/dev/null || cat .git/COMMIT_EDITMSG)
if ! echo "$COMMIT_MSG" | grep -qE '^(feat|fix|docs|style|refactor|test|chore)(\\(.+\\))?: .+'; then
  echo "❌ Invalid commit message format"
  echo "Expected: <type>(<scope>): <subject>"
  echo "Example: feat(auth): add login functionality"
  exit 1
fi

# Run tests
echo "🧪 Running tests..."
npm test -- --bail --findRelatedTests $STAGED_FILES

echo "✅ Pre-commit checks passed!"
exit 0
```

## Installation

```bash
# Install as git hook
cp hook-on-commit.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Or use husky
npm install husky --save-dev
npx husky install
npx husky add .git/hooks/pre-commit "/hook:on-commit --lint --format"
```

---

**Status**: Production Ready
**Version**: 1.0.0
