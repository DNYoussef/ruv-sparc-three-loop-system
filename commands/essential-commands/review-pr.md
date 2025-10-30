---
name: review-pr
binding: skill:code-review-assistant
category: essential
version: 1.0.0
---

# /review-pr

Multi-agent comprehensive PR review with specialized reviewers.

## Usage
```bash
/review-pr <pr_number> [focus_areas]
```

## Parameters
- `pr_number` - PR number to review (required)
- `focus_areas` - Comma-separated: security,performance,style,tests,documentation (default: all)
- `--suggest-fixes` - Generate fix suggestions (default: true)
- `--auto-merge` - Auto-merge if passing (default: false)

## What It Does

**Multi-Agent Swarm Review** (5 specialists in parallel):
1. 🔒 **Security Reviewer** - Vulnerabilities, secrets, unsafe patterns
2. ⚡ **Performance Analyst** - Bottlenecks, optimization opportunities
3. 🎨 **Style Reviewer** - Code style, best practices, maintainability
4. 🧪 **Test Specialist** - Coverage, quality, edge cases
5. 📝 **Documentation Reviewer** - Comments, API docs, README updates

**Quality Gates** (for merge approval):
- ✅ All tests passing
- ✅ Code quality ≥ 80/100
- ✅ No critical security issues
- ✅ No high-severity bugs

## Examples

```bash
# Review PR with all checks
/review-pr 123

# Security-focused review
/review-pr 123 security

# Performance and tests only
/review-pr 123 performance,tests

# Review with auto-merge
/review-pr 123 --auto-merge true

# Review without fix suggestions
/review-pr 123 --suggest-fixes false
```

## Output

```
🤖 Code Review Complete: PR #123

Overall Score: 87/100
Merge Ready: ✅ Yes

Review Summary:
  🔒 Security: 95/100 ✅
  ⚡ Performance: 88/100 ✅
  🎨 Style: 90/100 ✅
  🧪 Tests: 82/100 ✅
  📝 Docs: 85/100 ✅

Fix Suggestions:
  1. Add input validation to API endpoint
  2. Extract duplicated code into helper function
  3. Add JSDoc comments to public methods

Decision: ✅ APPROVED

PR comment posted with detailed review.
```

## Review Process

1. **Fetch PR**: Get files changed, additions, deletions
2. **Initialize Swarm**: Spawn 5 specialized review agents
3. **Parallel Review**: All agents review simultaneously
4. **Quality Audit**: Run complete audit pipeline
5. **Aggregate Results**: Combine all findings
6. **Generate Fixes**: Codex suggests improvements
7. **Assess Merge Readiness**: Check quality gates
8. **Post Comment**: Detailed review on GitHub
9. **Approve/Request Changes**: Based on quality gates

## Chains With

```bash
# Review → fix issues → re-review
/review-pr 123 || /fix-issues && /review-pr 123

# Review → merge → deploy
/review-pr 123 --auto-merge && /deploy
```

## See Also
- `/pr-enhance` - Improve PR description
- `/audit-pipeline` - Complete quality audit
- `/fix-bug` - Fix review findings
