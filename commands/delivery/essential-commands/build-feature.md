---
name: build-feature
binding: skill:feature-dev-complete
category: essential
version: 1.0.0
---

# /build-feature

Complete feature development lifecycle from research to deployment (12 stages).

## Usage
```bash
/build-feature "<feature description>" [target_directory]
```

## Parameters
- `feature_description` - Feature to build (required)
- `target_directory` - Target directory (default: src/)
- `--create-pr` - Create PR after completion (default: true)
- `--deploy-after` - Deploy after PR merge (default: false)

## What It Does

**12-Stage Complete Lifecycle**:
1. 🔍 Research best practices (Gemini Search)
2. 📊 Analyze codebase patterns (Gemini MegaContext)
3. 🏗️ Design architecture (Claude Architect)
4. 🎨 Generate diagrams (Gemini Media)
5. ⚡ Rapid prototype (Codex Auto)
6. 🎭 Theater detection (find TODOs)
7. ✅ Comprehensive testing (Codex iteration)
8. 💎 Style polish (Claude)
9. 🔒 Security review (deep scan)
10. 📝 Documentation (multi-model)
11. 🎯 Production readiness check
12. 🚀 Create PR / Deploy

**Multi-Model Strategy**:
- **Gemini Search**: Latest best practices
- **Gemini MegaContext**: Large codebase analysis
- **Gemini Media**: Visual documentation
- **Codex**: Rapid prototyping
- **Claude**: Architecture & testing

## Examples

```bash
# Build complete feature
/build-feature "User authentication with JWT and refresh tokens"

# Feature with custom target
/build-feature "Payment processing integration" src/payments/

# Feature without PR
/build-feature "Dark mode toggle" --create-pr false

# Feature with auto-deploy
/build-feature "API rate limiting" --deploy-after true
```

## Output

```
🚀 Feature Development Complete!

Feature: User authentication with JWT

Artifacts:
  ✅ Research (best practices documented)
  ✅ Architecture (design + diagrams)
  ✅ Implementation (all tests passing)
  ✅ Documentation (comprehensive)

Quality Metrics:
  - Code Quality: 92/100
  - Test Coverage: 95%
  - Security: No issues
  - Performance: Within SLAs

PR Created: https://github.com/user/repo/pull/123

Time: 18 minutes
```

## Chains With

```bash
# Build feature → review → deploy
/build-feature "feature" && /code-review \$PR && /deploy

# Build multiple features in parallel
parallel ::: \
  "/build-feature 'feature1'" \
  "/build-feature 'feature2'" \
  "/build-feature 'feature3'"
```

## See Also
- `/quick-check` - Fast quality check
- `/code-review` - PR review
- `/production-readiness` - Deployment validation
