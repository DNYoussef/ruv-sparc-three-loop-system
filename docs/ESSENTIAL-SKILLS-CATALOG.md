# Essential Skills Catalog

**Must-Have Skills for Every Claude Code Instance**

These 5 essential skills should be in every Claude Code installation. They provide the core workflows for daily development tasks.

---

## The Essential 5

### 1. 🚀 Quick Quality Check (`/quick-check`)

**Purpose**: Lightning-fast quality validation in under 30 seconds

**When to Use**:
- Before committing code
- Quick sanity check
- Pre-PR validation
- During development

**What It Does**:
Runs 5 checks in parallel:
- 🎭 Theater detection
- 🎨 Style audit
- 🔒 Security scan
- 🧪 Basic tests
- 📊 Token analysis

**Usage**:
```bash
/quick-check .                  # Check current directory
/quick-check src/api/users.js   # Check specific file
/quick-check src/ --detailed    # Detailed output
```

**Output**: Quality score (0-100) + prioritized issues

**Files**:
- Skill: `.claude/skills/quick-quality-check/SKILL.md`
- Command: `.claude/commands/essential-commands/quick-check.md`

---

### 2. 🐛 Smart Bug Fix (`/fix-bug`)

**Purpose**: Systematic bug fixing with RCA + multi-model reasoning

**When to Use**:
- Production incidents
- Mysterious bugs
- Intermittent failures
- Any debugging

**What It Does**:
6-phase intelligent debugging:
1. 🔍 Root cause analysis (Claude RCA)
2. 🧠 Context analysis (Gemini if large)
3. 💡 Alternative solutions (Multi-model)
4. 🔧 Auto-fix (Codex sandbox)
5. ✅ Comprehensive testing (Codex iteration)
6. 📊 Performance impact

**Usage**:
```bash
/fix-bug "API timeout under load" src/api/
/fix-bug "Login fails Firefox" src/auth/ --reproduction-steps "..."
/fix-bug "Database fails" src/db/ --error-logs "logs/error.log"
```

**Output**: Root cause + fix + validation + performance impact

**Models Used**: Claude (RCA), Gemini (context), Codex (fix+test)

**Files**:
- Skill: `.claude/skills/smart-bug-fix/SKILL.md`
- Command: `.claude/commands/essential-commands/fix-bug.md`

---

### 3. ⚡ Feature Development Complete (`/build-feature`)

**Purpose**: Complete feature lifecycle from research to deployment

**When to Use**:
- New feature development
- Major enhancements
- Complete implementations
- Sprint work

**What It Does**:
12-stage complete lifecycle:
1. 🔍 Research (Gemini Search)
2. 📊 Codebase analysis (Gemini MegaContext)
3. 🏗️ Architecture design (Claude)
4. 🎨 Diagrams (Gemini Media)
5. ⚡ Prototype (Codex)
6. 🎭 Theater detection
7. ✅ Testing (Codex iteration)
8. 💎 Style polish (Claude)
9. 🔒 Security review
10. 📝 Documentation
11. 🎯 Readiness check
12. 🚀 PR / Deploy

**Usage**:
```bash
/build-feature "User auth with JWT"
/build-feature "Payment integration" src/payments/
/build-feature "Dark mode" --create-pr false
```

**Output**: Complete feature + tests + docs + PR

**Time**: ~15-20 minutes for typical feature

**Files**:
- Skill: `.claude/skills/feature-dev-complete/SKILL.md`
- Command: `.claude/commands/essential-commands/build-feature.md`

---

### 4. 🔍 Code Review Assistant (`/review-pr`)

**Purpose**: Multi-agent comprehensive PR review

**When to Use**:
- Before merging PRs
- Code review automation
- Quality gate enforcement
- Team code reviews

**What It Does**:
Multi-agent swarm review (5 specialists):
- 🔒 Security Reviewer
- ⚡ Performance Analyst
- 🎨 Style Reviewer
- 🧪 Test Specialist
- 📝 Documentation Reviewer

**Quality Gates**:
- ✅ All tests passing
- ✅ Code quality ≥ 80/100
- ✅ No critical security issues
- ✅ No high-severity bugs

**Usage**:
```bash
/review-pr 123                        # Full review
/review-pr 123 security               # Security focus
/review-pr 123 performance,tests      # Selected areas
/review-pr 123 --auto-merge true      # Auto-merge if passing
```

**Output**: Overall score + detailed reviews + fix suggestions + merge decision

**Files**:
- Skill: `.claude/skills/code-review-assistant/SKILL.md`
- Command: `.claude/commands/essential-commands/review-pr.md`

---

### 5. 🚢 Production Readiness (`/deploy-check`)

**Purpose**: Comprehensive pre-deployment validation

**When to Use**:
- Before production deployment
- Release validation
- Staging promotion
- Deployment gates

**What It Does**:
10-gate production readiness:
1. ✅ Tests (100% passing)
2. ✅ Quality (≥ 85/100)
3. ✅ Coverage (≥ 80%)
4. ✅ Security (0 critical/high)
5. ✅ Performance (SLA compliance)
6. ✅ Documentation (complete)
7. ✅ Dependencies (no vulnerabilities)
8. ✅ Configuration (proper secrets)
9. ✅ Monitoring (configured)
10. ✅ Rollback plan (documented)

**Usage**:
```bash
/deploy-check . production           # Production check
/deploy-check ./dist staging         # Staging check
/deploy-check . --skip-performance   # Skip perf tests
```

**Output**: Gate status + deployment checklist + go/no-go decision

**Files**:
- Skill: `.claude/skills/production-readiness/SKILL.md`
- Command: `.claude/commands/essential-commands/deploy-check.md`

---

## Installation

### Quick Install (All 5 Skills)

```bash
# Already installed in your Claude Code instance!
# Skills are in: ~/.claude/skills/
# Commands are in: ~/.claude/commands/essential-commands/
```

### Verify Installation

```bash
# Check skills exist
ls ~/.claude/skills/ | grep -E "quick-quality|smart-bug|feature-dev|code-review|production-readiness"

# Check commands exist
ls ~/.claude/commands/essential-commands/
```

---

## Daily Workflow Examples

### Morning Development Start

```bash
# 1. Quick check current work
/quick-check .

# 2. Fix any issues
/fix-bug "issue description" src/

# 3. Continue development
```

### Feature Development

```bash
# Complete feature lifecycle
/build-feature "User authentication with OAuth"

# Review output
# → Research done
# → Architecture designed
# → Code implemented
# → Tests passing
# → Docs generated
# → PR created
```

### Pre-Commit Workflow

```bash
# Quick quality check before commit
/quick-check . && git commit -m "feat: ..." || echo "Fix issues first"
```

### PR Review Workflow

```bash
# Automated comprehensive review
/review-pr 123

# Review comment posted with:
# - Overall score
# - Detailed findings
# - Fix suggestions
# - Merge decision
```

### Deployment Workflow

```bash
# 1. Check staging readiness
/deploy-check ./dist staging

# 2. Deploy to staging
/deploy staging

# 3. Check production readiness
/deploy-check . production

# 4. Deploy to production
/deploy production
```

---

## Composition Patterns

### Quality Pipeline

```bash
# Quick → Full Audit → Deploy Check
/quick-check src/ && \
/audit-pipeline src/ && \
/deploy-check .
```

### Bug Fix → Test → Deploy

```bash
# Fix → Validate → Deploy
/fix-bug "bug" src/ && \
/functionality-audit src/ && \
/deploy-check . && \
/deploy
```

### Feature → Review → Deploy

```bash
# Build → Review → Deploy
/build-feature "feature" && \
/review-pr $PR_NUMBER && \
/deploy-check . && \
/deploy
```

### Parallel Quality Checks

```bash
# Multiple checks in parallel
parallel ::: \
  "/quick-check src/" \
  "/security-scan src/" \
  "/test-coverage src/"
```

---

## Integration with 65+ Commands

These 5 essential skills compose with all 65+ slash commands:

| Essential Skill | Uses Commands | Chains With |
|----------------|---------------|-------------|
| `/quick-check` | `/theater-detect`, `/style-audit`, `/security-scan`, `/test-coverage`, `/token-usage` | `/audit-pipeline`, `/production-readiness` |
| `/fix-bug` | `/agent-rca`, `/gemini-megacontext`, `/codex-auto`, `/functionality-audit` | `/quick-check`, `/review-pr` |
| `/build-feature` | `/gemini-search`, `/gemini-media`, `/codex-auto`, `/theater-detect`, `/audit-pipeline` | `/review-pr`, `/deploy-check` |
| `/review-pr` | `/swarm-init`, `/security-scan`, `/bottleneck-detect`, `/style-audit`, `/test-coverage` | `/fix-bug`, `/audit-pipeline` |
| `/deploy-check` | `/audit-pipeline`, `/security-scan`, `/performance-report`, `/bottleneck-detect` | `/build-feature`, `/review-pr` |

---

## Comparison with Other Commands

### Quick Check vs Full Audit

| Feature | `/quick-check` | `/audit-pipeline` |
|---------|----------------|-------------------|
| Speed | < 30 seconds | 2-5 minutes |
| Depth | Basic | Comprehensive |
| Parallel | Yes | Sequential |
| Use Case | During dev | Before merge/deploy |
| Codex Iteration | No | Yes (Phase 2) |

### Fix Bug vs Codex Auto

| Feature | `/fix-bug` | `/codex-auto` |
|---------|------------|---------------|
| RCA | ✅ Deep analysis | ❌ No analysis |
| Multi-model | ✅ Yes | ❌ Codex only |
| Testing | ✅ Comprehensive | ❌ Basic |
| Validation | ✅ 5 iterations | ❌ Single run |
| Use Case | Production bugs | Quick prototypes |

### Build Feature vs Manual Development

| Feature | `/build-feature` | Manual |
|---------|------------------|--------|
| Research | ✅ Automated | ❌ Manual |
| Architecture | ✅ Automated | ❌ Manual |
| Implementation | ✅ Codex Auto | ❌ Manual |
| Testing | ✅ Codex iteration | ❌ Manual |
| Docs | ✅ Auto-generated | ❌ Manual |
| PR | ✅ Auto-created | ❌ Manual |
| Time | ~18 minutes | Hours/days |

---

## Best Practices

### 1. Use Quick Check Frequently

```bash
# Before every commit
/quick-check . && git commit

# During development (fast feedback)
watch -n 60 "/quick-check src/"
```

### 2. Fix Bugs Systematically

```bash
# Always use RCA, don't guess
/fix-bug "bug description" src/

# Not: /codex-auto "fix the bug"  ❌
```

### 3. Build Features Completely

```bash
# Use full lifecycle, not pieces
/build-feature "feature spec"

# Not: manually research, design, code, test  ❌
```

### 4. Review Every PR

```bash
# Automated review before human review
/review-pr 123

# Catches issues humans miss
```

### 5. Validate Before Deploying

```bash
# Always check production readiness
/deploy-check . production

# Never deploy without validation  ❌
```

---

## Troubleshooting

### Quick Check Taking Too Long

```bash
# Reduce parallelism if resource constrained
/quick-check . --parallel false
```

### Fix Bug Not Finding Root Cause

```bash
# Increase depth for complex bugs
/fix-bug "bug" src/ --depth deep

# Provide more context
/fix-bug "bug" src/ --error-logs "logs/" --reproduction-steps "..."
```

### Build Feature Failing Tests

```bash
# Increase Codex iteration limit
# Edit skill SKILL.md: max-iterations: 10 (default: 5)
```

### Review PR GitHub Auth Error

```bash
# Authenticate GitHub CLI
gh auth login
```

### Deploy Check Failing Gates

```bash
# Check which gates failed
cat production-readiness-*/DEPLOYMENT-CHECKLIST.md

# Fix issues and recheck
/deploy-check .
```

---

## Summary

### The Essential 5 (Must-Have)

1. **`/quick-check`** - Fast quality validation (< 30s)
2. **`/fix-bug`** - Systematic debugging with RCA
3. **`/build-feature`** - Complete feature lifecycle (12 stages)
4. **`/review-pr`** - Multi-agent PR review (5 specialists)
5. **`/deploy-check`** - Production readiness (10 gates)

### Why These 5?

✅ **Cover daily workflows**: Dev, debug, review, deploy
✅ **Use multi-model AI**: Gemini, Codex, Claude orchestrated
✅ **Compose with 65+ commands**: Part of larger ecosystem
✅ **Save time**: Hours → minutes for common tasks
✅ **Improve quality**: Automated best practices

### Installation Status

✅ **All 5 skills installed** in: `~/.claude/skills/`
✅ **All 5 commands registered** in: `~/.claude/commands/essential-commands/`
✅ **Ready to use immediately**

### Next Steps

1. **Try quick check**: `/quick-check .`
2. **Fix a bug**: `/fix-bug "description" src/`
3. **Build a feature**: `/build-feature "feature spec"`
4. **Review a PR**: `/review-pr PR_NUMBER`
5. **Check deployment**: `/deploy-check . production`

🎉 **You now have the 5 essential skills every Claude Code instance needs!**
