# CI/CD Intelligent Recovery - Resources & Scripts

**Skill**: cicd-intelligent-recovery (Loop 3 of Three-Loop System)
**Tier**: Gold
**Version**: 2.0.0

## Overview

This directory contains production-ready scripts, templates, and tests for automated CI/CD failure recovery with intelligent root cause analysis and connascence-aware fix generation.

## Directory Structure

```
resources/
├── scripts/               # Python scripts for automated analysis and repair
│   ├── auto_repair.py           # Automated fix generation with PoT structure
│   ├── root_cause.py            # Graph-based cascade detection
│   ├── recovery_pipeline.sh     # Complete 8-step recovery pipeline
│   └── failure_detect.py        # Real-time failure monitoring
├── templates/             # Configuration and pattern templates
│   ├── recovery-config.yaml     # Complete configuration template
│   ├── failure-patterns.json    # Pattern templates for Loop 1 feedback
│   └── repair-strategies.json   # Fix strategy templates
└── README.md              # This file

tests/
├── test-1-failure-detection.md      # Pattern recognition tests
├── test-2-auto-repair.md             # Repair generation tests
└── test-3-root-cause-analysis.md    # Cascade detection tests
```

## Scripts

### 1. `auto_repair.py` - Automated Fix Generation

**Purpose**: Generate and execute fixes with connascence-aware context bundling.

**Features**:
- Program-of-Thought structure (Plan → Execute → Validate → Approve)
- Automatic strategy selection (isolated/bundled/architectural)
- Connascence coupling analysis
- Dual validation (sandbox + theater)
- Git patch generation with reasoning

**Usage**:
```bash
# Generate fixes for all root causes
python3 auto_repair.py .claude/.artifacts

# Output artifacts:
# - fix-plan-*.json (planning phase)
# - fix-impl-*.json (implementation phase)
# - fix-validation-*.json (validation phase)
# - fixes/*.patch (git patches)
# - auto-repair-summary.json (summary)
```

**Fix Strategies**:
- **Isolated** (0 connascence files): Single file fix
- **Bundled** (1-5 files): Atomic multi-file changes
- **Architectural** (6+ files): System refactor

**Classes**:
- `FixStrategy`: Enum for fix strategies
- `ConnascenceContext`: Coupling analysis
- `RootCause`: Failure information
- `AutoRepair`: Main repair engine

### 2. `root_cause.py` - Root Cause Analysis

**Purpose**: Graph-based cascade detection with Raft consensus.

**Features**:
- Multi-heuristic graph construction
- Root node detection (no incoming edges)
- Cascade depth calculation (BFS)
- Circular dependency detection (DFS)
- 5-Whys methodology validation
- Failure categorization

**Usage**:
```bash
# Analyze failure dependencies
python3 root_cause.py .claude/.artifacts

# Output artifacts:
# - root-causes-consensus.json (validated root causes)
# - Statistics: total, roots, cascaded, cascade ratio
```

**Graph Heuristics**:
1. **Temporal**: Same file, line order (A before B → A causes B)
2. **Error References**: Error message mentions other file
3. **Gemini Dependencies**: From Gemini dependency graph

**Classes**:
- `Failure`: Test failure information
- `FailureGraph`: Dependency graph with cycle detection
- `RootCauseAnalyzer`: Main analysis engine

### 3. `recovery_pipeline.sh` - Complete Pipeline

**Purpose**: Orchestrate full 8-step CI/CD recovery process.

**Features**:
- End-to-end automation
- GitHub integration
- Progress visualization
- Error handling
- Artifact management

**Usage**:
```bash
# Run complete pipeline
./recovery_pipeline.sh

# With custom artifacts directory
ARTIFACTS_DIR=/custom/path ./recovery_pipeline.sh

# With specific repository
GITHUB_REPOSITORY=owner/repo ./recovery_pipeline.sh
```

**Pipeline Steps**:
1. GitHub Hook Integration (download failures)
2. AI-Powered Analysis (Gemini + 7 agents)
3. Root Cause Detection (graph analysis)
4. Intelligent Fixes (connascence-aware)
5. Theater Detection Audit (6-agent Byzantine)
6. Sandbox Validation (production-like tests)
7. Differential Analysis (before/after comparison)
8. GitHub Feedback (PR creation + memory storage)

**Exit Codes**:
- `0`: Success
- `1`: Critical error
- `2`: Warning (non-critical issues)

### 4. `failure_detect.py` - Real-Time Monitoring

**Purpose**: Pattern recognition and trend analysis for CI/CD failures.

**Features**:
- 10+ failure pattern types
- Severity classification (critical/high/medium/low)
- Historical tracking
- Trend detection
- Alert generation

**Usage**:
```bash
# Analyze current failures
python3 failure_detect.py .claude/.artifacts

# Output artifacts:
# - failure-detection-analysis.json (current analysis)
# - failure-patterns-history.json (historical data)
```

**Pattern Types**:
- `null_pointer`: Null/undefined access
- `type_error`: Type mismatches
- `async_error`: Async handling issues
- `import_error`: Module dependencies
- `assertion_error`: Test failures
- `timeout_error`: Performance issues
- `network_error`: Connectivity problems
- `database_error`: Database issues
- `permission_error`: Authorization failures
- `syntax_error`: Code syntax errors

**Classes**:
- `FailurePattern`: Pattern metadata
- `FailureMonitor`: Real-time monitoring engine

## Templates

### 1. `recovery-config.yaml` - Configuration Template

Complete configuration for all pipeline components:

**Sections**:
- GitHub integration (webhooks, authentication)
- Analysis configuration (Gemini, agents, consensus)
- Root cause detection (graph, validation, connascence)
- Fix generation (strategies, program-of-thought)
- Theater detection (agents, patterns, baseline)
- Sandbox validation (environment, test suites)
- Differential analysis (metrics, reports)
- GitHub feedback (commits, PRs, status)
- Memory integration (storage, retrieval)
- Loop 1 feedback (patterns, recommendations)
- Performance metrics (targets, efficiency)
- Artifact management (directories, files)

**Usage**:
```bash
# Copy template for customization
cp recovery-config.yaml .claude/cicd-recovery-config.yaml

# Edit for your project
vim .claude/cicd-recovery-config.yaml
```

### 2. `failure-patterns.json` - Pattern Templates

Templates for 6 common failure categories with prevention strategies.

**Categories**:
1. **null-safety**: Null/undefined handling
2. **type-mismatch**: Type validation
3. **async-handling**: Promise/async errors
4. **authorization**: Permission issues
5. **data-persistence**: Database failures
6. **network-resilience**: Network timeouts

**Each Pattern Includes**:
- Description
- Indicators (regex patterns)
- Prevention strategy
- Pre-mortem question (for Loop 1)
- Testing focus areas
- Common root causes

**Severity Classification**:
- **Critical**: System-wide failures (< 1 hour response)
- **High**: Major features (< 4 hours response)
- **Medium**: Functional issues (< 1 day response)
- **Low**: Minor issues (< 1 week response)

### 3. `repair-strategies.json` - Fix Strategy Templates

Detailed templates for all three fix strategies.

**Strategies**:
1. **Isolated**: Single file fix
   - Applicability: 0 connascence files
   - Approach: Single-file changes
   - Testing: Unit tests only

2. **Bundled**: Multi-file atomic fix
   - Applicability: 1-5 connascence files
   - Approach: Coordinated atomic changes
   - Testing: Unit + integration tests

3. **Architectural**: System refactor
   - Applicability: 6+ connascence files
   - Approach: Phased refactoring
   - Testing: Full test suite + new tests

**Program-of-Thought Phases**:
- **Plan**: Understand, identify, design, predict, validate
- **Execute**: Load, apply, document, generate
- **Validate**: Sandbox test, theater check
- **Approve**: Criteria evaluation, decision logic

## Tests

### Test 1: Failure Detection

**File**: `tests/test-1-failure-detection.md`

**Tests**:
- ✅ Parse CI/CD failure logs
- ✅ Categorize failure patterns
- ✅ Generate severity-based alerts
- ✅ Track historical patterns
- ✅ Detect trends over time

**Expected**: 4 patterns detected with correct severity

### Test 2: Automated Repair

**File**: `tests/test-2-auto-repair.md`

**Tests**:
- ✅ Generate fix plans (PoT structure)
- ✅ Identify connascence coupling
- ✅ Select correct fix strategy
- ✅ Execute bundled/isolated fixes
- ✅ Validate fixes (dual validation)
- ✅ Make approval decisions

**Expected**: 2 fixes generated (1 bundled, 1 isolated)

### Test 3: Root Cause Analysis

**File**: `tests/test-3-root-cause-analysis.md`

**Tests**:
- ✅ Build failure dependency graph
- ✅ Identify root causes
- ✅ Detect cascade failures
- ✅ Apply 5-Whys validation
- ✅ Detect circular dependencies
- ✅ Calculate cascade statistics

**Expected**: 2 roots, 3 cascaded, 60% cascade ratio

## Integration with SKILL.md

These resources support the 8-step process documented in `SKILL.md`:

**Step 1**: `recovery_pipeline.sh` (GitHub hooks)
**Step 2**: Agent orchestration (via Claude Code Task tool)
**Step 3**: `root_cause.py` (graph analysis)
**Step 4**: `auto_repair.py` (intelligent fixes)
**Step 5**: Agent orchestration (theater detection)
**Step 6**: `recovery_pipeline.sh` (sandbox validation)
**Step 7**: `recovery_pipeline.sh` (differential analysis)
**Step 8**: `recovery_pipeline.sh` (GitHub feedback)

## Dependencies

**Python** (3.8+):
- `dataclasses` (built-in)
- `json` (built-in)
- `pathlib` (built-in)
- `typing` (built-in)

**Bash** (4.0+):
- Standard utilities (`jq`, `gh`, `git`)

**Node.js** (14+):
- For JSON processing in pipeline scripts

**System**:
- GitHub CLI (`gh`) authenticated
- Git repository with remote

## Performance Targets

**Analysis**:
- Failure detection: < 1s per 100 failures
- Root cause analysis: < 1s per 100 failures
- Graph construction: < 100ms

**Repair**:
- Fix plan generation: < 1s per root cause
- Fix implementation: < 2s per fix
- Validation simulation: < 500ms per fix

**Pipeline**:
- Complete 8-step pipeline: < 10 minutes
- Individual steps: 30s - 2 minutes each

## Error Handling

**Retry Logic**:
- Max attempts: 3
- Backoff: Exponential

**Fallbacks**:
- Consensus fail → Manual review
- Sandbox fail → Retry with verbose logging
- All fixes rejected → Alert + feedback

**Notifications**:
- Critical failures: Always
- Theater detected: Always
- All fixes rejected: Always

## Best Practices

1. **Always run tests** before full pipeline
2. **Review config template** for your project
3. **Customize pattern templates** for your tech stack
4. **Monitor cascade ratios** (>50% indicates systemic issues)
5. **Track historical patterns** for trend analysis
6. **Use architectural strategy** for high coupling (prevents technical debt)

## Troubleshooting

**Issue**: Root causes not detected
**Solution**: Check Gemini analysis for dependency graph, verify parsed failures format

**Issue**: Fix strategy always isolated
**Solution**: Ensure connascence analysis is running, check coupling detection thresholds

**Issue**: All fixes rejected
**Solution**: Review validation logs, check sandbox environment matches production

**Issue**: Pipeline hangs
**Solution**: Check GitHub CLI authentication, verify network connectivity

## Version History

**2.0.0** (Current):
- Program-of-Thought fix generation
- Connascence-aware bundling
- Dual validation (sandbox + theater)
- Graph-based cascade detection
- Real-time failure monitoring

**1.0.0**:
- Basic failure detection
- Simple fix generation
- Manual validation

## License

Part of Three-Loop Integrated Development System
See main project LICENSE for details

## Support

For issues or questions:
1. Check test files for examples
2. Review SKILL.md for full workflow
3. Consult configuration template
4. Review script docstrings

---

**Status**: Production Ready ✅
**Last Updated**: 2025-11-02
**Maintainer**: Three-Loop System Team
