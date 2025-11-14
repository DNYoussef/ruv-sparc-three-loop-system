# Dogfooding Implementation Guide

**Version**: 1.0.0
**Last Updated**: 2025-01-13
**Audience**: Development team implementing the 6-week dogfooding plan
**Status**: Production Ready

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Week-by-Week Implementation](#week-by-week-implementation)
3. [Daily Workflows](#daily-workflows)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Metrics Dashboard Guide](#metrics-dashboard-guide)
6. [Emergency Procedures](#emergency-procedures)
7. [Post-Dogfooding Maintenance](#post-dogfooding-maintenance)

---

## Getting Started

### Prerequisites

**Required Setup:**
```bash
# 1. Verify Python environment
python --version  # Should be 3.9+

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify Connascence Analyzer
python -c "from connascence_analyzer import UnifiedConnascenceAnalyzer; print('OK')"

# 4. Verify ChromaDB
python -c "import chromadb; print(f'ChromaDB: {chromadb.__version__}')"

# 5. Check git hooks
ls -la .git/hooks/
# Should see: pre-commit, post-commit, pre-push

# 6. Run initial health check
python scripts/health_check.py
```

**Expected Output:**
```
System Health Check
===================
[PASS] Python 3.9+
[PASS] Dependencies installed
[PASS] Connascence Analyzer available
[PASS] ChromaDB accessible
[PASS] Git hooks installed
[PASS] Memory MCP reachable

Health Score: 100%
Ready to begin dogfooding!
```

### Team Assignments

**Core Team (4 people minimum):**

| Role | Responsibilities | Time Commitment |
|------|-----------------|-----------------|
| **Lead Developer** | Architecture decisions, PR reviews, Gate approvals | 8h/week |
| **Quality Engineer** | Clarity Linter development, metrics tracking | 10h/week |
| **Backend Developer** | Core fixes (ISSUE-001 to ISSUE-004) | 12h/week |
| **DevOps Engineer** | CI/CD setup, automation, monitoring | 6h/week |

**Extended Team (optional):**
- Documentation specialist (4h/week)
- Security reviewer (2h/week)
- Performance analyst (2h/week)

### Communication Channels

**Required:**
- **Daily Standup**: 15 minutes, 9:00 AM
- **Weekly Review**: 1 hour, Friday 3:00 PM
- **Slack Channel**: `#dogfooding-implementation`
- **Issue Tracker**: GitHub Issues with labels:
  - `dogfooding`, `quality-gate`, `clarity-linter`, `blocker`

**Daily Standup Structure:**
```
1. What did you complete yesterday?
2. What will you work on today?
3. Any blockers?
4. Quality Gate status update
```

**Weekly Review Agenda:**
```
1. Metrics review (10 min)
2. Gate progression status (10 min)
3. Blocker resolution (15 min)
4. Plan next week (15 min)
5. Celebrate wins (10 min)
```

---

## Week-by-Week Implementation

### Week 1: Foundation (Gate 1 Activation)

**Goal**: Fix critical issues, activate basic quality gates

#### Day 1: Environment Setup

**Morning (2h):**
```bash
# 1. Clone and branch
git checkout -b dogfooding/week1-foundation
git pull origin main

# 2. Install pre-commit hooks
cd .git/hooks
ln -sf ../../scripts/git-hooks/pre-commit pre-commit
ln -sf ../../scripts/git-hooks/post-commit post-commit
chmod +x pre-commit post-commit

# 3. Verify hook execution
git add README.md
git commit -m "test: Verify hooks"
# Should see: [Hook] Pre-commit quality check...
git reset HEAD~1

# 4. Initialize metrics tracking
python scripts/init_metrics_db.py
```

**Afternoon (3h):**
```bash
# 5. Run baseline analysis
python scripts/baseline_analysis.py --output reports/baseline_week1.json

# 6. Review baseline report
cat reports/baseline_week1.json | jq '.summary'

# Expected violations:
# - God Object (UnifiedConnascenceAnalyzer): 1
# - Missing Method (detect_cop): 1
# - Thin Helpers: 2
# - Long Functions: 1
# Total: 5 critical violations

# 7. Create GitHub issues
python scripts/create_github_issues.py --from-report reports/baseline_week1.json
```

**End of Day Checklist:**
- [ ] Environment verified
- [ ] Git hooks active
- [ ] Baseline metrics captured
- [ ] 5 GitHub issues created (ISSUE-001 to ISSUE-005)
- [ ] Team notified in Slack

---

#### Day 2-3: Fix ISSUE-001 (Detector Pool Refactor)

**ISSUE-001**: God Object - UnifiedConnascenceAnalyzer (26 methods)

**Day 2 Morning (3h): Plan Refactoring**
```bash
# 1. Create feature branch
git checkout -b fix/issue-001-detector-pool

# 2. Analyze current structure
python scripts/analyze_class.py UnifiedConnascenceAnalyzer

# Output shows:
# - 26 methods
# - 4 distinct responsibility clusters:
#   * File system operations (3 methods)
#   * Violation detection (8 methods)
#   * Result aggregation (6 methods)
#   * Report generation (5 methods)
#   * Utility methods (4 methods)

# 3. Design new structure
cat > design/detector_pool_refactor.md << EOF
# Detector Pool Refactoring

## New Classes:
1. FileSystemScanner (3 methods)
2. ViolationDetectorPool (8 methods)
3. ResultAggregator (6 methods)
4. ReportGenerator (5 methods)
5. AnalysisContext (4 methods)

## Migration Strategy:
- Extract classes one by one
- Update tests after each extraction
- Keep old class as facade initially
- Remove facade after all tests pass
EOF
```

**Day 2 Afternoon (4h): Extract FileSystemScanner**
```python
# File: connascence_analyzer/file_system_scanner.py

class FileSystemScanner:
    """Handles file system traversal and filtering."""

    def __init__(self, exclude_patterns=None):
        self.exclude_patterns = exclude_patterns or []

    def scan_directory(self, path, extensions=None):
        """Scan directory for Python files."""
        # Move logic from UnifiedConnascenceAnalyzer._scan_directory
        pass

    def filter_files(self, files):
        """Apply exclude patterns."""
        # Move logic from UnifiedConnascenceAnalyzer._filter_files
        pass

    def validate_path(self, path):
        """Validate file path."""
        # Move logic from UnifiedConnascenceAnalyzer._validate_path
        pass
```

```bash
# Run tests
python -m pytest tests/test_file_system_scanner.py -v

# Update UnifiedConnascenceAnalyzer to use new class
# In connascence_analyzer/__init__.py:
from .file_system_scanner import FileSystemScanner

class UnifiedConnascenceAnalyzer:
    def __init__(self):
        self.fs_scanner = FileSystemScanner()

    def _scan_directory(self, path):
        return self.fs_scanner.scan_directory(path)
```

**Day 3 Morning (4h): Extract ViolationDetectorPool**
```python
# File: connascence_analyzer/detector_pool.py

class ViolationDetectorPool:
    """Manages violation detectors with parallel execution."""

    def __init__(self, max_workers=4):
        self.detectors = {}
        self.max_workers = max_workers

    def register_detector(self, name, detector_class):
        """Register a violation detector."""
        self.detectors[name] = detector_class()

    def detect_all(self, ast_tree, context):
        """Run all detectors in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                name: executor.submit(detector.detect, ast_tree, context)
                for name, detector in self.detectors.items()
            }
            return {
                name: future.result()
                for name, future in futures.items()
            }
```

**Day 3 Afternoon (3h): Integration & Testing**
```bash
# 1. Run full test suite
python -m pytest tests/ -v --cov=connascence_analyzer

# 2. Check method count
python scripts/analyze_class.py UnifiedConnascenceAnalyzer
# Should show: 15 methods (down from 26)

# 3. Run baseline again
python scripts/baseline_analysis.py --output reports/after_issue001.json

# 4. Compare metrics
python scripts/compare_metrics.py reports/baseline_week1.json reports/after_issue001.json

# Expected improvement:
# - God Object violations: 1 -> 0 (RESOLVED)

# 5. Commit and push
git add .
git commit -m "fix: Extract detector pool and file system scanner (ISSUE-001)

- Extract FileSystemScanner (3 methods)
- Extract ViolationDetectorPool (8 methods)
- Reduce UnifiedConnascenceAnalyzer from 26 to 15 methods
- Add comprehensive tests for new classes
- Update documentation

Closes #ISSUE-001"

git push origin fix/issue-001-detector-pool

# 6. Create PR
gh pr create --title "Fix ISSUE-001: Detector Pool Refactor" \
  --body "Resolves God Object violation by extracting specialized classes."
```

---

#### Day 3-4: Fix ISSUE-002 (Missing Method)

**ISSUE-002**: Missing Method - `detect_cop` not implemented

**Day 3 Evening (2h): Implement detect_cop**
```python
# File: connascence_analyzer/detectors/cop_detector.py

class ParameterBombDetector:
    """Detects Connascence of Position (CoP) violations."""

    NASA_MAX_PARAMS = 6  # NASA coding standard limit

    def detect_cop(self, ast_tree, context):
        """
        Detect functions/methods with excessive parameters.

        NASA Standard: Maximum 6 parameters
        Rationale: Reduces cognitive load, improves testability
        """
        violations = []

        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                param_count = len(node.args.args)

                if param_count > self.NASA_MAX_PARAMS:
                    violations.append({
                        'type': 'CoP',
                        'severity': 'HIGH',
                        'line': node.lineno,
                        'function': node.name,
                        'param_count': param_count,
                        'threshold': self.NASA_MAX_PARAMS,
                        'message': f"Function '{node.name}' has {param_count} parameters (NASA limit: {self.NASA_MAX_PARAMS})",
                        'suggestion': 'Consider using a configuration object or builder pattern'
                    })

        return violations
```

**Day 4 Morning (2h): Add Tests**
```python
# File: tests/test_cop_detector.py

import ast
import pytest
from connascence_analyzer.detectors.cop_detector import ParameterBombDetector

def test_detect_cop_under_threshold():
    code = """
def valid_function(a, b, c, d, e):
    pass
"""
    tree = ast.parse(code)
    detector = ParameterBombDetector()
    violations = detector.detect_cop(tree, {})
    assert len(violations) == 0

def test_detect_cop_over_threshold():
    code = """
def invalid_function(a, b, c, d, e, f, g, h):
    pass
"""
    tree = ast.parse(code)
    detector = ParameterBombDetector()
    violations = detector.detect_cop(tree, {})

    assert len(violations) == 1
    assert violations[0]['type'] == 'CoP'
    assert violations[0]['param_count'] == 8
    assert violations[0]['threshold'] == 6

def test_detect_cop_async_function():
    code = """
async def async_function(a, b, c, d, e, f, g):
    await something()
"""
    tree = ast.parse(code)
    detector = ParameterBombDetector()
    violations = detector.detect_cop(tree, {})
    assert len(violations) == 1

def test_detect_cop_suggestion():
    code = """
def bad_function(a, b, c, d, e, f, g):
    pass
"""
    tree = ast.parse(code)
    detector = ParameterBombDetector()
    violations = detector.detect_cop(tree, {})
    assert 'configuration object' in violations[0]['suggestion']
```

**Day 4 Afternoon (3h): Integration**
```bash
# 1. Register detector in pool
# In connascence_analyzer/detector_pool.py:
from .detectors.cop_detector import ParameterBombDetector

class ViolationDetectorPool:
    def __init__(self):
        self.register_detector('cop', ParameterBombDetector)

# 2. Run tests
python -m pytest tests/test_cop_detector.py -v

# 3. Test on real codebase
python scripts/test_detector.py --detector cop --file connascence_analyzer/__init__.py

# 4. Commit
git add .
git commit -m "feat: Implement CoP detector (ISSUE-002)

- Add ParameterBombDetector with NASA 6-parameter limit
- Comprehensive test coverage
- Integration with ViolationDetectorPool
- Suggestion engine for refactoring

Closes #ISSUE-002"

git push origin fix/issue-002-cop-detector
gh pr create --title "Fix ISSUE-002: Implement CoP Detector"
```

---

#### Day 4-5: Fix ISSUE-003 & ISSUE-004 (Thin Helpers & Long Functions)

**ISSUE-003**: Thin Helpers (2 violations)
**ISSUE-004**: Long Function (1 violation)

**Day 4 Evening (2h): Identify Thin Helpers**
```bash
# Run thin helper detector
python scripts/detect_thin_helpers.py --file connascence_analyzer/utils.py

# Output:
# - format_violation_report(): 3 lines (just calls json.dumps)
# - validate_threshold(): 2 lines (just compares numbers)
```

**Day 5 Morning (3h): Inline Thin Helpers**
```python
# Before (utils.py):
def format_violation_report(violations):
    return json.dumps(violations, indent=2)

def validate_threshold(value, threshold):
    return value > threshold

# After: Inline at call sites
# In report_generator.py:
report = json.dumps(violations, indent=2)  # Direct call
if param_count > self.NASA_MAX_PARAMS:    # Direct comparison
```

```bash
# Remove thin helpers
git rm connascence_analyzer/utils.py

# Update imports
find . -name "*.py" -exec sed -i 's/from .utils import.*//g' {} \;

# Run tests
python -m pytest tests/ -v

# Commit
git commit -m "refactor: Remove thin helper wrappers (ISSUE-003)

- Inline format_violation_report (3 lines)
- Inline validate_threshold (2 lines)
- Reduces indirection and improves clarity

Closes #ISSUE-003"
```

**Day 5 Afternoon (4h): Split Long Function**
```python
# ISSUE-004: analyze_workspace() is 72 lines (threshold: 50)

# Before:
def analyze_workspace(self, path):
    # Lines 1-20: File scanning
    # Lines 21-40: AST parsing
    # Lines 41-60: Violation detection
    # Lines 61-72: Report generation
    pass

# After: Extract 3 sub-methods
def analyze_workspace(self, path):
    files = self._scan_workspace(path)
    trees = self._parse_files(files)
    violations = self._detect_violations(trees)
    return self._generate_report(violations)

def _scan_workspace(self, path):
    # 15 lines
    pass

def _parse_files(self, files):
    # 15 lines
    pass

def _detect_violations(self, trees):
    # 18 lines
    pass

def _generate_report(self, violations):
    # 10 lines
    pass
```

```bash
# Run tests
python -m pytest tests/ -v

# Check line counts
python scripts/check_line_counts.py connascence_analyzer/__init__.py

# Commit
git commit -m "refactor: Split long analyze_workspace method (ISSUE-004)

- Extract _scan_workspace (15 lines)
- Extract _parse_files (15 lines)
- Extract _detect_violations (18 lines)
- Extract _generate_report (10 lines)
- Main method now 4 lines (was 72)

Closes #ISSUE-004"

git push origin fix/issue-003-004-refactoring
gh pr create --title "Fix ISSUE-003 & ISSUE-004: Refactoring"
```

---

#### Day 5 End: Activate Gate 1

**Gate 1 Criteria:**
- [x] ISSUE-001 resolved (God Object)
- [x] ISSUE-002 resolved (Missing Method)
- [x] ISSUE-003 resolved (Thin Helpers)
- [x] ISSUE-004 resolved (Long Function)
- [x] All tests passing
- [x] CI green

**Activate Gate 1:**
```bash
# 1. Merge all PRs
gh pr merge fix/issue-001-detector-pool --squash
gh pr merge fix/issue-002-cop-detector --squash
gh pr merge fix/issue-003-004-refactoring --squash

# 2. Update main
git checkout main
git pull origin main

# 3. Run full analysis
python scripts/baseline_analysis.py --output reports/week1_complete.json

# 4. Activate Gate 1 in CI
# In .github/workflows/quality-gate.yml:
env:
  QUALITY_GATE_LEVEL: 1

git add .github/workflows/quality-gate.yml
git commit -m "chore: Activate Quality Gate 1"
git push origin main

# 5. First metrics collection
python scripts/collect_metrics.py --week 1 --gate 1

# Expected metrics:
# - God Object: 1 -> 0 (100% improvement)
# - Missing Methods: 1 -> 0 (100% improvement)
# - Thin Helpers: 2 -> 0 (100% improvement)
# - Long Functions: 1 -> 0 (100% improvement)
# - Total violations: 5 -> 0 (100% clean)
```

**End of Week 1 Celebration:**
```bash
# Post to Slack:
echo "Week 1 Complete! Gate 1 Activated!
- 5 critical issues resolved
- 100% test coverage maintained
- CI green
- Zero violations detected
Great work team! " | slack-cli post -c dogfooding-implementation
```

---

### Week 2: Clarity Linter MVP

**Goal**: Build basic Clarity Linter with 5 core rules, run first self-scan

#### Day 6: Clarity Linter Foundation

**Morning (4h): Project Setup**
```bash
# 1. Create Clarity Linter directory
mkdir -p clarity_linter/{rules,reporters,cli}
cd clarity_linter

# 2. Initialize package
cat > __init__.py << EOF
"""
Clarity Linter - Python code quality linter
Focuses on code clarity, readability, and maintainability
"""
__version__ = '0.1.0'

from .linter import ClarityLinter
from .rules import Rule

__all__ = ['ClarityLinter', 'Rule']
EOF

# 3. Create base linter class
cat > linter.py << EOF
import ast
from pathlib import Path
from .rules.registry import RuleRegistry

class ClarityLinter:
    """Main linter class for analyzing Python code clarity."""

    def __init__(self):
        self.registry = RuleRegistry()
        self.violations = []

    def lint_file(self, file_path):
        """Analyze a single file."""
        with open(file_path, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        for rule in self.registry.get_enabled_rules():
            violations = rule.check(tree, file_path)
            self.violations.extend(violations)

        return self.violations

    def lint_directory(self, dir_path):
        """Recursively analyze a directory."""
        violations = []
        for py_file in Path(dir_path).rglob('*.py'):
            violations.extend(self.lint_file(py_file))
        return violations
EOF
```

**Afternoon (3h): Rule Base Class**
```python
# File: clarity_linter/rules/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Violation:
    """Represents a clarity violation."""
    rule_id: str
    rule_name: str
    severity: str  # ERROR, WARNING, INFO
    file_path: str
    line_number: int
    column: int
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None

class Rule(ABC):
    """Base class for all clarity rules."""

    rule_id: str = "CL000"
    rule_name: str = "BaseRule"
    severity: str = "WARNING"
    description: str = ""

    @abstractmethod
    def check(self, tree: ast.AST, file_path: str) -> List[Violation]:
        """Check code against this rule."""
        pass

    def create_violation(self, node, message, suggestion=None):
        """Helper to create a violation."""
        return Violation(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            severity=self.severity,
            file_path=self.file_path,
            line_number=node.lineno,
            column=node.col_offset,
            message=message,
            suggestion=suggestion
        )
```

---

#### Day 7-8: Implement 5 Core Rules

**Core Rules:**
1. CL001: God Object Detection
2. CL002: Parameter Bomb Detection
3. CL003: Deep Nesting Detection
4. CL004: Long Function Detection
5. CL005: Magic Literal Detection

**Day 7 Morning (3h): CL001 - God Object Rule**
```python
# File: clarity_linter/rules/cl001_god_object.py

from .base import Rule, Violation
import ast

class GodObjectRule(Rule):
    """Detect classes with too many methods (God Objects)."""

    rule_id = "CL001"
    rule_name = "GodObject"
    severity = "ERROR"
    description = "Classes should have a single responsibility (max 15 methods)"

    MAX_METHODS = 15

    def check(self, tree, file_path):
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                method_count = sum(
                    1 for item in node.body
                    if isinstance(item, ast.FunctionDef)
                )

                if method_count > self.MAX_METHODS:
                    violations.append(self.create_violation(
                        node,
                        f"Class '{node.name}' has {method_count} methods (max: {self.MAX_METHODS})",
                        f"Consider splitting into {(method_count // self.MAX_METHODS) + 1} classes by responsibility"
                    ))

        return violations
```

**Day 7 Afternoon (3h): CL002 - Parameter Bomb Rule**
```python
# File: clarity_linter/rules/cl002_parameter_bomb.py

class ParameterBombRule(Rule):
    """Detect functions with too many parameters."""

    rule_id = "CL002"
    rule_name = "ParameterBomb"
    severity = "ERROR"
    description = "Functions should have <=6 parameters (NASA standard)"

    NASA_MAX_PARAMS = 6

    def check(self, tree, file_path):
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                param_count = len(node.args.args)

                if param_count > self.NASA_MAX_PARAMS:
                    violations.append(self.create_violation(
                        node,
                        f"Function '{node.name}' has {param_count} parameters (NASA max: {self.NASA_MAX_PARAMS})",
                        "Use a configuration object or builder pattern"
                    ))

        return violations
```

**Day 8 Morning (4h): CL003, CL004, CL005**
```python
# File: clarity_linter/rules/cl003_deep_nesting.py

class DeepNestingRule(Rule):
    """Detect excessive nesting depth."""

    rule_id = "CL003"
    rule_name = "DeepNesting"
    severity = "WARNING"
    NASA_MAX_DEPTH = 4

    def check(self, tree, file_path):
        violations = []

        def get_nesting_depth(node, depth=0):
            max_depth = depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                    child_depth = get_nesting_depth(child, depth + 1)
                    max_depth = max(max_depth, child_depth)
            return max_depth

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                depth = get_nesting_depth(node)
                if depth > self.NASA_MAX_DEPTH:
                    violations.append(self.create_violation(
                        node,
                        f"Function '{node.name}' has nesting depth {depth} (NASA max: {self.NASA_MAX_DEPTH})",
                        "Extract nested logic into separate functions"
                    ))

        return violations

# File: clarity_linter/rules/cl004_long_function.py

class LongFunctionRule(Rule):
    """Detect functions that are too long."""

    rule_id = "CL004"
    rule_name = "LongFunction"
    severity = "WARNING"
    MAX_LINES = 50

    def check(self, tree, file_path):
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Calculate line count
                if hasattr(node, 'end_lineno'):
                    line_count = node.end_lineno - node.lineno + 1

                    if line_count > self.MAX_LINES:
                        violations.append(self.create_violation(
                            node,
                            f"Function '{node.name}' is {line_count} lines (max: {self.MAX_LINES})",
                            "Split into smaller, focused functions"
                        ))

        return violations

# File: clarity_linter/rules/cl005_magic_literal.py

class MagicLiteralRule(Rule):
    """Detect magic literals (hardcoded values)."""

    rule_id = "CL005"
    rule_name = "MagicLiteral"
    severity = "INFO"

    ALLOWED_LITERALS = {0, 1, -1, None, True, False, '', []}

    def check(self, tree, file_path):
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                value = node.value

                if value not in self.ALLOWED_LITERALS and not isinstance(value, str):
                    violations.append(self.create_violation(
                        node,
                        f"Magic literal '{value}' should be a named constant",
                        f"Define as constant: MAX_RETRIES = {value}"
                    ))

        return violations
```

**Day 8 Afternoon (3h): Rule Registry**
```python
# File: clarity_linter/rules/registry.py

from .cl001_god_object import GodObjectRule
from .cl002_parameter_bomb import ParameterBombRule
from .cl003_deep_nesting import DeepNestingRule
from .cl004_long_function import LongFunctionRule
from .cl005_magic_literal import MagicLiteralRule

class RuleRegistry:
    """Registry for all clarity rules."""

    def __init__(self):
        self.rules = {
            'CL001': GodObjectRule(),
            'CL002': ParameterBombRule(),
            'CL003': DeepNestingRule(),
            'CL004': LongFunctionRule(),
            'CL005': MagicLiteralRule(),
        }
        self.enabled = set(self.rules.keys())

    def get_enabled_rules(self):
        return [self.rules[rule_id] for rule_id in self.enabled]

    def enable_rule(self, rule_id):
        self.enabled.add(rule_id)

    def disable_rule(self, rule_id):
        self.enabled.discard(rule_id)
```

---

#### Day 9: CLI & Testing

**Morning (3h): CLI Interface**
```python
# File: clarity_linter/cli/main.py

import click
from ..linter import ClarityLinter

@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['text', 'json', 'junit']), default='text')
@click.option('--severity', type=click.Choice(['ERROR', 'WARNING', 'INFO']), default='WARNING')
@click.option('--rules', help='Comma-separated rule IDs to enable')
def lint(path, format, severity, rules):
    """Run Clarity Linter on PATH."""
    linter = ClarityLinter()

    if rules:
        # Enable only specified rules
        rule_ids = [r.strip() for r in rules.split(',')]
        linter.registry.enabled = set(rule_ids)

    violations = linter.lint_directory(path)

    # Filter by severity
    severity_order = {'ERROR': 3, 'WARNING': 2, 'INFO': 1}
    min_severity = severity_order[severity]
    filtered = [v for v in violations if severity_order[v.severity] >= min_severity]

    # Output
    if format == 'text':
        for v in filtered:
            click.echo(f"{v.file_path}:{v.line_number}:{v.column}: {v.rule_id} {v.message}")
            if v.suggestion:
                click.echo(f"  Suggestion: {v.suggestion}")
    elif format == 'json':
        import json
        click.echo(json.dumps([vars(v) for v in filtered], indent=2))

    # Exit code
    error_count = sum(1 for v in filtered if v.severity == 'ERROR')
    return 1 if error_count > 0 else 0

if __name__ == '__main__':
    lint()
```

**Afternoon (4h): Comprehensive Tests**
```python
# File: tests/test_clarity_linter.py

import pytest
from clarity_linter import ClarityLinter
from clarity_linter.rules import GodObjectRule, ParameterBombRule

def test_god_object_detection():
    code = """
class BadClass:
    def m1(self): pass
    def m2(self): pass
    # ... 16 methods total
"""
    linter = ClarityLinter()
    violations = linter.lint_string(code, 'test.py')

    assert len(violations) == 1
    assert violations[0].rule_id == 'CL001'

def test_parameter_bomb_detection():
    code = """
def bad_function(a, b, c, d, e, f, g, h):
    pass
"""
    linter = ClarityLinter()
    violations = linter.lint_string(code, 'test.py')

    assert len(violations) == 1
    assert violations[0].rule_id == 'CL002'

def test_cli_integration():
    from click.testing import CliRunner
    from clarity_linter.cli.main import lint

    runner = CliRunner()
    result = runner.invoke(lint, ['test_files/', '--format', 'json'])

    assert result.exit_code == 0 or result.exit_code == 1
    assert 'violations' in result.output or '[]' in result.output
```

---

#### Day 10: First Self-Scan

**Morning (2h): Run Clarity Linter on Itself**
```bash
# 1. Install Clarity Linter
pip install -e .

# 2. Run on Connascence Analyzer
clarity-lint connascence_analyzer/ --format json --severity ERROR > reports/clarity_scan_week2.json

# 3. Review results
cat reports/clarity_scan_week2.json | jq '.violations | length'

# Expected violations (report only, don't fix yet):
# - CL001 (God Object): 0 (fixed in Week 1)
# - CL002 (Parameter Bomb): 0 (fixed in Week 1)
# - CL003 (Deep Nesting): 2-3 instances
# - CL004 (Long Function): 1-2 instances
# - CL005 (Magic Literal): 5-10 instances (INFO level)

# 4. Run on Clarity Linter itself
clarity-lint clarity_linter/ --format json > reports/clarity_self_scan.json

# 5. Store results in Memory MCP
python scripts/store_clarity_results.py --report reports/clarity_scan_week2.json
```

**Afternoon (3h): Create Unified Quality Gate**
```python
# File: scripts/unified_quality_gate.py

from connascence_analyzer import UnifiedConnascenceAnalyzer
from clarity_linter import ClarityLinter

class UnifiedQualityGate:
    """Unified quality gate combining Connascence and Clarity checks."""

    def __init__(self, gate_level=1):
        self.connascence = UnifiedConnascenceAnalyzer()
        self.clarity = ClarityLinter()
        self.gate_level = gate_level

    def check(self, path):
        """Run both analyzers and combine results."""
        connascence_violations = self.connascence.analyze_workspace(path)
        clarity_violations = self.clarity.lint_directory(path)

        return {
            'connascence': connascence_violations,
            'clarity': clarity_violations,
            'gate_level': self.gate_level,
            'passed': self._evaluate_gate(connascence_violations, clarity_violations)
        }

    def _evaluate_gate(self, conn_violations, clarity_violations):
        """Determine if quality gate passed."""
        if self.gate_level == 1:
            # Gate 1: No ERROR-level violations
            conn_errors = [v for v in conn_violations if v['severity'] == 'HIGH']
            clarity_errors = [v for v in clarity_violations if v.severity == 'ERROR']
            return len(conn_errors) == 0 and len(clarity_errors) == 0

        elif self.gate_level == 2:
            # Gate 2: No ERROR or WARNING violations
            conn_issues = [v for v in conn_violations if v['severity'] in ['HIGH', 'MEDIUM']]
            clarity_issues = [v for v in clarity_violations if v.severity in ['ERROR', 'WARNING']]
            return len(conn_issues) == 0 and len(clarity_issues) == 0

        # Add more gate levels...
        return True

if __name__ == '__main__':
    gate = UnifiedQualityGate(gate_level=1)
    result = gate.check('.')
    print(f"Gate {result['gate_level']}: {'PASSED' if result['passed'] else 'FAILED'}")
```

**End of Week 2:**
```bash
# Test unified gate
python scripts/unified_quality_gate.py

# Output:
# Connascence violations: 0
# Clarity violations (ERROR): 0
# Clarity violations (WARNING): 3
# Gate 1: PASSED

# Commit Clarity Linter
git add clarity_linter/ tests/ scripts/unified_quality_gate.py
git commit -m "feat: Add Clarity Linter MVP with 5 core rules

- CL001: God Object Detection
- CL002: Parameter Bomb Detection
- CL003: Deep Nesting Detection
- CL004: Long Function Detection
- CL005: Magic Literal Detection
- CLI interface with JSON/text output
- Integration with UnifiedQualityGate
- First self-scan completed (report only)"

git push origin main
```

---

### Week 3-4: Major Refactoring

**Goal**: Refactor remaining violations, activate Gate 2

#### Week 3 Overview

**Day 11-12**: Refactor Deep Nesting violations (2-3 instances)
**Day 13-14**: Refactor Long Function violations (1-2 instances)
**Day 15**: Code review and testing

#### Week 4 Overview

**Day 16-17**: Extract specialized classes from UnifiedConnascenceAnalyzer
**Day 18-19**: Performance optimization
**Day 20**: Activate Gate 2

**Detailed implementation similar to Week 1-2...**

---

### Week 5: Dogfooding Activation

**Goal**: Run full self-analysis, fix all remaining violations

#### Day 21: Full Self-Analysis

```bash
# Run unified gate on entire codebase
python scripts/unified_quality_gate.py --gate-level 3 --path . --output reports/week5_full_scan.json

# Create GitHub issues for all violations
python scripts/create_github_issues.py --from-report reports/week5_full_scan.json --auto-assign

# Expected: 5-10 minor violations (INFO/WARNING level)
```

#### Day 22-24: Fix Remaining Violations

(Similar pattern to Week 1-2)

#### Day 25: Scaffolding Cleanup

```bash
# Remove temporary test files
find . -name "test_*.py.bak" -delete
find . -name "*.pyc" -delete

# Remove debug code
grep -r "print(" --include="*.py" . | grep -v "# DEBUG-KEEP"

# Clean up comments
python scripts/clean_comments.py --remove-old-todos
```

---

### Week 6: Full Dogfooding

**Goal**: Achieve Gate 4 (zero violations), document everything

#### Day 26-28: Gate 4 Activation

```bash
# Final self-scan
python scripts/unified_quality_gate.py --gate-level 4 --strict

# Expected: 0 violations

# Activate Gate 4 in CI
# In .github/workflows/quality-gate.yml:
env:
  QUALITY_GATE_LEVEL: 4
  STRICT_MODE: true
```

#### Day 29: Documentation

```bash
# Generate final metrics
python scripts/generate_final_metrics.py --weeks 1-6

# Create summary report
python scripts/create_summary_report.py --output DOGFOODING_SUCCESS_REPORT.md
```

#### Day 30: Celebration

**Celebration Checklist:**
- [ ] Team retrospective meeting
- [ ] Publish blog post about dogfooding journey
- [ ] Demo to stakeholders
- [ ] Pizza party
- [ ] Update team metrics dashboard

---

## Daily Workflows

### Morning Routine (15 minutes)

```bash
# 1. Check CI status
gh workflow view quality-gate --branch main

# 2. Pull latest changes
git checkout main
git pull origin main

# 3. Run quick quality check
python scripts/quick_quality_check.py

# 4. Check Memory MCP for overnight updates
npx claude-flow@alpha memory retrieve --key "dogfooding/overnight-updates"

# 5. Daily standup (see Communication Channels section)
```

### Development Workflow

**Before Starting Work:**
```bash
# 1. Create feature branch
git checkout -b fix/issue-XXX-description

# 2. Run baseline check
python scripts/baseline_analysis.py --output reports/baseline_$(date +%Y%m%d).json

# 3. Review issue details
gh issue view XXX
```

**During Development:**
```bash
# 1. Run pre-commit hooks automatically
git add file.py
git commit -m "feat: Your change"
# Hooks run automatically:
# - Clarity Linter check
# - Connascence analysis
# - Test suite
# - Code formatting

# 2. Manual quality check (if needed)
clarity-lint file.py --format text

# 3. Store progress in Memory MCP
npx claude-flow@alpha memory store \
  --key "dogfooding/progress/issue-XXX" \
  --value "Completed 60% - Tests passing"
```

**Evening Review:**
```bash
# 1. Review auto-generated issues
gh issue list --label "auto-generated" --limit 10

# 2. Update todo list
python scripts/update_todos.py --completed "Fixed ISSUE-XXX"

# 3. Push changes
git push origin fix/issue-XXX-description

# 4. Create PR if ready
gh pr create --title "Fix ISSUE-XXX" --body "Description..."
```

### Weekly Review Workflow

**Every Friday 3:00 PM:**
```bash
# 1. Generate weekly metrics
python scripts/generate_weekly_metrics.py --week $(date +%U)

# 2. Review gate progression
python scripts/gate_status.py

# 3. Identify blockers
gh issue list --label "blocker" --state open

# 4. Plan next week
python scripts/plan_next_week.py --output plans/week_$(date +%U).md
```

---

## Troubleshooting Guide

### Common Issues and Fixes

#### Issue 1: CI Completely Broken

**Symptoms:**
- All PRs failing
- `quality-gate.yml` workflow failing
- Error: "Unable to import connascence_analyzer"

**Diagnosis:**
```bash
# Check Python environment
python --version
pip list | grep connascence

# Check import paths
python -c "import connascence_analyzer; print(connascence_analyzer.__file__)"
```

**Fix:**
```bash
# Option A: Reinstall dependencies
pip uninstall connascence-analyzer
pip install -e .

# Option B: Rollback to last working commit
git log --oneline -10  # Find last green commit
git revert HEAD~3..HEAD  # Revert last 3 commits

# Option C: Disable gate temporarily (EMERGENCY ONLY)
# In .github/workflows/quality-gate.yml:
env:
  QUALITY_GATE_LEVEL: 0  # Disable
```

---

#### Issue 2: Too Many False Positives

**Symptoms:**
- Clarity Linter reporting 50+ violations
- Most violations seem incorrect
- Blocking legitimate work

**Diagnosis:**
```bash
# Run with debug mode
clarity-lint . --format json --debug > debug_output.json

# Analyze false positives
python scripts/analyze_false_positives.py --input debug_output.json
```

**Fix:**
```bash
# Option A: Adjust thresholds
# In clarity_linter/rules/cl004_long_function.py:
MAX_LINES = 60  # Was 50, increase temporarily

# Option B: Disable specific rule
clarity-lint . --rules CL001,CL002,CL003  # Exclude CL004

# Option C: Add suppression comments
# In code:
def long_but_necessary_function():
    # clarity-lint: disable=CL004
    # This function is long but cannot be easily split
    pass
```

---

#### Issue 3: Performance Degradation

**Symptoms:**
- CI taking >10 minutes (was 2 minutes)
- Local analysis very slow
- High CPU usage

**Diagnosis:**
```bash
# Profile analysis
python -m cProfile -o profile.stats scripts/unified_quality_gate.py
python -m pstats profile.stats
# (pstats) sort time
# (pstats) stats 10

# Check for regression
git bisect start
git bisect bad HEAD
git bisect good v1.0.0
# Git will checkout commits to test
```

**Fix:**
```bash
# Option A: Enable caching
# In .github/workflows/quality-gate.yml:
- uses: actions/cache@v3
  with:
    path: ~/.cache/clarity-linter
    key: ${{ runner.os }}-clarity-${{ hashFiles('**/*.py') }}

# Option B: Parallelize analysis
# In unified_quality_gate.py:
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(analyze_file, f) for f in files]

# Option C: Reduce scope
# Only analyze changed files in PRs:
git diff --name-only main...HEAD | grep "\.py$" | xargs clarity-lint
```

---

#### Issue 4: Memory MCP Connection Issues

**Symptoms:**
- Error: "Failed to connect to Memory MCP"
- Hooks not storing/retrieving data
- ChromaDB errors

**Diagnosis:**
```bash
# Check ChromaDB service
curl http://localhost:8000/api/v1/heartbeat

# Check Memory MCP configuration
cat ~/.config/mcp/memory-mcp-config.json

# Test direct connection
python -c "import chromadb; client = chromadb.Client(); print('OK')"
```

**Fix:**
```bash
# Option A: Restart ChromaDB
pkill -f chroma
chromadb run --host localhost --port 8000 &

# Option B: Clear corrupted database
rm -rf ~/.chroma
python scripts/init_memory_mcp.py

# Option C: Fallback to file storage
# In scripts/memory_fallback.py:
import json
def store_memory(key, value):
    with open(f".memory/{key}.json", "w") as f:
        json.dump(value, f)
```

---

#### Issue 5: Merge Conflicts in CI Config

**Symptoms:**
- Multiple PRs updating `.github/workflows/quality-gate.yml`
- Merge conflicts blocking progress

**Fix:**
```bash
# Standardize CI updates in single PR
git checkout -b chore/unify-ci-config

# Merge all pending changes
git merge fix/issue-XXX --no-commit
git merge fix/issue-YYY --no-commit

# Resolve conflicts manually
vim .github/workflows/quality-gate.yml

# Test locally
act -j quality-gate  # Using nektos/act

# Commit unified version
git add .github/workflows/quality-gate.yml
git commit -m "chore: Unify CI configuration from multiple PRs"
```

---

## Metrics Dashboard Guide

### Understanding Weekly Metrics

**Metrics Collected:**
- Violation counts by type
- Code quality trends
- Gate progression
- Team velocity
- Fix time average

**Accessing Metrics:**
```bash
# View current week
python scripts/view_metrics.py --week current

# View trend over time
python scripts/view_metrics.py --weeks 1-6 --chart

# Export to CSV
python scripts/export_metrics.py --format csv --output metrics.csv
```

### Sample Metrics Dashboard

```
Dogfooding Metrics - Week 3
===========================

Violations by Type:
- God Object:      0 (was 1)  [100% improvement]
- Parameter Bomb:  0 (was 1)  [100% improvement]
- Deep Nesting:    2 (was 5)  [60% improvement]
- Long Function:   1 (was 3)  [67% improvement]
- Magic Literal:   8 (was 15) [47% improvement]

Gate Status:
- Gate 1: PASSING (5/5 criteria)
- Gate 2: PENDING (3/5 criteria)
- Gate 3: NOT STARTED
- Gate 4: NOT STARTED

Team Velocity:
- Issues closed: 6
- PRs merged: 8
- Average fix time: 4.2 hours

Quality Trend:
Week 1: 5 violations
Week 2: 3 violations
Week 3: 2 violations  <- Current
Projected Week 6: 0 violations
```

### Setting Goals

**SMART Goals for Dogfooding:**
- **Specific**: Reduce violations from 5 to 0
- **Measurable**: Weekly metrics dashboard
- **Achievable**: 6-week timeline
- **Relevant**: Improves code quality
- **Time-bound**: Complete by [End Date]

---

## Emergency Procedures

### Emergency 1: CI Completely Blocked

**Severity**: CRITICAL
**Impact**: No PRs can merge, entire team blocked

**Immediate Actions (within 1 hour):**
```bash
# 1. Notify team
echo "CI EMERGENCY: Quality gate blocking all PRs. Investigating..." | slack-cli post -c dogfooding-implementation

# 2. Identify breaking commit
git log --oneline --all --graph -10
gh run list --workflow quality-gate --limit 5

# 3. Revert breaking commit
git revert [commit-sha]
git push origin main --force-with-lease

# 4. Create hotfix PR
gh pr create --title "HOTFIX: Revert breaking change" --body "Emergency revert to unblock CI"

# 5. Merge without review (if team lead approves)
gh pr merge --admin --squash
```

**Follow-up Actions (within 24 hours):**
- Post-mortem meeting
- Root cause analysis
- Update CI safeguards
- Document incident

---

### Emergency 2: False Positives Blocking Launch

**Severity**: HIGH
**Impact**: Production deployment delayed

**Immediate Actions:**
```bash
# 1. Identify false positives
python scripts/analyze_violations.py --filter false-positive

# 2. Suppress false positives (temporary)
# In problematic files:
# clarity-lint: disable=RULE_ID
# Justification: [Reason]

# 3. Create exception list
cat > .clarity-exceptions.yml << EOF
exceptions:
  - file: src/legacy_code.py
    rules: [CL001, CL004]
    reason: "Legacy code, scheduled for refactor in Q2"
EOF

# 4. Update gate to allow exceptions
# In unified_quality_gate.py:
def _evaluate_gate(self, violations):
    filtered = self._filter_exceptions(violations)
    return len(filtered) == 0
```

---

### Emergency 3: Performance Crisis

**Severity**: HIGH
**Impact**: CI takes >30 minutes, blocking development

**Immediate Actions:**
```bash
# 1. Enable fast mode (reduced checks)
# In .github/workflows/quality-gate.yml:
env:
  QUALITY_GATE_MODE: fast  # Skip expensive checks

# 2. Analyze changed files only
git diff --name-only ${{ github.base_ref }}...HEAD | grep "\.py$" > changed_files.txt
clarity-lint $(cat changed_files.txt)

# 3. Parallelize across runners
# In quality-gate.yml:
strategy:
  matrix:
    path: [src/, tests/, scripts/]
steps:
  - run: clarity-lint ${{ matrix.path }}
```

---

## Post-Dogfooding Maintenance

### Maintaining Zero Violations

**Daily Practices:**
```bash
# Pre-commit hooks (automatic)
# .git/hooks/pre-commit runs:
# 1. Clarity Lint on staged files
# 2. Connascence analysis on staged files
# 3. Block commit if violations found

# Manual pre-PR check
make quality-check

# CI enforcement (automatic)
# PR cannot merge if quality gate fails
```

**Weekly Practices:**
```bash
# Full codebase scan (Friday afternoon)
python scripts/weekly_quality_audit.py

# Review metrics
python scripts/view_metrics.py --week current

# Update thresholds if needed
python scripts/tune_thresholds.py --based-on-history
```

---

### Adding New Code Without Violations

**Best Practices:**

1. **Start with Tests (TDD)**
```python
# tests/test_new_feature.py
def test_new_feature():
    result = new_feature()
    assert result == expected
```

2. **Write Code to Pass Tests**
```python
# src/new_feature.py
def new_feature():
    # Keep functions <50 lines
    # Keep parameters <=6
    # Avoid deep nesting
    pass
```

3. **Run Quality Check Before Commit**
```bash
clarity-lint src/new_feature.py
python scripts/connascence_check.py src/new_feature.py
```

4. **Fix Violations Immediately**
- Don't accumulate violations
- Address feedback in real-time
- Use suggestions from linter

---

### Rule Tuning Process

**When to Tune Rules:**
- False positive rate >10%
- Team consensus that rule is too strict
- New Python features require adjustment

**Tuning Workflow:**
```bash
# 1. Collect data on false positives
python scripts/collect_false_positives.py --days 30

# 2. Analyze patterns
python scripts/analyze_tuning_needs.py

# 3. Propose threshold changes
# Example: Increase max function length 50 -> 60
cat > proposals/tune_cl004.md << EOF
## Proposal: Increase CL004 Threshold

Current: 50 lines
Proposed: 60 lines

Justification:
- 15 false positives in last month
- Functions are clear despite length
- Industry standard is 50-75 lines

Impact:
- 15 violations will be resolved
- No new violations expected
EOF

# 4. Team review and vote
# Require 2/3 majority

# 5. Implement change
# In clarity_linter/rules/cl004_long_function.py:
MAX_LINES = 60  # Updated from 50

# 6. Update documentation
# In docs/RULES.md:
# Update CL004 threshold and justification
```

---

### Quarterly Quality Audits

**Schedule:** First Monday of each quarter

**Audit Checklist:**
```bash
# 1. Run comprehensive analysis
python scripts/quarterly_audit.py --quarter Q1

# 2. Review metrics trends
python scripts/compare_quarters.py Q4 Q1

# 3. Identify areas for improvement
python scripts/identify_improvement_areas.py

# 4. Update quality standards
# Review and update:
# - Thresholds in rules
# - Gate criteria
# - Team practices

# 5. Celebrate achievements
python scripts/generate_achievement_report.py --quarter Q1
```

**Quarterly Review Meeting Agenda:**
```
1. Metrics Review (30 min)
   - Violation trends
   - Fix velocity
   - Team satisfaction

2. Rule Effectiveness (20 min)
   - Which rules are most valuable?
   - Which rules have too many false positives?
   - Should we add new rules?

3. Process Improvements (20 min)
   - What's working well?
   - What's frustrating?
   - Tool improvements needed?

4. Goals for Next Quarter (10 min)
   - New quality initiatives
   - Tooling improvements
   - Team training needs
```

---

## Appendix: Quick Reference Commands

### Daily Commands
```bash
# Morning health check
python scripts/health_check.py

# Run quality gate locally
python scripts/unified_quality_gate.py

# View current metrics
python scripts/view_metrics.py --week current

# Check CI status
gh workflow view quality-gate
```

### Development Commands
```bash
# Create feature branch
git checkout -b fix/issue-XXX

# Run Clarity Linter
clarity-lint file.py --format text

# Run Connascence Analysis
python -m connascence_analyzer analyze file.py

# Run unified gate
python scripts/unified_quality_gate.py --path .

# Commit with hooks
git add file.py
git commit -m "fix: Your change"
```

### Emergency Commands
```bash
# Revert last commit
git revert HEAD
git push origin main --force-with-lease

# Disable gate temporarily
# In .github/workflows/quality-gate.yml:
env:
  QUALITY_GATE_LEVEL: 0

# Clear ChromaDB
rm -rf ~/.chroma
python scripts/init_memory_mcp.py

# Fast mode CI
env QUALITY_GATE_MODE=fast python scripts/unified_quality_gate.py
```

---

## Success Criteria

**By End of Week 6:**
- [ ] Zero violations in entire codebase
- [ ] Gate 4 active and passing
- [ ] CI green for 7 consecutive days
- [ ] Team satisfaction score >8/10
- [ ] Documentation complete
- [ ] Automated issue creation working
- [ ] Metrics dashboard live
- [ ] Post-mortem completed
- [ ] Celebration held

**Long-term Success (3 months post-dogfooding):**
- [ ] Zero violations maintained
- [ ] <2 hour fix time for new violations
- [ ] No false positive complaints
- [ ] Quality practices adopted by team
- [ ] Other teams requesting Clarity Linter

---

## Support and Resources

**Documentation:**
- Main docs: `docs/`
- API docs: `docs/api/`
- Examples: `examples/`

**Communication:**
- Slack: `#dogfooding-implementation`
- Email: dogfooding-team@company.com
- Wiki: https://wiki.company.com/dogfooding

**Tools:**
- GitHub: https://github.com/company/connascence-analyzer
- CI Dashboard: https://ci.company.com/quality-gate
- Metrics: https://metrics.company.com/dogfooding

**Emergency Contacts:**
- Lead Developer: @lead-dev (Slack)
- DevOps: @devops-team (Slack)
- CTO: cto@company.com

---

**End of Implementation Guide**

Ready to start dogfooding? Begin with Week 1, Day 1: Environment Setup!
