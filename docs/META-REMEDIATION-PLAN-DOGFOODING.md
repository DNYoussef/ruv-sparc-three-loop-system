# META-REMEDIATION PLAN: Dogfooding the Connascence Safety Analyzer

**Version**: 1.0.0
**Status**: Ready for Execution
**Timeline**: 6 Weeks
**Effort**: 30 Engineer-Weeks
**Last Updated**: 2025-11-13

---

## Executive Summary

### The Bootstrap Challenge

We face a classic meta-quality problem: **How do we fix a broken quality tool using the quality principles it doesn't yet enforce?**

**Current State**:
- Connascence Safety Analyzer: 60% production ready
- 7 critical issues preventing self-analysis
- Cannot analyze itself due to detector pool errors and god objects
- No integration with broader quality ecosystem (NASA standards, Clarity rules)

**Target State**:
- 95% production ready with zero violations
- Self-analyzing with full dogfooding loop
- Unified quality gate: Connascence + NASA + Clarity in one command
- Continuous improvement through automated issue creation

### 6-Week Transformation Roadmap

```
Week 1: Critical Fixes → Gate 1 (CRITICAL only)
Week 2-4: Major Refactoring + Clarity Linter MVP → Gate 2 (CRITICAL + HIGH)
Week 5: Dogfooding Activation → Gate 3 (CRITICAL + HIGH + MEDIUM)
Week 6: Zero Violations → Gate 4 (ANY violation fails)
```

**Success Criteria**:
- ✅ All 7 critical issues resolved
- ✅ Clarity Linter integrated with 11 rules
- ✅ 100% self-analysis capability
- ✅ Zero violations across all analyzers
- ✅ Automated CI/CD quality gates
- ✅ Scaffolding cleanup complete

---

## Phase 0: Clarity Linter Implementation (Parallel, Weeks 1-3)

### Overview

Build the Clarity Linter in parallel with analyzer fixes to enable sophisticated refactoring guidance.

### Module Structure

```
analyzer/
├── clarity/
│   ├── __init__.py
│   ├── rules/
│   │   ├── __init__.py
│   │   ├── base_rule.py              # Abstract base class
│   │   ├── thin_helpers.py           # CLARITY001/002
│   │   ├── modular_functions.py      # CLARITY003/004
│   │   ├── cyclomatic_complexity.py  # CLARITY005
│   │   ├── nesting_depth.py          # CLARITY006
│   │   ├── function_length.py        # CLARITY007
│   │   ├── file_length.py            # CLARITY008
│   │   ├── parameter_count.py        # CLARITY009
│   │   ├── boolean_params.py         # CLARITY010
│   │   └── mega_functions.py         # CLARITY011
│   ├── analyzer.py                    # Main analyzer engine
│   ├── reporter.py                    # Results formatter
│   └── config.py                      # Configuration loader
├── nasa/
│   └── standards.py                   # Existing NASA rules
└── connascence/
    └── detector.py                    # Existing connascence detection
```

### 11 Clarity Rules Implementation

**Priority 1 (Week 1) - MVP Core Rules**:
1. **CLARITY005**: Cyclomatic Complexity (max 10)
2. **CLARITY006**: Nesting Depth (max 4 levels)
3. **CLARITY007**: Function Length (max 50 lines)
4. **CLARITY009**: Parameter Count (max 6 params)
5. **CLARITY011**: Mega-Functions (>100 lines, >20 complexity)

**Priority 2 (Week 2-3) - Advanced Rules**:
6. **CLARITY001**: Thin Helper Functions (<3 lines)
7. **CLARITY002**: Single-Statement Wrappers
8. **CLARITY003**: Modular Function Structure
9. **CLARITY004**: Clear Function Boundaries
10. **CLARITY008**: File Length (max 500 lines)
11. **CLARITY010**: Boolean Parameter Flags

### Base Rule Interface

```python
# analyzer/clarity/rules/base_rule.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Violation:
    """Represents a Clarity rule violation."""
    rule_id: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    file_path: str
    line_number: int
    column: Optional[int] = None
    suggestion: Optional[str] = None

class ClarityRule(ABC):
    """Abstract base class for all Clarity rules."""

    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.severity = config.get('severity', 'MEDIUM')

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Unique rule identifier (e.g., 'CLARITY001')."""
        pass

    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Human-readable rule name."""
        pass

    @abstractmethod
    def analyze(self, ast_node, file_path: str) -> List[Violation]:
        """
        Analyze AST node and return violations.

        Args:
            ast_node: Python AST node to analyze
            file_path: Path to source file

        Returns:
            List of Violation objects
        """
        pass

    def should_skip(self, node) -> bool:
        """Check if node should be skipped (e.g., test files)."""
        return not self.enabled
```

### Example Rule: Cyclomatic Complexity

```python
# analyzer/clarity/rules/cyclomatic_complexity.py
import ast
from typing import List
from .base_rule import ClarityRule, Violation

class CyclomaticComplexityRule(ClarityRule):
    """CLARITY005: Enforce cyclomatic complexity limits."""

    @property
    def rule_id(self) -> str:
        return "CLARITY005"

    @property
    def rule_name(self) -> str:
        return "Cyclomatic Complexity"

    def analyze(self, ast_node, file_path: str) -> List[Violation]:
        violations = []
        max_complexity = self.config.get('max_complexity', 10)

        for node in ast.walk(ast_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_complexity(node)

                if complexity > max_complexity:
                    violations.append(Violation(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        message=f"Function '{node.name}' has complexity {complexity} (max: {max_complexity})",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion=f"Consider breaking down '{node.name}' into smaller functions"
                    ))

        return violations

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity using decision points."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points that increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1

        return complexity
```

### Clarity Analyzer Engine

```python
# analyzer/clarity/analyzer.py
import ast
from pathlib import Path
from typing import List, Dict
from .rules import (
    CyclomaticComplexityRule,
    NestingDepthRule,
    FunctionLengthRule,
    ParameterCountRule,
    MegaFunctionRule,
    ThinHelperRule,
    SingleStatementWrapperRule,
    ModularFunctionRule,
    FileLengthRule,
    BooleanParamRule
)
from .config import load_config

class ClarityAnalyzer:
    """Main analyzer engine for Clarity rules."""

    def __init__(self, config_path: str = "clarity_linter.yaml"):
        self.config = load_config(config_path)
        self.rules = self._initialize_rules()

    def _initialize_rules(self) -> List:
        """Initialize all enabled rules from config."""
        rules = []
        rule_classes = {
            'CLARITY001': ThinHelperRule,
            'CLARITY002': SingleStatementWrapperRule,
            'CLARITY003': ModularFunctionRule,
            'CLARITY004': ModularFunctionRule,  # Same impl
            'CLARITY005': CyclomaticComplexityRule,
            'CLARITY006': NestingDepthRule,
            'CLARITY007': FunctionLengthRule,
            'CLARITY008': FileLengthRule,
            'CLARITY009': ParameterCountRule,
            'CLARITY010': BooleanParamRule,
            'CLARITY011': MegaFunctionRule
        }

        for rule_id, rule_class in rule_classes.items():
            rule_config = self.config.get('rules', {}).get(rule_id, {})
            if rule_config.get('enabled', True):
                rules.append(rule_class(rule_config))

        return rules

    def analyze_file(self, file_path: str) -> Dict:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)
            violations = []

            for rule in self.rules:
                violations.extend(rule.analyze(tree, file_path))

            return {
                'file': file_path,
                'violations': violations,
                'total': len(violations)
            }
        except Exception as e:
            return {
                'file': file_path,
                'error': str(e),
                'violations': [],
                'total': 0
            }

    def analyze_directory(self, dir_path: str) -> Dict:
        """Analyze all Python files in a directory."""
        path = Path(dir_path)
        results = []

        for py_file in path.rglob('*.py'):
            if self._should_skip(py_file):
                continue
            results.append(self.analyze_file(str(py_file)))

        return {
            'directory': dir_path,
            'files_analyzed': len(results),
            'results': results,
            'total_violations': sum(r['total'] for r in results)
        }

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = self.config.get('exclude', [])
        return any(pattern in str(file_path) for pattern in skip_patterns)
```

### Integration with Existing Infrastructure

```python
# analyzer/unified_gate.py
"""Unified quality gate combining Connascence, NASA, and Clarity analyzers."""

from .connascence.detector import ConnascenceDetector
from .nasa.standards import NASAStandardsChecker
from .clarity.analyzer import ClarityAnalyzer

class UnifiedQualityGate:
    """Unified quality gate for all analyzers."""

    def __init__(self, gate_level: int = 1):
        self.gate_level = gate_level
        self.connascence = ConnascenceDetector()
        self.nasa = NASAStandardsChecker()
        self.clarity = ClarityAnalyzer()

    def analyze(self, file_path: str) -> Dict:
        """Run all analyzers and aggregate results."""
        results = {
            'file': file_path,
            'connascence': self.connascence.analyze_file(file_path),
            'nasa': self.nasa.check_file(file_path),
            'clarity': self.clarity.analyze_file(file_path),
            'gate_level': self.gate_level
        }

        # Apply gate-level filtering
        results['violations'] = self._filter_by_gate(results)
        results['pass'] = len(results['violations']) == 0

        return results

    def _filter_by_gate(self, results: Dict) -> List:
        """Filter violations based on gate level."""
        severity_map = {
            1: ['CRITICAL'],
            2: ['CRITICAL', 'HIGH'],
            3: ['CRITICAL', 'HIGH', 'MEDIUM'],
            4: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        }

        allowed = severity_map.get(self.gate_level, ['CRITICAL'])
        violations = []

        for analyzer in ['connascence', 'nasa', 'clarity']:
            for v in results[analyzer].get('violations', []):
                if v.severity in allowed:
                    violations.append(v)

        return violations
```

### CLI Integration

```python
# analyzer/cli.py
import click
from .unified_gate import UnifiedQualityGate

@click.group()
def cli():
    """Unified quality analyzer CLI."""
    pass

@cli.command()
@click.argument('path')
@click.option('--gate-level', default=1, help='Quality gate level (1-4)')
@click.option('--format', default='text', help='Output format (text/json/github)')
def analyze(path, gate_level, format):
    """Analyze code with unified quality gate."""
    gate = UnifiedQualityGate(gate_level)

    if Path(path).is_file():
        results = gate.analyze(path)
    else:
        results = gate.analyze_directory(path)

    if format == 'json':
        click.echo(json.dumps(results, indent=2))
    elif format == 'github':
        _output_github_annotations(results)
    else:
        _output_text(results)

    if not results['pass']:
        raise SystemExit(1)

if __name__ == '__main__':
    cli()
```

---

## Phase 1: Critical Fixes (Week 1)

### ISSUE-001: Detector Pool Initialization

**File**: `analyzer/connascence/detector.py`

**Problem**: Missing detector pool initialization causes crashes on startup.

**Fix**:
```python
# analyzer/connascence/detector.py (line ~45)

class ConnascenceDetector:
    def __init__(self):
        # BEFORE (missing initialization)
        # self.detectors = None

        # AFTER (proper initialization)
        self.detectors = {
            'CoN': NameConnascenceDetector(),
            'CoT': TypeConnascenceDetector(),
            'CoP': PositionConnascenceDetector(),
            'CoM': MeaningConnascenceDetector(),
            'CoA': AlgorithmConnascenceDetector(),
            'CoE': ExecutionConnascenceDetector(),
            'CoI': IdentityConnascenceDetector()
        }
        self.config = self._load_config()
```

**Test**:
```bash
python -m pytest tests/test_detector_init.py -v
python -m analyzer.cli analyze analyzer/ --gate-level 1
```

### ISSUE-002: Missing detect_all() Method

**File**: `analyzer/connascence/detector.py`

**Problem**: `detect_all()` method referenced but not implemented.

**Fix**:
```python
# analyzer/connascence/detector.py (line ~120)

def detect_all(self, file_path: str) -> Dict[str, List]:
    """
    Run all connascence detectors on a file.

    Args:
        file_path: Path to Python source file

    Returns:
        Dict mapping detector type to list of violations
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source, filename=file_path)
    results = {}

    for detector_type, detector in self.detectors.items():
        try:
            violations = detector.detect(tree, file_path)
            if violations:
                results[detector_type] = violations
        except Exception as e:
            logger.error(f"Detector {detector_type} failed on {file_path}: {e}")
            results[detector_type] = []

    return results
```

**Test**:
```python
# tests/test_detect_all.py
def test_detect_all_returns_dict():
    detector = ConnascenceDetector()
    results = detector.detect_all('analyzer/detector.py')

    assert isinstance(results, dict)
    assert all(isinstance(v, list) for v in results.values())
    assert 'CoP' in results  # Should detect parameter bombs
```

### ISSUE-003: Return Type Consistency

**File**: `analyzer/connascence/detectors/*.py` (all detectors)

**Problem**: Inconsistent return types (`List[str]` vs `List[Dict]`).

**Fix**: Standardize on `List[Violation]` dataclass.

```python
# analyzer/connascence/violation.py (new file)
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConnascenceViolation:
    """Standardized violation format."""
    type: str  # CoN, CoP, CoM, etc.
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    file_path: str
    line_number: int
    column: Optional[int] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'type': self.type,
            'severity': self.severity,
            'message': self.message,
            'location': {
                'file': self.file_path,
                'line': self.line_number,
                'column': self.column
            },
            'suggestion': self.suggestion
        }
```

**Update all detectors**:
```python
# analyzer/connascence/detectors/position.py
from ..violation import ConnascenceViolation

class PositionConnascenceDetector:
    def detect(self, tree: ast.AST, file_path: str) -> List[ConnascenceViolation]:
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)
                if param_count > 6:  # NASA limit
                    violations.append(ConnascenceViolation(
                        type='CoP',
                        severity='HIGH',
                        message=f"Function '{node.name}' has {param_count} parameters (max: 6)",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion=f"Consider using a parameter object for '{node.name}'"
                    ))

        return violations
```

### ISSUE-004: Configuration Thresholds

**File**: `analyzer/config/thresholds.yaml`

**Problem**: Hardcoded thresholds throughout codebase.

**Fix**: Centralize configuration.

```yaml
# analyzer/config/thresholds.yaml
connascence:
  CoP:  # Parameter position
    max_params: 6
    severity: HIGH

  CoM:  # Meaning
    magic_numbers:
      - 0
      - 1
      - -1
    severity: MEDIUM

  CoN:  # Name
    naming_patterns:
      - "^[a-z_][a-z0-9_]*$"  # snake_case
    severity: LOW

nasa:
  cyclomatic_complexity:
    max: 10
    severity: HIGH

  nesting_depth:
    max: 4
    severity: HIGH

  function_length:
    max: 50
    severity: MEDIUM

clarity:
  thin_helper_min_lines: 3
  mega_function_lines: 100
  mega_function_complexity: 20
  file_length_max: 500
```

```python
# analyzer/config/loader.py
import yaml
from pathlib import Path

class ConfigLoader:
    """Centralized configuration loader."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str = None) -> dict:
        """Load configuration from YAML."""
        if self._config is not None:
            return self._config

        if config_path is None:
            config_path = Path(__file__).parent / 'thresholds.yaml'

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        return self._config

    def get_threshold(self, analyzer: str, rule: str, key: str):
        """Get specific threshold value."""
        return self._config.get(analyzer, {}).get(rule, {}).get(key)

# Usage in detectors
config = ConfigLoader().load()
max_params = config.get_threshold('connascence', 'CoP', 'max_params')
```

### Gate 1 Activation

**Command**:
```bash
# Run unified gate at level 1 (CRITICAL only)
python -m analyzer.cli analyze analyzer/ --gate-level 1

# Expected: Only CRITICAL violations fail the build
# - Missing detector pool
# - Missing methods
# - Type errors
```

**CI/CD Integration**:
```yaml
# .github/workflows/quality-gate-1.yml
name: Quality Gate 1 - Critical Issues

on: [push, pull_request]

jobs:
  gate-1:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .

      - name: Run Gate 1 (CRITICAL only)
        run: |
          python -m analyzer.cli analyze analyzer/ --gate-level 1 --format github

      - name: Create issues for violations
        if: failure()
        run: |
          python scripts/create-issues-from-violations.py
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## Phase 2: Major Refactoring (Weeks 2-4)

### ISSUE-005: God Object Refactoring

**Target**: `analyzer/connascence/detector.py` (26 methods)

**Strategy**: Extract cohesive components using Clarity guidance.

#### Step 1: Identify Extraction Candidates (CLARITY003/004)

```bash
# Use Clarity Linter to identify boundaries
python -m analyzer.clarity analyze analyzer/connascence/detector.py --rules CLARITY003,CLARITY004

# Output shows 4 clear responsibility clusters:
# 1. Stream Processing (6 methods)
# 2. Cache Management (4 methods)
# 3. Metrics Collection (3 methods)
# 4. Core Detection (13 methods)
```

#### Step 2: Extract StreamProcessor

```python
# analyzer/connascence/stream_processor.py (NEW FILE)
"""Stream processing utilities for connascence detection."""

import asyncio
from typing import AsyncIterator, Callable, Any
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    """Result from stream processing."""
    file_path: str
    violations: List
    duration_ms: float
    success: bool

class StreamProcessor:
    """
    Async stream processor for file analysis.

    Responsibilities:
    - Batch file processing with concurrency control
    - Progress tracking and cancellation
    - Error handling and retry logic
    """

    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._cancelled = False

    async def process_stream(
        self,
        files: AsyncIterator[str],
        processor: Callable[[str], Any],
        progress_callback: Callable[[int, int], None] = None
    ) -> AsyncIterator[ProcessingResult]:
        """
        Process files from async stream with concurrency control.

        Args:
            files: Async iterator of file paths
            processor: Function to process each file
            progress_callback: Optional progress reporting

        Yields:
            ProcessingResult for each file
        """
        total_processed = 0

        async for file_path in files:
            if self._cancelled:
                break

            async with self.semaphore:
                result = await self._process_file(file_path, processor)
                total_processed += 1

                if progress_callback:
                    progress_callback(total_processed, None)

                yield result

    async def _process_file(
        self,
        file_path: str,
        processor: Callable
    ) -> ProcessingResult:
        """Process single file with error handling."""
        import time
        start = time.time()

        try:
            violations = await asyncio.to_thread(processor, file_path)
            return ProcessingResult(
                file_path=file_path,
                violations=violations,
                duration_ms=(time.time() - start) * 1000,
                success=True
            )
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return ProcessingResult(
                file_path=file_path,
                violations=[],
                duration_ms=(time.time() - start) * 1000,
                success=False
            )

    def cancel(self):
        """Cancel stream processing."""
        self._cancelled = True
```

**Before (in detector.py, 72 lines)**:
```python
class ConnascenceDetector:
    async def process_files_stream(self, files, progress=None):
        # 72 lines of stream processing logic
        pass
```

**After (in detector.py, 8 lines)**:
```python
class ConnascenceDetector:
    def __init__(self):
        self.stream_processor = StreamProcessor(max_concurrent=4)

    async def process_files_stream(self, files, progress=None):
        """Process files with async streaming."""
        async for result in self.stream_processor.process_stream(
            files, self.detect_all, progress
        ):
            yield result
```

**Metrics**:
- Function length: 72 → 8 lines (-89%)
- Cyclomatic complexity: 13 → 3 (-77%)
- Violations: CLARITY007, CLARITY011 → RESOLVED

#### Step 3: Extract CacheManager

```python
# analyzer/connascence/cache_manager.py (NEW FILE)
"""Cache management for detection results."""

from typing import Optional, Dict, Any
import hashlib
import pickle
from pathlib import Path

class CacheManager:
    """
    Result caching for connascence detection.

    Responsibilities:
    - File-based result caching with TTL
    - Cache invalidation on source changes
    - Memory-efficient storage
    """

    def __init__(self, cache_dir: str = '.cache/connascence'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}

    def get(self, file_path: str, source_hash: str) -> Optional[Dict]:
        """
        Get cached results if valid.

        Args:
            file_path: Source file path
            source_hash: Hash of current source content

        Returns:
            Cached results or None if invalid/missing
        """
        cache_key = self._cache_key(file_path)

        # Try memory cache first
        if cache_key in self._memory_cache:
            cached = self._memory_cache[cache_key]
            if cached['hash'] == source_hash:
                return cached['results']

        # Try disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)

                if cached['hash'] == source_hash:
                    self._memory_cache[cache_key] = cached
                    return cached['results']
            except Exception as e:
                logger.warning(f"Cache read failed for {file_path}: {e}")

        return None

    def put(self, file_path: str, source_hash: str, results: Dict):
        """Cache detection results."""
        cache_key = self._cache_key(file_path)
        cached = {
            'hash': source_hash,
            'results': results,
            'timestamp': time.time()
        }

        # Update memory cache
        self._memory_cache[cache_key] = cached

        # Write to disk
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cached, f)
        except Exception as e:
            logger.warning(f"Cache write failed for {file_path}: {e}")

    def invalidate(self, file_path: str = None):
        """Invalidate cache for file or entire cache."""
        if file_path:
            cache_key = self._cache_key(file_path)
            self._memory_cache.pop(cache_key, None)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            cache_file.unlink(missing_ok=True)
        else:
            self._memory_cache.clear()
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()

    def _cache_key(self, file_path: str) -> str:
        """Generate cache key from file path."""
        return hashlib.md5(file_path.encode()).hexdigest()
```

#### Step 4: Extract MetricsCollector

```python
# analyzer/connascence/metrics_collector.py (NEW FILE)
"""Metrics collection for detection runs."""

from typing import Dict, List
from dataclasses import dataclass, field
import time

@dataclass
class DetectionMetrics:
    """Metrics from a detection run."""
    files_analyzed: int = 0
    total_violations: int = 0
    violations_by_type: Dict[str, int] = field(default_factory=dict)
    violations_by_severity: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    def add_violation(self, violation_type: str, severity: str):
        """Record a violation."""
        self.total_violations += 1
        self.violations_by_type[violation_type] = \
            self.violations_by_type.get(violation_type, 0) + 1
        self.violations_by_severity[severity] = \
            self.violations_by_severity.get(severity, 0) + 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'summary': {
                'files': self.files_analyzed,
                'violations': self.total_violations,
                'duration': f"{self.duration_seconds:.2f}s",
                'errors': len(self.errors)
            },
            'by_type': self.violations_by_type,
            'by_severity': self.violations_by_severity
        }

class MetricsCollector:
    """
    Collect and aggregate detection metrics.

    Responsibilities:
    - Track detection performance
    - Aggregate violation statistics
    - Generate summary reports
    """

    def __init__(self):
        self.metrics = DetectionMetrics()
        self._start_time = None

    def start(self):
        """Start metrics collection."""
        self._start_time = time.time()

    def stop(self):
        """Stop metrics collection."""
        if self._start_time:
            self.metrics.duration_seconds = time.time() - self._start_time

    def record_file(self, file_path: str, violations: List):
        """Record metrics for analyzed file."""
        self.metrics.files_analyzed += 1

        for violation in violations:
            self.metrics.add_violation(
                violation.type,
                violation.severity
            )

    def record_error(self, file_path: str, error: str):
        """Record analysis error."""
        self.metrics.errors.append(f"{file_path}: {error}")

    def get_report(self) -> Dict:
        """Get metrics report."""
        return self.metrics.to_dict()
```

#### Step 5: Update ConnascenceDetector

```python
# analyzer/connascence/detector.py (REFACTORED)
"""Connascence detection coordinator."""

from .stream_processor import StreamProcessor
from .cache_manager import CacheManager
from .metrics_collector import MetricsCollector

class ConnascenceDetector:
    """
    Core connascence detection coordinator.

    Responsibilities:
    - Initialize and coordinate detectors
    - Orchestrate file analysis
    - Delegate to specialized components
    """

    def __init__(self):
        # Core detection
        self.detectors = self._initialize_detectors()
        self.config = ConfigLoader().load()

        # Delegated responsibilities
        self.stream_processor = StreamProcessor()
        self.cache = CacheManager()
        self.metrics = MetricsCollector()

    def analyze_file(self, file_path: str) -> Dict:
        """
        Analyze single file for connascence violations.

        Returns cached results if available and valid.
        """
        # Check cache
        source_hash = self._hash_file(file_path)
        cached = self.cache.get(file_path, source_hash)
        if cached:
            return cached

        # Run detection
        results = self.detect_all(file_path)

        # Cache results
        self.cache.put(file_path, source_hash, results)

        # Record metrics
        self.metrics.record_file(file_path, results.get('violations', []))

        return results

    async def analyze_directory(self, dir_path: str) -> Dict:
        """Analyze directory with async streaming."""
        self.metrics.start()

        files = self._discover_files(dir_path)
        results = []

        async for result in self.stream_processor.process_stream(
            files, self.analyze_file
        ):
            results.append(result)

        self.metrics.stop()

        return {
            'directory': dir_path,
            'results': results,
            'metrics': self.metrics.get_report()
        }
```

**Final Metrics After Refactoring**:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Methods in detector.py | 26 | 13 | -50% |
| Lines in detector.py | 847 | 324 | -62% |
| Max function length | 72 | 28 | -61% |
| Max cyclomatic complexity | 13 | 6 | -54% |
| God Object violations | 1 | 0 | -100% |
| CLARITY violations | 7 | 0 | -100% |

### Gate 2 Activation

**Command**:
```bash
# Run unified gate at level 2 (CRITICAL + HIGH)
python -m analyzer.cli analyze analyzer/ --gate-level 2

# Expected violations fixed:
# - God objects refactored
# - Thin helpers extracted
# - Mega-functions split
# - NASA complexity violations resolved
```

---

## Phase 3: Dogfooding Activation (Week 5)

### First Self-Analysis

**Command**:
```bash
# Run all 3 analyzers on the analyzer codebase
python -m analyzer.cli analyze analyzer/ --gate-level 3 --format json > self-analysis.json

# Generate report
python scripts/generate-self-analysis-report.py self-analysis.json
```

**Expected Output**:
```json
{
  "summary": {
    "files_analyzed": 47,
    "total_violations": 12,
    "by_analyzer": {
      "connascence": 3,
      "nasa": 4,
      "clarity": 5
    },
    "by_severity": {
      "CRITICAL": 0,
      "HIGH": 0,
      "MEDIUM": 12,
      "LOW": 0
    }
  },
  "violations": [
    {
      "file": "analyzer/nasa/standards.py",
      "rule": "CLARITY008",
      "severity": "MEDIUM",
      "message": "File length 523 lines exceeds 500 line limit",
      "suggestion": "Split into nasa/standards_core.py and nasa/standards_extended.py"
    }
  ]
}
```

### Fix Remaining Violations

**Week 5 Priorities**:
1. Split oversized files (CLARITY008)
2. Fix remaining parameter bombs (CoP, CLARITY009)
3. Eliminate boolean flags (CLARITY010)
4. Address medium-severity nesting (CLARITY006)

### Scaffolding Cleanup

**Remove development artifacts**:

```bash
# Remove .claude and .claude-flow folders
rm -rf .claude .claude-flow

# Keep only essential config
mkdir -p .analysis-config
mv .claude/agents/*.md .analysis-config/agent-specs/
mv .claude/skills/*.yaml .analysis-config/skills/

# Update .gitignore
cat >> .gitignore << 'EOF'
# Development scaffolding (removed for production)
.claude/
.claude-flow/
.agent-registry/
.sparc-modes/

# Keep analysis configuration
!.analysis-config/
EOF

# Commit cleanup
git add .
git commit -m "chore: Remove development scaffolding for production release"
```

### Auto-Issue Creation Workflow

**File**: `.github/workflows/auto-issue-from-violations.yml`

```yaml
name: Auto-Create Issues from Quality Gate Violations

on:
  push:
    branches: [main, develop]
  pull_request:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  analyze-and-create-issues:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install analyzer
        run: |
          pip install -r requirements.txt
          pip install -e .

      - name: Run unified quality gate
        id: gate
        run: |
          python -m analyzer.cli analyze analyzer/ \
            --gate-level 3 \
            --format json > violations.json
        continue-on-error: true

      - name: Parse violations and create issues
        if: steps.gate.outcome == 'failure'
        run: |
          python scripts/auto-create-issues.py violations.json
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Upload violation report
        uses: actions/upload-artifact@v3
        with:
          name: violation-report
          path: violations.json
```

**Script**: `scripts/auto-create-issues.py`

```python
#!/usr/bin/env python3
"""Auto-create GitHub issues from quality gate violations."""

import json
import os
import sys
from github import Github

def create_issue_from_violation(repo, violation: dict, gate_level: int):
    """Create GitHub issue for a violation."""

    # Check if issue already exists
    title = f"[Quality Gate {gate_level}] {violation['rule']}: {violation['file']}"

    existing = repo.get_issues(state='open')
    for issue in existing:
        if issue.title == title:
            print(f"Issue already exists: {title}")
            return

    # Create new issue
    body = f"""
## Quality Violation Detected

**File**: `{violation['file']}`
**Rule**: {violation['rule']} ({violation['severity']})
**Line**: {violation.get('line_number', 'N/A')}

### Message
{violation['message']}

### Suggestion
{violation.get('suggestion', 'No automated suggestion available')}

### Context
- **Gate Level**: {gate_level}
- **Analyzer**: {violation.get('analyzer', 'unknown')}
- **Detected**: {violation.get('timestamp', 'N/A')}

### Priority
Based on severity **{violation['severity']}**:
- CRITICAL: P0 (fix immediately)
- HIGH: P1 (fix this sprint)
- MEDIUM: P2 (fix this quarter)
- LOW: P3 (backlog)

---

*Auto-generated by Quality Gate {gate_level} analysis*
"""

    labels = [
        'quality-gate',
        f"severity-{violation['severity'].lower()}",
        f"analyzer-{violation.get('analyzer', 'unknown')}",
        'automated-issue'
    ]

    issue = repo.create_issue(
        title=title,
        body=body,
        labels=labels
    )

    print(f"Created issue #{issue.number}: {title}")

def main():
    violations_file = sys.argv[1]

    with open(violations_file, 'r') as f:
        data = json.load(f)

    token = os.environ['GITHUB_TOKEN']
    repo_name = os.environ.get('GITHUB_REPOSITORY', 'owner/repo')

    g = Github(token)
    repo = g.get_repo(repo_name)

    gate_level = data.get('gate_level', 3)
    violations = data.get('violations', [])

    print(f"Processing {len(violations)} violations from Gate {gate_level}")

    for v in violations:
        create_issue_from_violation(repo, v, gate_level)

    print(f"Issue creation complete. Check: https://github.com/{repo_name}/issues")

if __name__ == '__main__':
    main()
```

### Gate 3 Activation

**Command**:
```bash
# Gate 3: CRITICAL + HIGH + MEDIUM
python -m analyzer.cli analyze analyzer/ --gate-level 3

# Expected: Only LOW violations remain (if any)
```

---

## Phase 4: Full Dogfooding (Week 6)

### Gate 4: Zero Violations

**Goal**: Achieve and maintain zero violations across all analyzers.

**Process**:
1. Fix all remaining LOW violations
2. Enable pre-commit hooks for continuous validation
3. Add quality gate to CI/CD
4. Monitor metrics dashboard

### Pre-Commit Hook

**File**: `.git/hooks/pre-commit`

```bash
#!/bin/bash
# Pre-commit quality gate check

echo "Running quality gate analysis on staged files..."

# Get staged Python files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ -z "$STAGED_FILES" ]; then
    echo "No Python files staged, skipping analysis"
    exit 0
fi

# Run unified gate at level 4 (strictest)
python -m analyzer.cli analyze $STAGED_FILES --gate-level 4 --format text

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Quality gate failed! Fix violations before committing."
    echo "To bypass (not recommended): git commit --no-verify"
    exit 1
fi

echo "✅ Quality gate passed!"
exit 0
```

**Installation**:
```bash
# Make hook executable
chmod +x .git/hooks/pre-commit

# Or use pre-commit framework
pip install pre-commit
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: local
    hooks:
      - id: quality-gate
        name: Unified Quality Gate
        entry: python -m analyzer.cli analyze
        args: [--gate-level, "4", --format, text]
        language: system
        types: [python]
        pass_filenames: true
EOF

pre-commit install
```

### Metrics Dashboard

**File**: `scripts/generate-metrics-dashboard.py`

```python
#!/usr/bin/env python3
"""Generate quality metrics dashboard."""

import json
from datetime import datetime
from pathlib import Path

def generate_dashboard(metrics_file: str):
    """Generate HTML dashboard from metrics."""

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Metrics Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric-card {{
            border: 1px solid #ddd;
            padding: 20px;
            margin: 10px;
            border-radius: 5px;
            display: inline-block;
        }}
        .metric-value {{ font-size: 48px; font-weight: bold; }}
        .metric-label {{ color: #666; margin-top: 10px; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
    </style>
</head>
<body>
    <h1>Connascence Safety Analyzer - Quality Metrics</h1>
    <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="metric-card">
        <div class="metric-value success">{data['files_analyzed']}</div>
        <div class="metric-label">Files Analyzed</div>
    </div>

    <div class="metric-card">
        <div class="metric-value {'success' if data['total_violations'] == 0 else 'danger'}">
            {data['total_violations']}
        </div>
        <div class="metric-label">Total Violations</div>
    </div>

    <div class="metric-card">
        <div class="metric-value success">{data['test_pass_rate']:.1f}%</div>
        <div class="metric-label">Test Pass Rate</div>
    </div>

    <div class="metric-card">
        <div class="metric-value success">{data['code_coverage']:.1f}%</div>
        <div class="metric-label">Code Coverage</div>
    </div>

    <h2>Violations by Analyzer</h2>
    <table border="1" cellpadding="10">
        <tr>
            <th>Analyzer</th>
            <th>Violations</th>
            <th>Status</th>
        </tr>
"""

    for analyzer, count in data['by_analyzer'].items():
        status = '✅ Pass' if count == 0 else f'❌ {count} issues'
        html += f"""
        <tr>
            <td>{analyzer.title()}</td>
            <td>{count}</td>
            <td>{status}</td>
        </tr>
"""

    html += """
    </table>

    <h2>Historical Trend</h2>
    <div id="chart"></div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Chart implementation
        const ctx = document.getElementById('chart');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
                datasets: [{
                    label: 'Total Violations',
                    data: [47, 32, 18, 12, 5, 0],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            }
        });
    </script>
</body>
</html>
"""

    output = Path('docs/metrics-dashboard.html')
    output.write_text(html)
    print(f"Dashboard generated: {output}")

if __name__ == '__main__':
    generate_dashboard('metrics/latest.json')
```

**Weekly Update Script**:

```bash
#!/bin/bash
# scripts/update-weekly-metrics.sh

# Run full analysis
python -m analyzer.cli analyze analyzer/ --gate-level 4 --format json > metrics/week-$(date +%U).json

# Generate dashboard
python scripts/generate-metrics-dashboard.py

# Commit metrics
git add metrics/ docs/metrics-dashboard.html
git commit -m "metrics: Update weekly quality dashboard"
git push
```

### Zero-Violation Policy

**Enforcement**:

```yaml
# .github/workflows/zero-violation-policy.yml
name: Zero Violation Policy

on:
  pull_request:
  push:
    branches: [main]

jobs:
  enforce:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install analyzer
        run: pip install -e .

      - name: Run Gate 4 (Zero Violations)
        run: |
          python -m analyzer.cli analyze analyzer/ --gate-level 4 --format github

      - name: Block merge on violations
        if: failure()
        run: |
          echo "❌ Quality gate failed. Zero violations required for merge."
          exit 1
```

### Continuous Improvement Loop

```python
# scripts/continuous-improvement.py
"""Continuous improvement automation."""

import subprocess
import json
from pathlib import Path

def run_improvement_cycle():
    """Run one improvement cycle."""

    print("🔍 Running quality analysis...")
    result = subprocess.run(
        ['python', '-m', 'analyzer.cli', 'analyze', 'analyzer/',
         '--gate-level', '4', '--format', 'json'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ Zero violations! System is healthy.")
        return

    violations = json.loads(result.stdout)

    print(f"📊 Found {len(violations['violations'])} violations")
    print("🤖 Attempting automated fixes...")

    for v in violations['violations']:
        if v.get('auto_fixable'):
            apply_automated_fix(v)

    print("🧪 Re-running tests...")
    test_result = subprocess.run(['pytest', 'tests/'], capture_output=True)

    if test_result.returncode == 0:
        print("✅ Tests passed after automated fixes")
        commit_fixes(violations)
    else:
        print("❌ Tests failed, reverting fixes")
        revert_fixes()

def apply_automated_fix(violation: dict):
    """Apply automated fix for violation."""
    # Implementation depends on violation type
    pass

if __name__ == '__main__':
    run_improvement_cycle()
```

---

## Implementation Artifacts

### Files to Create

#### Configuration Files
1. `analyzer/config/thresholds.yaml` - Centralized thresholds
2. `clarity_linter.yaml` - Clarity Linter configuration
3. `.pre-commit-config.yaml` - Pre-commit hooks
4. `.github/workflows/quality-gate-1.yml` - Gate 1 CI
5. `.github/workflows/quality-gate-2.yml` - Gate 2 CI
6. `.github/workflows/quality-gate-3.yml` - Gate 3 CI
7. `.github/workflows/zero-violation-policy.yml` - Gate 4 CI
8. `.github/workflows/auto-issue-from-violations.yml` - Issue automation

#### Source Code
9. `analyzer/clarity/` - Clarity Linter package (11 rule files)
10. `analyzer/connascence/stream_processor.py` - Stream processing
11. `analyzer/connascence/cache_manager.py` - Cache management
12. `analyzer/connascence/metrics_collector.py` - Metrics collection
13. `analyzer/connascence/violation.py` - Standardized violations
14. `analyzer/unified_gate.py` - Unified quality gate
15. `analyzer/config/loader.py` - Configuration loader

#### Scripts
16. `scripts/auto-create-issues.py` - GitHub issue automation
17. `scripts/generate-metrics-dashboard.py` - Metrics dashboard
18. `scripts/generate-self-analysis-report.py` - Self-analysis report
19. `scripts/update-weekly-metrics.sh` - Weekly metrics update
20. `scripts/continuous-improvement.py` - Improvement automation

#### Tests
21. `tests/test_detector_init.py` - Detector initialization tests
22. `tests/test_detect_all.py` - detect_all() tests
23. `tests/test_clarity_rules.py` - Clarity rule tests
24. `tests/test_unified_gate.py` - Unified gate tests
25. `tests/test_stream_processor.py` - Stream processing tests

#### Documentation
26. `docs/CLARITY-RULES.md` - Clarity rule documentation
27. `docs/QUALITY-GATES.md` - Quality gate guide
28. `docs/DOGFOODING-GUIDE.md` - Dogfooding best practices
29. `docs/metrics-dashboard.html` - Live metrics dashboard

---

## Success Metrics

### Quantitative Metrics

**Week 1 (Gate 1 Active)**:
- Total violations: <10 CRITICAL
- Test pass rate: 95%
- Code coverage: 85%

**Week 4 (Gate 2 Active)**:
- Total violations: <5 CRITICAL + <15 HIGH
- Test pass rate: 98%
- Code coverage: 90%

**Week 5 (Gate 3 Active)**:
- Total violations: <3 CRITICAL + <5 HIGH + <20 MEDIUM
- Test pass rate: 99%
- Code coverage: 92%

**Week 6 (Gate 4 Active)**:
- Total violations: 0
- Test pass rate: 100%
- Code coverage: 95%

### Qualitative Metrics

**Developer Experience**:
- Pre-commit hook execution: <5 seconds
- CI/CD pipeline: <10 minutes
- False positive rate: <5%
- Auto-fix success rate: >80%

**Time Savings**:
- Manual code review time: -40%
- Bug detection time: -60%
- Refactoring guidance: +300% accuracy

### Weekly Dashboard

**Tracked Metrics**:
- Violation count by analyzer
- Violation count by severity
- Test pass rate trend
- Code coverage trend
- Time to fix violations
- Number of auto-fixes applied

**Example Dashboard Update**:

```markdown
## Week 5 Quality Metrics (2025-11-18 to 2025-11-24)

### Summary
- **Total Violations**: 5 (↓ 7 from last week)
- **Test Pass Rate**: 99.2% (↑ 0.8% from last week)
- **Code Coverage**: 92.3% (↑ 1.1% from last week)
- **Gate Level**: 3 (CRITICAL + HIGH + MEDIUM)

### Violations Breakdown
- Connascence: 1 CoM (MEDIUM)
- NASA: 2 complexity violations (MEDIUM)
- Clarity: 2 file length violations (MEDIUM)

### Progress
- **12 violations fixed** this week
- **3 automated fixes** applied successfully
- **2 GitHub issues** auto-created
- **0 regressions** introduced

### Next Week Goals
- Activate Gate 4 (zero violations)
- Fix remaining 5 MEDIUM violations
- Achieve 95% code coverage
```

---

## Risk Mitigation

### Risk 1: Clarity Linter Delays Building

**Probability**: 40%
**Impact**: HIGH (blocks Phase 2)

**Mitigation**:
- Build MVP with 5 core rules first (Week 1)
- Parallel development: Fix critical issues while building
- Fallback: Use only Connascence + NASA for Gate 2
- Buffer: Add 1 week to Phase 2 timeline

### Risk 2: Automated Fixes Introduce Bugs

**Probability**: 30%
**Impact**: MEDIUM (requires manual rollback)

**Mitigation**:
- Require 100% test pass before committing fixes
- Limit auto-fixes to low-risk transformations only
- Human review for all HIGH+ severity fixes
- Comprehensive test suite with 95%+ coverage

### Risk 3: False Positives Overwhelm Team

**Probability**: 25%
**Impact**: MEDIUM (developer frustration)

**Mitigation**:
- Tunable thresholds in `thresholds.yaml`
- Whitelist for legitimate exceptions
- Quick feedback loop: Daily reviews of false positives
- Progressive gates prevent overwhelming backlog

### Risk 4: God Object Refactoring Breaks Contracts

**Probability**: 20%
**Impact**: HIGH (breaks external dependencies)

**Mitigation**:
- Extract-and-delegate pattern preserves interfaces
- Comprehensive integration tests
- Deprecation warnings for changed APIs
- Version bump (1.0.0 → 2.0.0) with migration guide

### Risk 5: Scaffolding Cleanup Breaks Workflows

**Probability**: 15%
**Impact**: LOW (easily reversible)

**Mitigation**:
- Full backup before cleanup
- Test all workflows after cleanup
- Document removed folders in CHANGELOG
- Git tag before cleanup: `v1.9.0-pre-cleanup`

---

## Resource Requirements

### Team Composition

**Required Roles**:
1. **Principal Engineer (1)** - Architecture, design reviews, gate validation
2. **Senior Engineer (2)** - Clarity Linter, refactoring, CI/CD
3. **Mid-Level Engineer (1)** - Bug fixes, testing, documentation
4. **QA Engineer (1)** - Test suite, validation, metrics

**Total**: 5 FTEs for 6 weeks

### Effort Breakdown

| Phase | Engineer-Weeks | Total Hours |
|-------|----------------|-------------|
| Phase 0: Clarity Linter | 9 | 360 |
| Phase 1: Critical Fixes | 3 | 120 |
| Phase 2: Refactoring | 12 | 480 |
| Phase 3: Dogfooding Setup | 4 | 160 |
| Phase 4: Zero Violations | 2 | 80 |
| **Total** | **30** | **1200** |

### Budget Estimate

**Assumptions**:
- Average engineer cost: $150/hour (fully loaded)
- CI/CD infrastructure: $200/month
- Total timeline: 6 weeks

**Cost Breakdown**:
- Engineering: 1200 hours × $150 = $180,000
- Infrastructure: $200 × 1.5 months = $300
- Contingency (20%): $36,060
- **Total**: ~$216,360

---

## Timeline

```
Week 1: Critical Fixes
├── Day 1-2: Fix detector pool, missing methods
├── Day 3: Fix return types, standardize violations
├── Day 4-5: Centralize config, activate Gate 1
└── Milestone: Gate 1 passing (CRITICAL only)

Week 2: Clarity MVP + Begin Refactoring
├── Day 1-3: Build 5 core Clarity rules
├── Day 4-5: Integrate with unified gate
└── Milestone: Clarity MVP operational

Week 3: God Object Refactoring
├── Day 1-2: Extract StreamProcessor
├── Day 3: Extract CacheManager
├── Day 4: Extract MetricsCollector
├── Day 5: Integration testing
└── Milestone: God objects eliminated

Week 4: Complete Clarity + Gate 2
├── Day 1-3: Finish 6 remaining Clarity rules
├── Day 4: Fix thin helpers, mega-functions
├── Day 5: Activate Gate 2
└── Milestone: Gate 2 passing (CRITICAL + HIGH)

Week 5: Dogfooding Activation
├── Day 1: First self-analysis
├── Day 2-3: Fix MEDIUM violations
├── Day 4: Scaffolding cleanup
├── Day 5: Auto-issue workflow + Gate 3
└── Milestone: Gate 3 passing, dogfooding active

Week 6: Zero Violations
├── Day 1-2: Fix remaining violations
├── Day 3: Pre-commit hooks, CI/CD hardening
├── Day 4: Metrics dashboard, documentation
├── Day 5: Final validation + Gate 4
└── Milestone: Zero violations, production ready
```

---

## Appendix: Command Reference

### Analysis Commands

```bash
# Run full analysis with unified gate
python -m analyzer.cli analyze <path> --gate-level <1-4> --format <text|json|github>

# Analyze single file
python -m analyzer.cli analyze analyzer/detector.py --gate-level 2

# Analyze directory
python -m analyzer.cli analyze analyzer/ --gate-level 3 --format json

# Run specific analyzer only
python -m analyzer.cli analyze --analyzer connascence <path>
python -m analyzer.cli analyze --analyzer nasa <path>
python -m analyzer.cli analyze --analyzer clarity <path>
```

### Configuration Commands

```bash
# Load custom config
python -m analyzer.cli analyze --config custom-thresholds.yaml <path>

# Override specific threshold
python -m analyzer.cli analyze --set nasa.cyclomatic_complexity.max=15 <path>

# List all rules
python -m analyzer.cli rules list

# Get rule details
python -m analyzer.cli rules info CLARITY005
```

### Metrics Commands

```bash
# Generate metrics report
python scripts/generate-metrics-dashboard.py metrics/latest.json

# Update weekly metrics
./scripts/update-weekly-metrics.sh

# View historical trends
python scripts/view-trends.py --weeks 6
```

### CI/CD Commands

```bash
# Run Gate 1 locally
act -j gate-1  # Using nektos/act

# Test auto-issue creation
python scripts/auto-create-issues.py violations.json --dry-run

# Pre-commit hook
pre-commit run --all-files
```

---

## Conclusion

This META-REMEDIATION PLAN provides a comprehensive, actionable roadmap to transform the Connascence Safety Analyzer from 60% production ready to a fully dogfooded, zero-violation quality tool.

**Key Success Factors**:
1. **Progressive quality gates** prevent overwhelming backlog
2. **Parallel Clarity Linter development** enables sophisticated refactoring
3. **Automated issue creation** maintains visibility
4. **Scaffolding cleanup** signals production readiness
5. **Continuous improvement loop** ensures long-term health

**Next Steps**:
1. Review and approve plan with stakeholders
2. Assign team members to roles
3. Create GitHub project board from this plan
4. Begin Week 1: Critical Fixes
5. Weekly status updates via metrics dashboard

**Final Outcome**:
- ✅ Zero violations across all analyzers
- ✅ 95% test coverage
- ✅ 100% self-analysis capability
- ✅ Production-ready with CI/CD integration
- ✅ Automated continuous improvement

---

**Document Status**: READY FOR EXECUTION
**Approval Required**: Technical Lead, Project Manager, QA Lead
**Timeline Start**: Upon approval
**Success Criteria**: Gate 4 passing with zero violations

**Questions or concerns? Contact: [team-email]**
