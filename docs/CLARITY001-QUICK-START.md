# CLARITY001 Quick Start Guide

## What is CLARITY001?

CLARITY001 detects **thin helper functions** - functions that add unnecessary indirection without semantic value. These functions increase cognitive load and should be inlined.

## Quick Example

### ❌ BAD (Thin Helper)
```python
def get_status():
    return 200

def main():
    status = get_status()  # Useless indirection
```

### ✅ GOOD (Inlined)
```python
def main():
    status = 200  # Clear and direct
```

## Detection Criteria

A function is flagged if ALL of these are true:
- ✅ Has <20 lines of code
- ✅ Called from exactly ONE location
- ✅ Adds NO semantic value (see below)

## Semantic Value (Won't Be Flagged)

Functions with these qualities have semantic value:
- 🏷️ **Decorators**: `@property`, `@cached_property`, etc.
- 📝 **Meaningful names**: `validate_email()`, `parse_json()`, `calculate_total()`
- 📚 **Good docs**: Multi-line docstrings (>2 lines)
- 🔀 **Complex logic**: Multiple if/else, multiple returns, loops

## Installation

```bash
# No installation needed - pure Python
# Just ensure you're in the project root
cd /path/to/project
```

## Usage

### 1. Python API (Programmatic)

```python
from analyzer.clarity_linter import detect_thin_helpers

violations = detect_thin_helpers('myfile.py')
for v in violations:
    print(f"{v.rule_id} at line {v.line_number}: {v.message}")
    print(f"Fix: {v.suggested_fix}")
```

### 2. Command Line (Interactive)

```bash
# Analyze single file
python scripts/run_clarity001.py myfile.py

# Analyze directory
python scripts/run_clarity001.py src/

# JSON output
python scripts/run_clarity001.py src/ --json

# Quiet mode (violations only)
python scripts/run_clarity001.py src/ --quiet
```

### 3. Pre-commit Hook (Automated)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: clarity-linter
        name: CLARITY001 Thin Helper Detection
        entry: python scripts/run_clarity001.py
        language: system
        types: [python]
```

## Example Output

### Text Format
```
examples/clarity001_example.py:10: CLARITY001 [WARNING]
  Function: get_status_code (1 LOC)
  Message: Thin helper function 'get_status_code' (1 LOC) called from single location
  Suggested fix: Inline function into caller at line 73
```

### JSON Format
```json
{
  "total_files": 1,
  "total_violations": 3,
  "violations": [
    {
      "file": "examples/clarity001_example.py",
      "line": 10,
      "rule_id": "CLARITY001",
      "severity": "WARNING",
      "function": "get_status_code",
      "loc": 1,
      "call_site_line": 73,
      "message": "Thin helper function 'get_status_code' (1 LOC) called from single location",
      "suggested_fix": "Inline function into caller at line 73"
    }
  ]
}
```

## Common Examples

### Will Be Flagged ❌

```python
# Useless wrapper
def get_value():
    return 42

# Trivial calculation
def add(a, b):
    return a + b

# Single-use constant
def get_timeout():
    return 30
```

### Won't Be Flagged ✅

```python
# Multiple call sites
def calculate_total(items):
    return sum(item.price for item in items)

# Used in func1() and func2()

# Semantic validation
def validate_email(email):
    if '@' not in email:
        return False
    if '.' not in email.split('@')[1]:
        return False
    return True

# Has decorator
@property
def full_name(self):
    return f"{self.first_name} {self.last_name}"

# Complex logic
def determine_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    else:
        return 'F'
```

## Integration with CI/CD

### GitHub Actions

```yaml
name: Code Clarity Check
on: [pull_request]

jobs:
  clarity-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run CLARITY001
        run: python scripts/run_clarity001.py src/
```

### GitLab CI

```yaml
clarity-lint:
  script:
    - python scripts/run_clarity001.py src/
  only:
    - merge_requests
```

## Configuration

### Adjust Thresholds (Advanced)

Edit `analyzer/clarity_linter/detectors/clarity001_thin_helper.py`:

```python
class ThinHelperDetector:
    LOC_THRESHOLD = 20  # Change to 15 for stricter detection
    SEMANTIC_KEYWORDS = {
        'validate', 'parse', ...
        'your_custom_keyword'  # Add your keywords
    }
```

## Suppressing False Positives

If you need to suppress a violation:

```python
# Option 1: Add semantic keyword to function name
def validate_status():  # 'validate' = semantic value
    return 200

# Option 2: Add docstring
def get_status():
    """
    Get the HTTP status code.
    This is a placeholder for future logic.
    """
    return 200

# Option 3: Add complexity
def get_status():
    if condition:
        return 200
    return 500
```

## Testing

Run the test suite:

```bash
# All tests
python -m pytest tests/clarity_linter/ -v

# Specific test
python -m pytest tests/clarity_linter/test_clarity001.py::TestThinHelperDetection -v

# With coverage
python -m pytest tests/clarity_linter/ --cov=analyzer.clarity_linter
```

## Performance

- **Speed**: <50ms per file on average
- **Memory**: Minimal (AST-based, no caching)
- **Scalability**: Handles large files (1000+ LOC) efficiently

## Troubleshooting

### Import Error

```
Error: analyzer.clarity_linter not found
```

**Fix**: Run from project root or adjust PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
python scripts/run_clarity001.py myfile.py
```

### No Violations Detected

If you expect violations but see none:
1. Check LOC threshold (default: 20)
2. Verify single call site (use `--json` to see call sites)
3. Check for semantic keywords in function names
4. Verify function has no docstring

## Help & Support

- **Documentation**: `analyzer/clarity_linter/README.md`
- **Tests**: `tests/clarity_linter/test_clarity001.py`
- **Examples**: `examples/clarity001_example.py`
- **Implementation**: `analyzer/clarity_linter/detectors/clarity001_thin_helper.py`

## Quick Commands Cheat Sheet

```bash
# Analyze file
python scripts/run_clarity001.py myfile.py

# Analyze directory
python scripts/run_clarity001.py src/

# JSON output
python scripts/run_clarity001.py src/ --json

# Quiet mode
python scripts/run_clarity001.py src/ -q

# Run tests
python -m pytest tests/clarity_linter/test_clarity001.py -v

# Python API
python -c "from analyzer.clarity_linter import detect_thin_helpers; print(detect_thin_helpers('file.py'))"
```

## What's Next?

After fixing CLARITY001 violations, consider:
- **CLARITY002**: Excessive Call Chain Depth (coming soon)
- **CLARITY003**: Poor Naming Patterns (coming soon)
- **CLARITY004**: Comment Issues (coming soon)

---

**Remember**: The goal is code clarity. If a "thin helper" improves readability for your team, keep it! This tool provides guidance, not absolute rules.
