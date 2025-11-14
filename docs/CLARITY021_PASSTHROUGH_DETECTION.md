# CLARITY021: Pass-Through Function Detection

## Overview

CLARITY021 detects functions that serve as simple pass-throughs to other functions without adding meaningful value. These functions increase cognitive load and indirection without providing benefits like error handling, logging, or data transformation.

**Rule ID**: CLARITY021
**Severity**: INFO
**Category**: Code Clarity / Cognitive Load

## Rule Description

A pass-through function is defined as:
- Contains exactly one statement (excluding docstrings)
- That statement is a `return` of a function call
- Arguments are passed to the called function unchanged or in the same order
- No additional logic, validation, or transformation is performed

## Rationale

Pass-through functions:
- **Increase cognitive load**: Developers must trace through unnecessary layers
- **Add maintenance overhead**: Changes require updates in multiple places
- **Obscure actual functionality**: Hide the real implementation
- **Violate KISS principle**: Unnecessary complexity without benefit

### When Pass-Through Functions Are Valid

The detector **excludes** the following valid delegation patterns:

1. **Decorators**: Functions with `@cached`, `@retry`, `@validate`, `@log`, etc.
2. **Type conversions**: `int()`, `str()`, `list()`, etc.
3. **Error handling**: Functions that add try/except blocks
4. **Logging**: Functions that add logging (when not using decorators)
5. **Validation**: Functions that check preconditions
6. **Deprecated functions**: Marked with `@deprecated` to maintain backward compatibility

## Examples

### ❌ Violations (Pass-Through Detected)

```python
# Simple pass-through
def wrapper(x, y):
    return other_func(x, y)
# VIOLATION: Function 'wrapper' is pass-through to 'other_func' with no added value

# Pass-through with docstring
def get_user(user_id):
    """Get user by ID."""
    return api.get_user(user_id)
# VIOLATION: Docstring doesn't add executable logic

# Method call pass-through
def process_data(data):
    return processor.process(data)
# VIOLATION: No added value

# Pass-through with varargs
def wrapper(*args, **kwargs):
    return other_func(*args, **kwargs)
# VIOLATION: Pure argument forwarding
```

### ✅ Valid Delegation Patterns (Not Violations)

```python
# Type conversion
def to_int(value):
    return int(value)
# VALID: Type conversion is a transformation

# With decorator
@cached
def get_user(user_id):
    return api.get_user(user_id)
# VALID: Caching adds value

# Error handling
def safe_divide(a, b):
    try:
        return divide(a, b)
    except ZeroDivisionError:
        return None
# VALID: Adds error handling

# Validation
def get_user(user_id):
    if not user_id:
        raise ValueError("user_id required")
    return api.get_user(user_id)
# VALID: Adds validation

# Logging (manual)
def get_user(user_id):
    logger.info(f"Fetching user {user_id}")
    return api.get_user(user_id)
# VALID: Adds logging

# Argument transformation
def wrapper(x, y):
    return other_func(y, x)
# VALID: Reorders arguments

# Deprecated function
@deprecated
def old_function(x):
    return new_function(x)
# VALID: Maintains backward compatibility
```

## Detection Algorithm

```python
1. Parse AST and find all function definitions
2. Check if function has valid delegation decorators
   - If yes, SKIP (valid delegation)
3. Get executable body (excluding docstrings)
4. If body has != 1 statement, SKIP (not pass-through)
5. If statement is not a return, SKIP
6. If return value is not a function call, SKIP
7. Extract function parameters and call arguments
8. If arguments are not passed through unchanged, SKIP
9. If call is a type conversion, SKIP (valid)
10. FLAG as VIOLATION
```

## Violation Output

```python
{
    "rule_id": "CLARITY021",
    "rule_name": "Pass-Through Function Detection",
    "severity": "INFO",
    "line": 42,
    "column": 0,
    "function_name": "wrapper",
    "target_function": "other_func",
    "message": "Function 'wrapper' is pass-through to 'other_func' with no added value",
    "suggested_fix": "Remove function 'wrapper' and call 'other_func' directly\nReplace all calls to 'wrapper()' with 'other_func()'",
    "metadata": {
        "is_async": false,
        "argument_count": 2,
        "has_transformations": false,
        "passthrough_type": "direct_passthrough"
    }
}
```

## Integration

### Programmatic Usage

```python
import ast
from analyzer.clarity_linter.detectors.clarity021_passthrough import (
    CLARITY021Detector,
    detect_passthrough_functions
)

# Using detector class
detector = CLARITY021Detector()
with open('mycode.py') as f:
    code = f.read()
tree = ast.parse(code)
violations = detector.detect(tree, code.split('\n'))

# Using convenience function
violations = detect_passthrough_functions(tree, code.split('\n'))

# Process violations
for v in violations:
    print(f"{v['line']}: {v['message']}")
    print(f"Fix: {v['suggested_fix']}")
```

### Command-Line Usage

```bash
# Run detector on file
python -m analyzer.clarity_linter.detectors.clarity021_passthrough mycode.py

# Run with other detectors
python -m analyzer.clarity_linter mycode.py
```

## Configuration

The detector has no configurable thresholds as pass-through detection is binary.

### Exclusion Patterns

To exclude specific functions, use valid delegation decorators:

```python
# Add decorator to exclude from detection
@passthrough_ok  # Custom decorator
def wrapper(x):
    return process(x)
```

Update `VALID_DECORATORS` in the detector if you use custom decorators:

```python
CLARITY021Detector.VALID_DECORATORS.add('passthrough_ok')
CLARITY021Detector.VALID_DECORATORS.add('wrapper')
```

## Testing

Run tests with pytest:

```bash
# Run all CLARITY021 tests
pytest tests/clarity_linter/test_clarity021.py -v

# Run specific test
pytest tests/clarity_linter/test_clarity021.py::TestCLARITY021Detector::test_simple_passthrough_detected -v

# Run with coverage
pytest tests/clarity_linter/test_clarity021.py --cov=analyzer.clarity_linter.detectors.clarity021_passthrough
```

Example test runner:

```bash
python tests/run_clarity021_tests.py
```

## Implementation Details

### File Structure

```
analyzer/
  clarity_linter/
    detectors/
      clarity021_passthrough.py    # Main detector implementation
tests/
  clarity_linter/
    test_clarity021.py             # Comprehensive test suite
  run_clarity021_tests.py          # Example test runner
docs/
  CLARITY021_PASSTHROUGH_DETECTION.md  # This document
```

### Key Components

1. **CLARITY021Detector**: Main detector class
   - `detect()`: Analyze AST for violations
   - `_analyze_function()`: Analyze single function
   - `_has_valid_decorator()`: Check for valid delegation decorators
   - `_analyze_call()`: Determine if call is pass-through
   - `_is_valid_delegation()`: Check for valid patterns

2. **detect_passthrough_functions()**: Convenience function

### Excluded Decorators

```python
VALID_DECORATORS = {
    'cached', 'cache', 'lru_cache', 'memoize',
    'retry', 'retries', 'backoff',
    'validate', 'validator', 'validates',
    'log', 'logged', 'logging',
    'monitor', 'monitored', 'trace', 'traced',
    'timed', 'profile', 'measure',
    'rate_limit', 'throttle',
    'deprecated', 'deprecation'
}
```

### Excluded Type Conversions

```python
VALID_TYPE_CONVERSIONS = {
    'int', 'float', 'str', 'bool',
    'list', 'dict', 'set', 'tuple',
    'bytes', 'bytearray', 'complex', 'frozenset'
}
```

## Known Limitations

1. **Class methods**: Detection of `self` parameter in class methods may need refinement
2. **Async context managers**: May not detect all async patterns
3. **Property decorators**: `@property` functions are not analyzed (by design)
4. **Lambda functions**: Not analyzed (only `FunctionDef` and `AsyncFunctionDef`)

## Future Enhancements

1. **Configurable decorator list**: Allow users to specify custom valid decorators
2. **Simple transformation detection**: Detect minimal transformations (e.g., `x.strip()`)
3. **Call graph analysis**: Detect pass-through chains (A→B→C)
4. **Auto-fix capability**: Automatically refactor pass-through functions
5. **IDE integration**: Real-time detection in editors

## Related Rules

- **CLARITY011**: Mega-Function Detection (functions > 60 LOC)
- **CLARITY022**: Thin Helper Detection (similar concept, different pattern)
- **CLARITY023**: Excessive Indirection (detects call chains)

## References

- **NASA JPL Coding Standard**: Avoid unnecessary function layers
- **Clean Code** by Robert C. Martin: Functions should do one thing well
- **Code Complete**: Minimize function call depth
- **Cognitive Load Theory**: Reduce mental burden on developers

## Contributing

To improve CLARITY021:

1. Add test cases to `tests/clarity_linter/test_clarity021.py`
2. Update exclusion patterns for new valid delegation patterns
3. Improve detection algorithm for edge cases
4. Add support for additional programming constructs

## License

This detector is part of the Clarity Linter project.

---

**Last Updated**: 2025-01-13
**Version**: 1.0.0
**Maintainer**: Code Quality Team
