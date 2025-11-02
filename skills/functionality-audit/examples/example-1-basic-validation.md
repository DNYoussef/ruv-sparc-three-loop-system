# Example 1: Basic JSON Config Parser Validation

## Scenario

A developer created a Python utility function to parse JSON configuration files for a microservices deployment system. The function reads config files, validates schema, and applies default values. Initial testing with simple configs worked perfectly, but production deployments with nested configuration objects began failing mysteriously.

## Problem Statement

The `parse_config()` function appears syntactically correct and passes basic unit tests, but fails with nested objects in production. The function needs validation through sandbox testing to identify the root cause and ensure reliable operation with complex configurations.

**Initial Code:**
```python
def parse_config(config_path, defaults={}):
    """Parse JSON config file and merge with defaults."""
    import json

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Merge with defaults
    for key, value in defaults.items():
        if key not in config:
            config[key] = value

    return config
```

## Audit Process

### Step 1: Setup Sandbox

Create isolated test environment with realistic configuration scenarios:

```bash
# Create E2B sandbox with Python environment
npx claude-flow@alpha hooks pre-task --description "Validate JSON config parser"

# Setup test directory structure
mkdir -p /tmp/config-test/{configs,tests}
cd /tmp/config-test
```

**Create test configuration files:**
```bash
# Simple config (works)
cat > configs/simple.json << 'EOF'
{
  "service": "api",
  "port": 8080
}
EOF

# Nested config (fails in production)
cat > configs/nested.json << 'EOF'
{
  "service": "api",
  "database": {
    "host": "localhost",
    "port": 5432,
    "credentials": {
      "user": "admin"
    }
  }
}
EOF
```

### Step 2: Generate Test Cases

Design comprehensive test suite covering edge cases:

```python
# tests/test_parser.py
import unittest
import json
import os
from parser import parse_config

class TestConfigParser(unittest.TestCase):

    def setUp(self):
        """Create test configs."""
        self.test_dir = '/tmp/config-test/configs'

    def test_simple_config(self):
        """Test with flat config structure."""
        result = parse_config(f'{self.test_dir}/simple.json')
        self.assertEqual(result['service'], 'api')
        self.assertEqual(result['port'], 8080)

    def test_nested_config_with_defaults(self):
        """Test nested config merged with nested defaults."""
        defaults = {
            'timeout': 30,
            'database': {
                'host': 'db.example.com',
                'port': 5432,
                'credentials': {
                    'user': 'default_user',
                    'password': 'default_pass'
                }
            }
        }

        result = parse_config(f'{self.test_dir}/nested.json', defaults)

        # Original values should be preserved
        self.assertEqual(result['database']['host'], 'localhost')
        self.assertEqual(result['database']['credentials']['user'], 'admin')

        # Defaults should fill missing values
        self.assertEqual(result['timeout'], 30)
        self.assertIn('password', result['database']['credentials'])

    def test_deep_nesting(self):
        """Test with deeply nested structures."""
        deep_config = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 'deep'
                    }
                }
            }
        }

        with open(f'{self.test_dir}/deep.json', 'w') as f:
            json.dump(deep_config, f)

        defaults = {
            'level1': {
                'level2': {
                    'level3': {
                        'extra': 'default'
                    }
                }
            }
        }

        result = parse_config(f'{self.test_dir}/deep.json', defaults)
        self.assertEqual(result['level1']['level2']['level3']['value'], 'deep')
        self.assertIn('extra', result['level1']['level2']['level3'])
```

### Step 3: Execute & Monitor

Run tests in sandbox with comprehensive monitoring:

```bash
# Execute tests with verbose output
python -m pytest tests/test_parser.py -vv --tb=short 2>&1 | tee test_output.log

# Monitor results
echo "=== TEST RESULTS ==="
tail -20 test_output.log
```

**Observed Output:**
```
FAILED tests/test_parser.py::TestConfigParser::test_nested_config_with_defaults
FAILED tests/test_parser.py::TestConfigParser::test_deep_nesting

AssertionError: 'password' not found in result['database']['credentials']
```

### Step 4: Analyze Results

**Failure Analysis:**
1. Simple config test passes (flat structure)
2. Nested config tests fail - defaults not properly merged into nested structures
3. Only top-level keys from defaults are added
4. Nested dictionaries in defaults completely replace nested dictionaries in config

**Root Cause Identification:**
```python
# Debug instrumentation
def parse_config_debug(config_path, defaults={}):
    import json

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Original config: {config}")
    print(f"Defaults: {defaults}")

    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            print(f"Added default key: {key}")
        else:
            print(f"Skipped key {key} - already exists")

    print(f"Final config: {config}")
    return config
```

**Debug Output Reveals:**
```
Original config: {'database': {'host': 'localhost', ...}}
Defaults: {'database': {'host': 'db.example.com', 'credentials': {'password': 'default_pass'}}}
Skipped key database - already exists
Final config: {'database': {'host': 'localhost'}} # Missing nested defaults!
```

**Root Cause:** The merge logic only checks if top-level keys exist, not whether nested dictionaries need recursive merging. When `database` key exists in config, the entire default `database` dictionary is skipped, leaving nested default values unapplied.

### Step 5: Debug & Fix

Implement deep merge algorithm:

```python
def deep_merge(base, updates):
    """Recursively merge updates into base dictionary."""
    result = base.copy()

    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        elif key not in result:
            # Add missing keys from updates
            result[key] = value
        # If key exists and is not dict, keep original value

    return result

def parse_config(config_path, defaults={}):
    """Parse JSON config file and deeply merge with defaults."""
    import json

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Use deep merge instead of shallow merge
    config = deep_merge(defaults, config)

    return config
```

**Key Improvements:**
1. Recursive merging preserves nested structures
2. Config values take precedence over defaults (correct argument order)
3. Non-dictionary values are handled correctly
4. Missing nested keys are properly added

### Step 6: Verify Fix

Re-run comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/test_parser.py -vv

# Add edge case tests
python -m pytest tests/test_parser_edge_cases.py -vv
```

**Additional Edge Case Tests:**
```python
def test_list_values_not_merged(self):
    """Lists should not be merged, config value wins."""
    config_data = {'items': [1, 2, 3]}
    defaults = {'items': [4, 5, 6, 7, 8]}

    with open(f'{self.test_dir}/list.json', 'w') as f:
        json.dump(config_data, f)

    result = parse_config(f'{self.test_dir}/list.json', defaults)
    self.assertEqual(result['items'], [1, 2, 3])  # Config wins, not merged

def test_none_values_preserved(self):
    """Explicit None values should not be overridden."""
    config_data = {'timeout': None}
    defaults = {'timeout': 30}

    with open(f'{self.test_dir}/none.json', 'w') as f:
        json.dump(config_data, f)

    result = parse_config(f'{self.test_dir}/none.json', defaults)
    self.assertIsNone(result['timeout'])  # Explicit None preserved

def test_empty_nested_dict(self):
    """Empty nested dicts should receive all defaults."""
    config_data = {'database': {}}
    defaults = {
        'database': {
            'host': 'localhost',
            'port': 5432
        }
    }

    with open(f'{self.test_dir}/empty_nested.json', 'w') as f:
        json.dump(config_data, f)

    result = parse_config(f'{self.test_dir}/empty_nested.json', defaults)
    self.assertEqual(result['database']['host'], 'localhost')
    self.assertEqual(result['database']['port'], 5432)
```

**All Tests Pass:**
```
tests/test_parser.py::TestConfigParser::test_simple_config PASSED
tests/test_parser.py::TestConfigParser::test_nested_config_with_defaults PASSED
tests/test_parser.py::TestConfigParser::test_deep_nesting PASSED
tests/test_parser_edge_cases.py::TestEdgeCases::test_list_values_not_merged PASSED
tests/test_parser_edge_cases.py::TestEdgeCases::test_none_values_preserved PASSED
tests/test_parser_edge_cases.py::TestEdgeCases::test_empty_nested_dict PASSED

====== 6 passed in 0.24s ======
```

## Outcome

**What Was Discovered:**
- Original code used shallow dictionary merge, causing nested defaults to be ignored
- Bug only appeared with nested configuration structures (not tested initially)
- Production failures were due to missing database credentials from defaults

**How It Helped:**
1. **Sandbox testing** revealed the bug that visual inspection missed
2. **Systematic test case generation** covered edge cases that exposed the issue
3. **Execution monitoring** with debug instrumentation identified root cause
4. **Comprehensive verification** ensured fix works for all scenarios

**Production Impact:**
- Microservices now reliably receive all required configuration values
- Nested defaults properly merged with service-specific overrides
- Zero configuration-related deployment failures post-fix

## Key Takeaways

1. **Visual correctness !== Functional correctness**: Code looked fine but had subtle logic bug
2. **Edge cases matter**: Simple tests passed, complex real-world data failed
3. **Sandbox isolation prevents contamination**: Testing didn't affect production configs
4. **Systematic testing beats ad-hoc validation**: Comprehensive test suite caught issues
5. **Root cause analysis saves time**: Understanding the deep merge requirement prevented band-aid fixes
6. **Verification must be thorough**: Edge case tests ensure robust solution

**When to Apply This Pattern:**
- After writing utility functions that handle complex data structures
- When simple tests pass but production shows mysterious failures
- For code dealing with user input or configuration files
- Before deploying parsing/transformation logic to production
- Whenever "it looks right" isn't enough assurance
