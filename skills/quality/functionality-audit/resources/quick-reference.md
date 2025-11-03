# Functionality-Audit Quick Reference Card

**Version**: 1.0.0 | **Last Updated**: 2025-11-02

---

## ⚡ Quick Commands

### Validate Code (Auto-generate tests)
```bash
python validate_code.py --code-path ./src/module.py --auto-generate-tests
```

### Validate Code (Existing tests)
```bash
python validate_code.py --code-path ./app.py --test-cases ./tests/test_app.py
```

### Generate Tests Only
```bash
python test_generator.py --code-path ./src/module.py --output ./tests/test_module.py
```

### Create Sandbox
```bash
./sandbox_manager.sh create --template python --timeout 600
```

### List Sandboxes
```bash
./sandbox_manager.sh list
```

### Cleanup All
```bash
./sandbox_manager.sh cleanup-all --force
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Validation
```bash
# Generate tests + validate in one command
python validate_code.py \
  --code-path ./src \
  --auto-generate-tests \
  --sandbox-type local
```

### Workflow 2: Production Validation
```bash
# Step 1: Generate comprehensive tests
python test_generator.py \
  --code-path ./src/module.py \
  --output ./tests/test_module.py \
  --include-edge-cases \
  --include-boundaries

# Step 2: Validate in E2B sandbox
python validate_code.py \
  --code-path ./src/module.py \
  --test-cases ./tests/test_module.py \
  --sandbox-type e2b
```

### Workflow 3: Docker-based Validation
```bash
# Create Docker sandbox
SANDBOX_ID=$(./sandbox_manager.sh create --template python)

# Install dependencies
./sandbox_manager.sh install \
  --sandbox-id $SANDBOX_ID \
  --packages "pytest coverage hypothesis"

# Run validation
python validate_code.py \
  --code-path ./src \
  --auto-generate-tests \
  --sandbox-type docker

# Cleanup
./sandbox_manager.sh cleanup --sandbox-id $SANDBOX_ID
```

---

## 📋 Script Options

### `validate_code.py`
| Option | Description | Example |
|--------|-------------|---------|
| `--code-path` | Path to code (required) | `./src/module.py` |
| `--test-cases` | Existing test file | `./tests/test_module.py` |
| `--auto-generate-tests` | Auto-generate tests | (flag) |
| `--sandbox-type` | Sandbox type | `e2b`, `docker`, `local` |
| `--config` | Config file | `config.json` |

### `test_generator.py`
| Option | Description | Example |
|--------|-------------|---------|
| `--code-path` | Path to code (required) | `./src/module.py` |
| `--output` | Output test file (required) | `./tests/test_module.py` |
| `--language` | Language | `python`, `javascript` |
| `--include-edge-cases` | Include edge cases | (flag) |
| `--include-boundaries` | Include boundaries | (flag) |
| `--recursive` | Process directory | (flag) |

### `sandbox_manager.sh`
| Command | Options | Example |
|---------|---------|---------|
| `create` | `--template`, `--timeout` | `create --template python` |
| `start` | `--sandbox-id` | `start --sandbox-id abc123` |
| `stop` | `--sandbox-id` | `stop --sandbox-id abc123` |
| `cleanup` | `--sandbox-id`, `--force` | `cleanup --sandbox-id abc123` |
| `install` | `--sandbox-id`, `--packages` | `install --sandbox-id abc123 --packages "pytest"` |
| `monitor` | `--sandbox-id`, `--interval`, `--duration` | `monitor --sandbox-id abc123` |
| `list` | (none) | `list` |
| `cleanup-all` | `--force` | `cleanup-all --force` |

---

## 🚦 Exit Codes

### `validate_code.py`
- **0**: All tests passed
- **1**: Test failures detected
- **2**: Validation error (setup/execution failed)

### `test_generator.py`
- **0**: Tests generated successfully
- **1**: Generation failed

### `sandbox_manager.sh`
- **0**: Operation successful
- **1**: Operation failed

---

## 📊 Report Interpretation

### Verdict Types
| Verdict | Pass Rate | Description | Action |
|---------|-----------|-------------|--------|
| **APPROVED** | ≥95% | Production-ready | Deploy |
| **CONDITIONAL** | 80-95% | Minor issues | Review recommended |
| **NEEDS_WORK** | 50-80% | Significant issues | Refactor required |
| **REJECTED** | <50% | Major failures | Rewrite needed |

### Coverage Targets
- **Excellent**: ≥90%
- **Good**: 80-90%
- **Acceptable**: 70-80%
- **Low**: <70%

### Severity Levels
- **Critical**: Blocks functionality, immediate fix
- **High**: Major impact, fix soon
- **Medium**: Moderate impact, schedule fix
- **Low**: Minor impact, nice to have

---

## 🔧 Troubleshooting

### Problem: Timeout
**Solution**: Increase timeout
```bash
python validate_code.py \
  --code-path ./src \
  --sandbox-type e2b \
  --config config.json  # "timeout": 1200
```

### Problem: Out of Memory
**Solution**: Use larger sandbox or reduce test parallelism
```json
{
  "memory_limit_mb": 1024,
  "parallel_execution": false
}
```

### Problem: Tests Not Generated
**Solution**: Check code syntax
```bash
python -m py_compile src/module.py
```

### Problem: Sandbox Creation Fails
**Solution**: Check backend availability
```bash
# E2B
npx flow-nexus@latest --version

# Docker
docker --version
docker ps
```

---

## 📦 Dependencies

### Python
```bash
pip install pytest pytest-cov pytest-json-report pyyaml hypothesis
```

### System
```bash
# Ubuntu/Debian
sudo apt-get install jq docker.io

# macOS
brew install jq docker
```

### Node.js (for E2B)
```bash
npx flow-nexus@latest --version
```

---

## 🎓 Examples

### Example 1: Simple Function
```python
# code.py
def add(a, b):
    return a + b
```

```bash
python validate_code.py --code-path code.py --auto-generate-tests
# Expected: APPROVED (simple function, high confidence)
```

### Example 2: Complex Class
```python
# processor.py
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process(self):
        return [x * 2 for x in self.data]

    def filter(self, threshold):
        return [x for x in self.data if x > threshold]
```

```bash
python test_generator.py --code-path processor.py --output test_processor.py
python validate_code.py --code-path processor.py --test-cases test_processor.py
# Expected: APPROVED (well-defined class)
```

### Example 3: Edge Case Issues
```python
# buggy.py
def divide(a, b):
    return a / b  # No zero check!

def get_first(items):
    return items[0]  # No empty check!
```

```bash
python validate_code.py --code-path buggy.py --auto-generate-tests
# Expected: NEEDS_WORK (edge case failures detected)
# Failures: ZeroDivisionError, IndexError
```

---

## 🔗 Quick Links

- [Full Documentation](./README.md)
- [Skill Definition](../skill.md)
- [Templates](./templates/)
- [Examples](./examples/)

---

## 💡 Pro Tips

1. **Start Local**: Use `--sandbox-type local` for rapid iteration
2. **Use E2B for Final**: Switch to E2B for production validation
3. **Review Generated Tests**: Always review auto-generated tests
4. **Monitor Resources**: Use `monitor` command for long-running tests
5. **Cleanup Regularly**: Run `cleanup-all` to prevent resource leaks
6. **Track Coverage**: Aim for 80%+ coverage
7. **Fix Critical First**: Address critical/high severity issues first
8. **Iterate Quickly**: Run validation after each fix

---

## 📞 Support

For issues or questions:
1. Check [Full Documentation](./README.md)
2. Review [Troubleshooting](#-troubleshooting) section
3. Inspect validation report for detailed findings
4. Check sandbox logs: `/tmp/sandbox-manager-YYYYMMDD.log`

---

**Remember**: The goal is code that works, not just code that looks good!

---

*Generated for functionality-audit skill v1.0.0*
