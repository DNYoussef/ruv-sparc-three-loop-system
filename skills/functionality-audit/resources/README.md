# Functionality-Audit Resources

Production-ready automation scripts and templates for the functionality-audit skill.

## 📁 Directory Structure

```
resources/
├── scripts/
│   ├── validate_code.py      # Main validation orchestrator
│   ├── sandbox_manager.sh    # Sandbox lifecycle management
│   └── test_generator.py     # Automated test case generation
├── templates/
│   ├── validation-report.yaml    # Report template
│   └── sandbox-config.json       # E2B sandbox configuration
└── README.md                 # This file
```

---

## 🚀 Scripts

### 1. `validate_code.py` - Main Validation Orchestrator

**Purpose**: End-to-end code validation through sandbox testing.

**Features**:
- Creates isolated sandbox environments (E2B, Docker, or local)
- Generates or uses existing test cases
- Executes tests with monitoring
- Analyzes results and identifies failures
- Generates comprehensive validation reports
- Automatic cleanup of resources

**Usage**:

```bash
# Basic validation with auto-generated tests
python validate_code.py --code-path ./src/module.py --auto-generate-tests

# Validation with existing test suite
python validate_code.py --code-path ./app.py --test-cases ./tests/test_app.py

# Use E2B sandbox
python validate_code.py --code-path ./src --sandbox-type e2b --auto-generate-tests

# Use Docker sandbox
python validate_code.py --code-path ./app --sandbox-type docker --test-cases ./tests

# Use configuration file
python validate_code.py --config validation-config.json
```

**Configuration File Example** (`validation-config.json`):

```json
{
  "code_path": "./src/module.py",
  "test_cases": "./tests/test_module.py",
  "auto_generate_tests": false,
  "sandbox_type": "e2b"
}
```

**Output**:
- Validation report (YAML or JSON)
- Exit code: 0 (all tests passed), 1 (failures), 2 (error)

**Workflow**:
1. **Create Sandbox** - Isolated test environment
2. **Generate Tests** - Auto-generate or load existing
3. **Execute Tests** - Run with coverage tracking
4. **Analyze Results** - Identify failures and root causes
5. **Report Findings** - Generate comprehensive report
6. **Cleanup** - Remove sandbox resources

**Supported Languages**: Python, JavaScript, TypeScript

---

### 2. `sandbox_manager.sh` - Sandbox Lifecycle Manager

**Purpose**: Manage creation, startup, shutdown, and cleanup of E2B sandboxes.

**Features**:
- Create sandboxes with custom templates
- Start/stop/cleanup operations
- Install dependencies
- Monitor resource usage
- State management
- Batch operations

**Usage**:

```bash
# Create Python sandbox
./sandbox_manager.sh create --template python --timeout 600

# Start existing sandbox
./sandbox_manager.sh start --sandbox-id abc123

# Stop sandbox
./sandbox_manager.sh stop --sandbox-id abc123

# Install packages
./sandbox_manager.sh install --sandbox-id abc123 --packages "pytest coverage hypothesis"

# Monitor resources (5s interval, 2min duration)
./sandbox_manager.sh monitor --sandbox-id abc123 --interval 5 --duration 120

# List all sandboxes
./sandbox_manager.sh list

# Cleanup specific sandbox
./sandbox_manager.sh cleanup --sandbox-id abc123

# Force cleanup all sandboxes
./sandbox_manager.sh cleanup-all --force
```

**Commands**:
- `create` - Create new sandbox
- `start` - Start existing sandbox
- `stop` - Stop running sandbox
- `cleanup` - Cleanup sandbox resources
- `install` - Install packages in sandbox
- `monitor` - Monitor sandbox resources
- `list` - List all sandboxes
- `cleanup-all` - Cleanup all sandboxes
- `help` - Show help message

**State Management**:
- Tracks sandbox state in `/tmp/sandbox-state.json`
- Logs operations to `/tmp/sandbox-manager-YYYYMMDD.log`
- Persistent across sessions

**Supported Sandbox Types**: E2B, Docker

---

### 3. `test_generator.py` - Automated Test Case Generator

**Purpose**: Analyze code to identify functions/classes and generate comprehensive test cases.

**Features**:
- AST-based code analysis (Python)
- Function and class extraction
- Auto-generate test fixtures
- Edge case testing
- Boundary value testing
- Mock data generation
- pytest and Jest output

**Usage**:

```bash
# Basic test generation
python test_generator.py --code-path ./src/module.py --output ./tests/test_module.py

# Generate with edge cases and boundaries
python test_generator.py \
  --code-path ./app.py \
  --output ./tests/test_app.py \
  --include-edge-cases \
  --include-boundaries

# JavaScript test generation
python test_generator.py \
  --code-path ./src/utils.js \
  --output ./tests/utils.test.js \
  --language javascript

# Recursive directory processing
python test_generator.py \
  --code-path ./src \
  --output ./tests \
  --recursive
```

**Output**:
- pytest-compatible tests (Python)
- Jest-compatible tests (JavaScript/TypeScript)
- Comprehensive test coverage

**Generated Test Types**:
1. **Basic Functionality Tests** - Core behavior validation
2. **Edge Case Tests** - None/null/empty inputs
3. **Boundary Tests** - Large/small/negative values
4. **Class Tests** - Initialization and method testing

**Supported Languages**: Python (full), JavaScript/TypeScript (basic)

---

## 📄 Templates

### 1. `validation-report.yaml` - Validation Report Template

**Purpose**: Standardized report format for validation results.

**Sections**:
- **Metadata**: Skill name, version, timestamp
- **Code Information**: Path, language, size, lines
- **Sandbox Information**: ID, type, template
- **Test Results**: Total, passed, failed, duration
- **Coverage**: Line/branch coverage percentages
- **Failures**: Detailed failure information with root cause analysis
- **Recommendations**: Prioritized improvement suggestions
- **Quality Metrics**: Complexity, maintainability, security
- **Verdict**: APPROVED, CONDITIONAL, NEEDS_WORK, or REJECTED
- **Audit Trail**: Action history

**Example**:

```yaml
validation_report:
  skill_name: functionality-audit
  timestamp: "2025-11-02T10:30:00Z"
  code_path: "/workspace/src/module.py"
  sandbox_id: "abc123"

  test_results:
    total_tests: 42
    passed: 40
    failed: 2
    pass_rate: 95.2

  failures:
    - test_name: "test_module_edge_case"
      expected: "ValueError raised"
      actual: "None returned"
      root_cause: "Missing input validation"
      severity: "high"

  recommendations:
    - priority: "high"
      category: "correctness"
      message: "Add input validation for empty strings"

  verdict:
    status: "CONDITIONAL"
    confidence: 0.85
    summary: "Minor issues found, review recommended"
```

---

### 2. `sandbox-config.json` - E2B Sandbox Configuration

**Purpose**: Comprehensive sandbox configuration for multiple languages and frameworks.

**Supported Templates**:
- **python** - Python 3.11 with pytest, coverage, hypothesis
- **node** - Node.js 20 with Jest, Mocha, Chai
- **typescript** - TypeScript with ts-jest
- **react** - React with Testing Library
- **nextjs** - Next.js with Jest
- **base** - Ubuntu 22.04 minimal

**Configuration Options**:
- Base image
- Timeout (default: 600s)
- Memory limit (default: 512MB)
- CPU limit (default: 1.0 core)
- Network isolation (default: disabled)
- Environment variables
- Package installation
- Working directory
- Startup script

**Security Features**:
- Network isolation
- Capability dropping
- Process limits
- Resource ulimits

**Example Configuration**:

```json
{
  "template": "python",
  "timeout": 600,
  "memory_limit_mb": 512,
  "cpu_limit": 1.0,
  "network_enabled": false,
  "install_packages": ["pytest", "coverage", "hypothesis"],
  "working_dir": "/workspace"
}
```

---

## 🔧 Integration with functionality-audit Skill

### Skill Invocation

The functionality-audit skill automatically uses these resources:

```bash
# Skill invokes validate_code.py
npx claude-code skill functionality-audit --code-path ./src/module.py
```

**Internal Flow**:
1. Skill reads user request
2. Calls `validate_code.py` with appropriate arguments
3. `validate_code.py` calls `sandbox_manager.sh` to create sandbox
4. `validate_code.py` calls `test_generator.py` if auto-generation requested
5. Tests executed in sandbox
6. Report generated from `validation-report.yaml` template
7. Results returned to user

### Manual Invocation

For standalone testing outside the skill:

```bash
# Direct script execution
python resources/scripts/validate_code.py \
  --code-path ./src/app.py \
  --auto-generate-tests \
  --sandbox-type e2b

# Generate tests only
python resources/scripts/test_generator.py \
  --code-path ./src/app.py \
  --output ./tests/test_app.py \
  --include-edge-cases

# Manage sandboxes
./resources/scripts/sandbox_manager.sh create --template python
./resources/scripts/sandbox_manager.sh list
```

---

## 📊 Expected Outcomes

### Successful Validation

```
==============================================================
FUNCTIONALITY-AUDIT CODE VALIDATOR
==============================================================
[1/5] Creating e2b sandbox...
✓ E2B sandbox created: sandbox_abc123

[2/5] Generating test cases...
✓ Generated tests: /workspace/generated_tests.py

[3/5] Executing tests...
✓ Tests completed: 42/42 passed

[4/5] Analyzing results...
✓ Found 0 failures, generated 3 recommendations

[5/5] Generating report...
✓ Report generated: validation-report-20251102-103000.yaml

==============================================================
VALIDATION COMPLETE
==============================================================
Report: validation-report-20251102-103000.yaml
Tests: 42/42 passed
Failures: 0
Verdict: APPROVED - Code is production-ready
```

### Failed Validation

```
==============================================================
VALIDATION COMPLETE
==============================================================
Report: validation-report-20251102-103500.yaml
Tests: 38/42 passed
Failures: 4
Verdict: NEEDS WORK - Significant issues, refactoring required

Blocking Issues:
  - Missing input validation (HIGH severity)
  - Unhandled edge cases (MEDIUM severity)

Next Steps:
  1. Add input validation for empty strings
  2. Handle None values in process_data()
  3. Increase test coverage to 80%+
```

---

## 🛠️ Dependencies

**Python Scripts**:
- Python 3.8+
- pytest, pytest-cov, pytest-json-report (auto-installed)
- PyYAML (optional, for YAML reports)

**Bash Scripts**:
- bash 4.0+
- jq (JSON processing)
- docker (optional, for Docker sandboxes)
- npx (for E2B sandboxes via flow-nexus)

**Sandbox Backends**:
- E2B (flow-nexus): `npx flow-nexus@latest`
- Docker: `docker` command
- Local: Python/Node.js environment

**Installation**:

```bash
# Python dependencies
pip install pytest pytest-cov pytest-json-report pyyaml

# System dependencies (Ubuntu/Debian)
sudo apt-get install jq docker.io

# E2B via flow-nexus
npx flow-nexus@latest --version
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Sandbox Creation Fails**

```
ERROR: Failed to create sandbox: Connection timeout
```

**Solution**: Check network connectivity, verify flow-nexus installation:

```bash
npx flow-nexus@latest --version
npx flow-nexus@latest sandbox list
```

**2. Test Generation Fails**

```
ERROR: Test generation failed: SyntaxError
```

**Solution**: Ensure code is syntactically valid before generating tests:

```bash
python -m py_compile src/module.py
```

**3. Permission Errors**

```
ERROR: Permission denied: /workspace
```

**Solution**: Check file permissions, ensure sandbox has write access:

```bash
chmod +x resources/scripts/*.py
chmod +x resources/scripts/*.sh
```

**4. Timeout Issues**

```
WARNING: Test execution timed out after 600s
```

**Solution**: Increase timeout in sandbox config:

```bash
# Edit sandbox-config.json
"timeout": 1200  # Increase to 20 minutes
```

---

## 📚 Best Practices

### Test Generation

1. **Review Generated Tests**: Always review auto-generated tests before execution
2. **Add Custom Logic**: Enhance with domain-specific assertions
3. **Use Fixtures**: Leverage pytest fixtures for complex setups
4. **Mock External Dependencies**: Use mocks for API calls, databases

### Sandbox Management

1. **Always Cleanup**: Use `cleanup` or `cleanup-all` after testing
2. **Monitor Resources**: Use `monitor` command for long-running tests
3. **Set Appropriate Limits**: Adjust memory/CPU for workload
4. **Disable Network**: Keep network disabled unless required

### Validation Workflow

1. **Start Small**: Test individual modules before full integration
2. **Iterate Quickly**: Use local sandbox for rapid iteration
3. **Use E2B for Production**: Use E2B sandbox for final validation
4. **Track Metrics**: Review coverage and quality metrics

---

## 🔗 Related Documentation

- [functionality-audit Skill Documentation](../skill.md)
- [SPARC Methodology Guide](../../../docs/SPARC-METHODOLOGY.md)
- [E2B Sandbox Documentation](https://e2b.dev/docs)
- [pytest Documentation](https://docs.pytest.org/)
- [Jest Documentation](https://jestjs.io/)

---

## 📝 License

These resources are part of the functionality-audit skill for the SPARC Three-Loop System.

---

**Generated**: 2025-11-02
**Version**: 1.0.0
**Maintainer**: functionality-audit skill
