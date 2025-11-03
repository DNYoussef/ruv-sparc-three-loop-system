# Functionality Audit - Verified Code Through Sandbox Testing

**Version**: 1.0.0 (MECE-Structured Universal Template)
**Purpose**: Validate that code actually works through sandbox testing, execution verification, and systematic debugging
**Quality Tier**: Gold (12+ files)

## 🎯 What This Is

Functionality Audit transforms theoretical correctness into verified functionality by actually executing code in isolated sandbox environments. Rather than assuming code is correct because it looks right or passes cursory checks, this skill systematically tests code with realistic inputs, verifies outputs match expectations, and debugs any issues discovered through a proven 6-step workflow.

**Key Difference**: This skill provides **single-agent execution verification** focused on functional correctness. For multi-agent Byzantine consensus validation to detect "theater code," use the `theater-detection-audit` skill instead.

## 🚀 Quick Start

### For Code Validation

1. **Invoke the skill** after generating or modifying code:
   ```bash
   # Via Claude Code
   "Run functionality audit on /path/to/code.py"

   # Auto-triggered by keywords
   "Does this code work?" → Auto-invokes functionality-audit
   "Validate this implementation" → Auto-invokes functionality-audit
   ```

2. **Provide context** for targeted testing:
   - Paths to code files to test
   - Description of expected behavior
   - Available test data or sample inputs
   - Any known issues or concerns

3. **Review the audit report** with:
   - Execution summary (pass/fail rates)
   - Detailed test results with evidence
   - Identified bugs with root causes
   - Systematic fixes with verification

## 📋 When to Use This Skill

### Auto-Trigger Conditions

The skill automatically activates when Claude detects:
- **"does it work?"** - Verify functionality after code generation
- **"validate"** - Ensure code delivers intended behavior
- **"test this code"** - Execute code with realistic inputs
- **AFTER code generation** - Automatic quality gate before deployment

### Manual Invocation Scenarios

Use the functionality-audit skill when:
- Code appears correct but behavior seems off
- After integrating code from multiple sources
- Before production releases as final validation
- When correctness is critical (financial, medical, safety)
- When code complexity makes visual inspection insufficient
- When debugging existing code that fails intermittently

### Not Needed When

Skip this skill when:
- Code is trivial (1-2 lines) with obvious correctness
- You need multi-agent consensus validation → Use `theater-detection-audit`
- You're validating code style → Use `style-audit`
- You're checking for security vulnerabilities → Use `security`

## 📁 Structure Overview

This skill follows the MECE (Mutually Exclusive, Collectively Exhaustive) universal template:

```
functionality-audit/
│
├── SKILL.md                          # ✅ Imperative instructions for Claude
├── README.md                         # ✅ This file - overview & quick start
│
├── examples/                         # ⚙️ Coming soon - concrete usage examples
│   ├── example-1-python-api.md       # Python API sandbox testing
│   ├── example-2-javascript-app.md   # JavaScript app execution verification
│   └── example-3-edge-cases.md       # Edge case debugging workflow
│
├── references/                       # ⚙️ Supporting documentation
│   ├── best-practices.md             # Debugging techniques & patterns
│   ├── sandbox-tools.md              # Python, JS, Docker, Cloud sandboxes
│   └── troubleshooting.md            # Common issues & solutions
│
├── resources/                        # ⚙️ Executable & reusable assets
│   ├── scripts/                      # Automation utilities
│   │   ├── validate.py               # Python validation script
│   │   └── sandbox-setup.sh          # Sandbox environment setup
│   └── templates/                    # Boilerplate templates
│       ├── test-case-template.yaml   # Test case structure
│       └── audit-report-template.md  # Output report format
│
└── graphviz/                         # ⚙️ Process diagrams
    └── functionality-audit-process.dot  # 6-step debugging workflow

**Status**: Resources placeholders exist. Scripts/templates to be added in future iterations.
```

### MECE Organization

- **SKILL.md**: Complete methodology, debugging workflow, best practices (imperative voice)
- **README.md**: Progressive disclosure overview for quick understanding
- **examples/**: Concrete use cases showing the skill in action (coming soon)
- **references/**: Deep-dive documentation on techniques and tools
- **resources/**: Executable scripts and reusable templates
- **graphviz/**: Visual workflow diagrams

## 🔬 Core Methodology

### Sandbox Testing Workflow

1. **Sandbox Creation** - Isolated environments that replicate production safely
2. **Test Case Generation** - Comprehensive coverage including edge cases
3. **Execution Monitoring** - Track outputs, exceptions, resource usage
4. **Output Verification** - Compare actual vs. expected results
5. **Failure Analysis** - Systematic root cause investigation
6. **Systematic Debugging** - Proven 6-step fix workflow

### The 6-Step Debugging Workflow

When audits reveal bugs, the skill follows this systematic approach:

```
1. Reproduce Reliably    → Create minimal test case
2. Understand Root Cause → Trace execution path, inspect state
3. Design the Fix        → Plan changes, consider side effects
4. Implement with Care   → Apply best practices, add comments
5. Verify Thoroughly     → Regression testing, edge cases
6. Document the Fix      → Record what, why, how, risks
```

### Integration with Tools

The skill integrates with various sandbox environments:

- **Python**: venv/conda + pytest + coverage.py + pdb
- **JavaScript**: Node.js + Jest/Mocha + Istanbul + debugger
- **Containerized**: Docker + docker-compose + isolated networks
- **Cloud**: E2B sandboxes, Flow-Nexus, AWS/GCP/Azure test environments

## 💡 Key Features

### What Makes This Skill Powerful

1. **Execution Verification** - Tests semantic correctness, not just syntax
2. **Sandbox Safety** - Isolated environments prevent production damage
3. **Systematic Debugging** - Proven workflow eliminates guesswork
4. **Comprehensive Testing** - Normal cases, boundaries, errors, edge cases
5. **Root Cause Analysis** - Fixes underlying problems, not just symptoms
6. **Evidence-Based Reports** - Structured output with test results and fixes

### Benefits

- **Transforms "looks correct" into "verified correct"**
- **Catches integration bugs** that static analysis misses
- **Prevents production failures** through pre-deployment validation
- **Reduces debugging time** with systematic investigation
- **Builds confidence** in code quality through evidence

## 📊 Examples

### Coming Soon

The `examples/` directory will contain:

1. **Python API Testing** - Validate REST API endpoints with realistic requests
2. **JavaScript App Testing** - Execute React/Node.js apps in sandboxes
3. **Edge Case Debugging** - Handle off-by-one errors, null pointers, race conditions

**Current Status**: Examples directory structure prepared. Content to be added in future updates.

## 🔗 Related Skills

### Quality Assurance Pipeline

1. **functionality-audit** (this skill) - Execution verification, single-agent testing
2. **theater-detection-audit** - Multi-agent Byzantine consensus, theater code detection
3. **style-audit** - Code style, readability, maintainability
4. **production-readiness** - Complete audit pipeline for deployment

### When to Use Which

| Scenario | Recommended Skill |
|----------|-------------------|
| "Does this code work?" | `functionality-audit` |
| "Is this code real or fake?" | `theater-detection-audit` |
| "Is this code clean and readable?" | `style-audit` |
| "Ready for production?" | `production-readiness` |
| "Quick validation before commit" | `quick-quality-check` |

## 🛠️ Advanced Usage

### Customizing Test Cases

Provide specific test scenarios for targeted validation:
```
"Run functionality audit on auth.py with:
- Test case 1: Valid login credentials
- Test case 2: Invalid password (expect 401)
- Test case 3: SQL injection attempt (expect sanitization)
- Test case 4: Concurrent login attempts (race conditions)"
```

### Integration with CI/CD

Incorporate functionality audits into automated pipelines:
```yaml
# .github/workflows/quality-gate.yml
- name: Functionality Audit
  run: |
    claude-code invoke functionality-audit \
      --files "src/**/*.py" \
      --config "test-config.yaml" \
      --report "audit-report.md"
```

### Debugging Complex Issues

For mysterious bugs, combine with other debugging skills:
```
"Run functionality audit + reverse-engineer-debug on payment.js
Focus on intermittent race condition in checkout flow"
```

## 📖 Documentation

### Full Methodology

Read `SKILL.md` for:
- Complete sandbox testing methodology
- Detailed debugging techniques (binary search, rubber duck, hypothesis-driven)
- Integration with sandbox tools (Python, JS, Docker, Cloud)
- Output report structure
- Claude Code workflow integration

### Supporting References

See `references/` for:
- **best-practices.md** - Debugging patterns and techniques
- **sandbox-tools.md** - Platform-specific testing guides
- **troubleshooting.md** - Common issues and solutions

### Visual Workflows

See `graphviz/` for:
- **functionality-audit-process.dot** - 6-step debugging workflow diagram

## 🎓 Quality Tier: Gold

This skill achieves **Gold tier** (12+ files) with:
- ✅ SKILL.md - Complete imperative instructions
- ✅ README.md - Progressive disclosure overview
- ✅ Structured resources/ with scripts/ and templates/
- ✅ GraphViz workflow diagram
- ✅ MECE organization following universal template
- ⚙️ examples/ and references/ directories prepared for future expansion

## 🚦 Getting Started Checklist

- [ ] Read this README for overview
- [ ] Review `SKILL.md` for full methodology
- [ ] Try auto-trigger: "Does my code work?"
- [ ] Provide context: file paths, expected behavior, test data
- [ ] Review audit report with execution evidence
- [ ] Apply systematic fixes using 6-step workflow
- [ ] Integrate with `theater-detection-audit` for complete validation

## 📝 Notes

### Difference from Theater Detection

**functionality-audit**:
- Single-agent execution verification
- Focuses on functional correctness
- Fast validation (30-60s for typical code)
- Tests semantic correctness with realistic inputs

**theater-detection-audit**:
- Multi-agent Byzantine consensus
- Detects fake/incomplete implementations
- Longer validation (60-120s for Byzantine agreement)
- Uses 6 agents with 67% consensus threshold

**Use both** for critical production code: `functionality-audit` ensures it works, `theater-detection-audit` ensures it's real.

### Memory Integration

All audit results are automatically stored in Memory-MCP with WHO/WHEN/PROJECT/WHY tagging:
- **WHO**: functionality-audit agent
- **WHEN**: ISO timestamp
- **PROJECT**: Your project name
- **WHY**: testing, bugfix, validation

Retrieve past audit results with:
```bash
npx claude-flow@alpha memory retrieve --key "audit/functionality/*"
```

---

**Remember**: Code that looks correct is not the same as code that works correctly. Functionality Audit bridges that gap through systematic execution verification.
