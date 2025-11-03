# Test 1: Basic Hook Installation and Execution

## Overview
Test basic hook installation, configuration, and execution to verify the hooks-automation system works correctly.

## Prerequisites
- Node.js 18+ installed
- npm/npx available
- Bash shell (Git Bash on Windows, native on Unix)
- Claude Flow MCP configured

## Test Scenarios

### Scenario 1.1: Hook Installation

**Objective**: Verify hook installer creates required directory structure and files

**Steps**:
1. Run hook installer:
   ```bash
   bash resources/scripts/hook-installer.sh --verbose
   ```

2. Verify hooks directory created:
   ```bash
   ls -la ~/.claude-flow/hooks/
   ```

3. Check for subdirectories:
   ```bash
   ls ~/.claude-flow/hooks/{pre-task,post-edit,post-task,session}
   ```

**Expected Results**:
- Hooks directory exists at `~/.claude-flow/hooks/`
- Four subdirectories created: `pre-task`, `post-edit`, `post-task`, `session`
- Each subdirectory contains `config.yaml` or `config.json`
- Each subdirectory contains executable `run.sh` script

**Success Criteria**:
- ✅ All directories exist
- ✅ All config files present
- ✅ All scripts are executable
- ✅ No errors during installation

**Troubleshooting**:
- If directories missing: Check permissions on `~/.claude-flow/`
- If scripts not executable: Run `chmod +x ~/.claude-flow/hooks/*/run.sh`
- If templates missing: Verify `resources/templates/` exists

---

### Scenario 1.2: Configuration Validation

**Objective**: Validate hook configuration files have correct syntax

**Steps**:
1. Run validator on all hooks:
   ```bash
   python resources/scripts/hook-validator.py --verbose
   ```

2. Run validator on specific hook:
   ```bash
   python resources/scripts/hook-validator.py --hook pre-task
   ```

3. Generate validation report:
   ```bash
   python resources/scripts/hook-validator.py --report validation-report.md
   ```

**Expected Results**:
- All YAML files parse without errors
- All JSON files parse without errors
- Configuration structure matches expected schema
- No missing required fields

**Success Criteria**:
- ✅ YAML syntax validation passes
- ✅ JSON syntax validation passes
- ✅ All required sections present in configs
- ✅ Validator exits with code 0

**Troubleshooting**:
- YAML errors: Check indentation (use spaces, not tabs)
- JSON errors: Validate with `jq` or online validator
- Missing fields: Compare with templates in `resources/templates/`

---

### Scenario 1.3: Pre-Task Hook Execution

**Objective**: Execute pre-task hook and verify it runs without errors

**Steps**:
1. Execute pre-task hook manually:
   ```bash
   npx claude-flow@alpha hooks pre-task --description "Test task"
   ```

2. Check hook wrapper script:
   ```bash
   bash ~/.claude-flow/hooks/pre-task/run.sh "Test task"
   ```

3. Verify output contains expected actions:
   - Agent assignment
   - Resource preparation
   - Command validation

**Expected Results**:
- Hook executes without errors
- Agent is assigned based on task description
- Resources are prepared (memory loaded, files checked)
- Command validation completes

**Success Criteria**:
- ✅ Hook executes successfully (exit code 0)
- ✅ Expected actions listed in output
- ✅ No error messages logged
- ✅ Execution completes in < 5 seconds

**Troubleshooting**:
- Timeout errors: Increase timeout in config
- Agent assignment fails: Check file type mappings
- Memory errors: Verify Memory MCP is configured

---

### Scenario 1.4: Post-Edit Hook Execution

**Objective**: Execute post-edit hook and verify formatting works

**Steps**:
1. Create test file:
   ```bash
   mkdir -p /tmp/hook-test
   echo "const x={a:1,b:2};" > /tmp/hook-test/test.js
   ```

2. Execute post-edit hook:
   ```bash
   npx claude-flow@alpha hooks post-edit \
     --file /tmp/hook-test/test.js \
     --memory-key "hooks/test/edit1"
   ```

3. Check file was formatted:
   ```bash
   cat /tmp/hook-test/test.js
   ```

4. Verify memory was updated:
   ```bash
   npx claude-flow@alpha memory retrieve \
     --key "hooks/test/edit1"
   ```

**Expected Results**:
- File is auto-formatted (code reformatted with proper spacing)
- Memory update stores edit metadata
- Neural patterns learned from edit
- No errors during execution

**Success Criteria**:
- ✅ File content is properly formatted
- ✅ Memory contains edit information
- ✅ Hook completes successfully
- ✅ Original file functionality preserved

**Troubleshooting**:
- Formatting fails: Check if prettier is installed globally
- Memory not updated: Verify Memory MCP configuration
- File not found: Check file path is absolute

---

### Scenario 1.5: Session Hook Execution

**Objective**: Test session start and end hooks

**Steps**:
1. Start new session:
   ```bash
   npx claude-flow@alpha hooks session-start
   ```

2. Perform some operations (create files, run commands)

3. End session:
   ```bash
   npx claude-flow@alpha hooks session-end --export-metrics true
   ```

4. Verify session state saved:
   ```bash
   ls -la ~/.claude-flow/sessions/
   ```

5. Check session summary:
   ```bash
   cat ~/.claude-flow/summaries/session-*.md
   ```

**Expected Results**:
- Session starts successfully
- Context is restored (if previous session exists)
- Session state persisted on end
- Summary generated with metrics

**Success Criteria**:
- ✅ Session file created in `~/.claude-flow/sessions/`
- ✅ Summary file created in `~/.claude-flow/summaries/`
- ✅ Metrics exported to `~/.claude-flow/metrics/`
- ✅ Session can be restored later

**Troubleshooting**:
- Session not saved: Check write permissions
- Summary empty: Verify summary generation enabled in config
- Metrics missing: Enable metrics in session-hooks.yaml

---

## Test Execution Summary

### Quick Test Run
Execute all basic tests in sequence:

```bash
# 1. Install hooks
bash resources/scripts/hook-installer.sh

# 2. Validate configuration
python resources/scripts/hook-validator.py

# 3. Test pre-task hook
npx claude-flow@alpha hooks pre-task --description "Quick test"

# 4. Test post-edit hook (create temp file first)
echo "const test = {a:1};" > /tmp/test.js
npx claude-flow@alpha hooks post-edit --file /tmp/test.js

# 5. Test session hooks
npx claude-flow@alpha hooks session-start
npx claude-flow@alpha hooks session-end
```

### Expected Total Duration
- Installation: ~30 seconds
- Validation: ~5 seconds
- Pre-task test: ~2 seconds
- Post-edit test: ~3 seconds
- Session test: ~5 seconds
- **Total**: ~45 seconds

### Success Metrics
| Test | Expected Result | Pass/Fail |
|------|----------------|-----------|
| Installation | All files created | ⬜ |
| Validation | No errors | ⬜ |
| Pre-task | Agent assigned | ⬜ |
| Post-edit | File formatted | ⬜ |
| Session | State saved | ⬜ |

## Cleanup

After testing, optionally clean up test files:

```bash
# Remove test files
rm -f /tmp/test.js
rm -rf /tmp/hook-test

# Keep hooks installed for actual use
# OR remove for clean slate:
# rm -rf ~/.claude-flow/hooks
```

## Next Steps

After basic tests pass:
1. Proceed to Test 2: Hook Chaining
2. Review test-2-hook-chaining.md
3. Test advanced hook coordination features

## References

- Main skill documentation: `skill.md`
- Resource scripts: `resources/scripts/`
- Configuration templates: `resources/templates/`
- Hook validator: `resources/scripts/hook-validator.py`
