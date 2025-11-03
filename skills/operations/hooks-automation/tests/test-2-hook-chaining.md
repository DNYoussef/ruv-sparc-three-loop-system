# Test 2: Hook Chaining and Sequential Execution

## Overview
Test chaining multiple hooks together for complex workflows, verifying hooks execute in correct order and share context between executions.

## Prerequisites
- Test 1 (Basic Hooks) passed successfully
- Hooks installed and validated
- Hook manager running: `node resources/scripts/hook-manager.js start`
- Memory MCP configured

## Test Scenarios

### Scenario 2.1: Pre-Task → Post-Task Chain

**Objective**: Verify hooks execute in sequence with context passing

**Steps**:
1. Enable both pre-task and post-task hooks:
   ```bash
   node resources/scripts/hook-manager.js enable pre-task
   node resources/scripts/hook-manager.js enable post-task
   ```

2. Create test workflow:
   ```bash
   # Pre-task: Prepare environment
   npx claude-flow@alpha hooks pre-task --description "Create test file"

   # Simulate task execution
   echo "console.log('test');" > /tmp/workflow-test.js

   # Post-task: Cleanup and record
   npx claude-flow@alpha hooks post-task --task-id "workflow-test-1"
   ```

3. Verify context was shared:
   ```bash
   npx claude-flow@alpha memory retrieve --key "hooks/workflow-test-1"
   ```

**Expected Results**:
- Pre-task hook prepares environment (creates directories, loads context)
- Post-task hook has access to pre-task context
- Memory stores complete workflow state
- Both hooks reference same task ID

**Success Criteria**:
- ✅ Pre-task completes before post-task
- ✅ Context is available in post-task
- ✅ Task ID matches across hooks
- ✅ Memory contains full workflow history

**Troubleshooting**:
- Context missing: Check memory key naming consistency
- Execution order wrong: Verify hook priorities in config
- Memory not shared: Ensure Memory MCP is running

---

### Scenario 2.2: Edit → Format → Memory Chain

**Objective**: Test post-edit hook action chaining

**Steps**:
1. Create unformatted test file:
   ```bash
   cat > /tmp/chain-test.js <<'EOF'
   const data={name:"test",value:123,nested:{a:1,b:2}};
   function process(x){return x*2;}
   EOF
   ```

2. Execute post-edit hook (triggers action chain):
   ```bash
   npx claude-flow@alpha hooks post-edit \
     --file /tmp/chain-test.js \
     --memory-key "hooks/chain/edit1"
   ```

3. Verify each action completed:
   ```bash
   # Check formatting
   cat /tmp/chain-test.js

   # Check memory update
   npx claude-flow@alpha memory retrieve --key "hooks/chain/edit1"

   # Check metrics
   node resources/scripts/hook-manager.js metrics
   ```

**Expected Results**:
- **Action 1 (auto-format)**: File is formatted with proper spacing
- **Action 2 (memory-update)**: Edit stored in Memory MCP with metadata
- **Action 3 (neural-training)**: Patterns learned from code style
- **Action 4 (quality-check)**: Syntax validated (no errors)

**Success Criteria**:
- ✅ All 4 actions execute in sequence
- ✅ File is properly formatted
- ✅ Memory contains tagged metadata (WHO/WHEN/PROJECT/WHY)
- ✅ Neural patterns updated
- ✅ Quality checks pass

**Troubleshooting**:
- Formatting skipped: Check formatter is installed (prettier)
- Memory update fails: Verify tagging protocol enabled
- Neural training errors: Check AgentDB configuration

---

### Scenario 2.3: Session Start → Multiple Tasks → Session End

**Objective**: Test complete session lifecycle with multiple operations

**Steps**:
1. Start session with context restoration:
   ```bash
   SESSION_ID="test-chain-$(date +%s)"
   npx claude-flow@alpha hooks session-start
   ```

2. Perform multiple operations:
   ```bash
   # Operation 1: Create file
   npx claude-flow@alpha hooks pre-task --description "Create config"
   echo '{"setting": "value"}' > /tmp/config.json
   npx claude-flow@alpha hooks post-edit --file /tmp/config.json

   # Operation 2: Create code
   npx claude-flow@alpha hooks pre-task --description "Create module"
   echo 'export const test = () => {};' > /tmp/module.js
   npx claude-flow@alpha hooks post-edit --file /tmp/module.js

   # Operation 3: Create documentation
   npx claude-flow@alpha hooks pre-task --description "Create docs"
   echo '# Test Documentation' > /tmp/README.md
   npx claude-flow@alpha hooks post-edit --file /tmp/README.md
   ```

3. End session and check persistence:
   ```bash
   npx claude-flow@alpha hooks session-end --export-metrics true
   ```

4. Verify session artifacts:
   ```bash
   # Check session state
   ls -la ~/.claude-flow/sessions/

   # Check summary
   cat ~/.claude-flow/summaries/session-*.md

   # Check metrics
   cat ~/.claude-flow/metrics/session-*.json
   ```

**Expected Results**:
- Session starts and restores previous context (if exists)
- All 3 operations execute successfully
- Each operation tracked in session state
- Session ends with complete summary
- Metrics exported with all operations

**Success Criteria**:
- ✅ Session state file created
- ✅ Summary includes all 3 operations
- ✅ Metrics show 3 pre-task + 3 post-edit executions
- ✅ All files created are listed in summary
- ✅ Next steps recommended in summary

**Troubleshooting**:
- Session not saved: Check session-hooks.yaml config
- Operations missing from summary: Verify logging enabled
- Metrics incomplete: Check metrics.enabled in config

---

### Scenario 2.4: Parallel Hook Execution

**Objective**: Test concurrent hook execution for independent operations

**Steps**:
1. Create multiple test files:
   ```bash
   mkdir -p /tmp/parallel-test
   echo 'const a = 1;' > /tmp/parallel-test/file1.js
   echo 'const b = 2;' > /tmp/parallel-test/file2.js
   echo 'const c = 3;' > /tmp/parallel-test/file3.js
   ```

2. Execute post-edit hooks in parallel:
   ```bash
   # Launch all hooks simultaneously
   npx claude-flow@alpha hooks post-edit --file /tmp/parallel-test/file1.js &
   npx claude-flow@alpha hooks post-edit --file /tmp/parallel-test/file2.js &
   npx claude-flow@alpha hooks post-edit --file /tmp/parallel-test/file3.js &

   # Wait for all to complete
   wait
   ```

3. Check execution metrics:
   ```bash
   node resources/scripts/hook-manager.js metrics
   ```

**Expected Results**:
- All 3 hooks execute concurrently
- Each completes independently without blocking
- Total execution time < sum of individual times
- No race conditions or conflicts

**Success Criteria**:
- ✅ All 3 files formatted correctly
- ✅ Parallel execution faster than sequential
- ✅ No file corruption or conflicts
- ✅ All memory updates successful

**Troubleshooting**:
- Race conditions: Add file locking in post-edit config
- Slow execution: Check max_parallel_tasks in performance config
- Memory conflicts: Use unique memory keys per file

---

### Scenario 2.5: Conditional Hook Execution

**Objective**: Test hooks with conditions and filters

**Steps**:
1. Create files of different types:
   ```bash
   mkdir -p /tmp/conditional-test
   echo 'const test = 1;' > /tmp/conditional-test/app.js
   echo 'SELECT * FROM users;' > /tmp/conditional-test/query.sql
   echo '# Documentation' > /tmp/conditional-test/README.md
   echo '{"key": "value"}' > /tmp/conditional-test/config.json
   ```

2. Execute post-edit hook on each (should filter):
   ```bash
   for file in /tmp/conditional-test/*; do
     npx claude-flow@alpha hooks post-edit --file "$file"
   done
   ```

3. Check which files were processed:
   ```bash
   # Verify formatting applied selectively
   node resources/scripts/hook-manager.js metrics

   # Check filter configuration
   cat ~/.claude-flow/hooks/post-edit/config.json | jq '.hooks["post-edit"].filters'
   ```

**Expected Results**:
- JavaScript file (.js): Formatted with prettier
- JSON file: Formatted with prettier
- Markdown file (.md): Formatted with prettier
- SQL file (.sql): Skipped (not in include_extensions)

**Success Criteria**:
- ✅ Included extensions processed
- ✅ Excluded extensions skipped
- ✅ Metrics show correct execution count
- ✅ No errors for skipped files

**Troubleshooting**:
- All files processed: Check filters.include_extensions
- Wrong files skipped: Verify extension matches exactly (.js not js)
- Errors on skip: Set continue_on_error: true

---

## Advanced Chain Testing

### Test 2.6: Error Recovery in Chain

**Objective**: Verify chain continues after non-fatal errors

**Steps**:
1. Create file that will fail formatting:
   ```bash
   echo 'this is not valid javascript {{{' > /tmp/error-test.js
   ```

2. Execute with error handling enabled:
   ```bash
   npx claude-flow@alpha hooks post-edit --file /tmp/error-test.js
   ```

3. Verify error was logged but chain continued:
   ```bash
   # Check logs
   tail ~/.claude-flow/logs/post-edit-hook.log

   # Check if memory update still happened
   npx claude-flow@alpha memory retrieve --key "hooks/post-edit/error-test"
   ```

**Expected Results**:
- Auto-format action fails (invalid syntax)
- Error logged to file
- Subsequent actions (memory-update) still execute
- Hook completes with partial success

**Success Criteria**:
- ✅ Error logged in hook logs
- ✅ Memory update completed despite format error
- ✅ Hook exits with non-zero code but continues
- ✅ Metrics show 1 failure, other successes

---

## Test Execution Summary

### Chain Test Suite
Run all chaining tests:

```bash
# Enable all hooks
node resources/scripts/hook-manager.js start

# Run chain tests
bash resources/scripts/hook-tester.sh --type post-edit

# Check metrics
node resources/scripts/hook-manager.js metrics
```

### Expected Performance

| Chain Type | Operations | Expected Time |
|------------|-----------|---------------|
| Pre→Post | 2 hooks | ~3 seconds |
| Edit→Format→Memory | 4 actions | ~2 seconds |
| Session Lifecycle | 7 operations | ~10 seconds |
| Parallel (3 files) | 3 concurrent | ~2 seconds |
| Conditional | 4 files (3 processed) | ~3 seconds |

### Success Rate Targets

- **Basic chains**: 100% success
- **Complex chains**: ≥95% success
- **Parallel execution**: ≥90% success
- **Error recovery**: 100% (with continue_on_error)

## Cleanup

```bash
# Remove test files
rm -rf /tmp/workflow-test.js /tmp/chain-test.js
rm -rf /tmp/parallel-test /tmp/conditional-test /tmp/error-test.js

# Reset metrics (optional)
node resources/scripts/hook-manager.js reset-metrics
```

## Next Steps

After chaining tests pass:
1. Proceed to Test 3: Error Handling
2. Review test-3-error-handling.md
3. Test failure scenarios and recovery

## References

- Hook manager: `resources/scripts/hook-manager.js`
- Configuration: `resources/templates/`
- Memory MCP: See CLAUDE.md for integration details
