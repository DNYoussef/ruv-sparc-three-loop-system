# Test 3: Error Handling and Recovery

## Overview
Test hook error handling, recovery mechanisms, and graceful degradation when failures occur. Verify hooks can recover from errors and continue operation.

## Prerequisites
- Test 1 (Basic Hooks) passed
- Test 2 (Hook Chaining) passed
- Hook manager running
- Error logging enabled in configs

## Test Scenarios

### Scenario 3.1: Invalid Hook Configuration

**Objective**: Verify validator catches configuration errors

**Steps**:
1. Create invalid YAML config:
   ```bash
   cat > /tmp/invalid-hook.yaml <<'EOF'
   hooks:
     pre-task:
       enabled: true
       actions
         - invalid-syntax  # Missing colon
       config:
         missing_closing_brace
   EOF
   ```

2. Run validator on invalid config:
   ```bash
   python resources/scripts/hook-validator.py --verbose 2>&1
   ```

3. Attempt to load invalid config:
   ```bash
   cp /tmp/invalid-hook.yaml ~/.claude-flow/hooks/pre-task/config.yaml
   npx claude-flow@alpha hooks pre-task --description "Test"
   ```

**Expected Results**:
- Validator detects YAML syntax error
- Specific error location identified (line number)
- Hook execution fails with clear error message
- System doesn't crash, error logged

**Success Criteria**:
- ✅ Validator reports syntax error
- ✅ Error message includes line number
- ✅ Hook refuses to execute with invalid config
- ✅ Original valid config can be restored

**Troubleshooting**:
- Restore valid config: `bash resources/scripts/hook-installer.sh`
- Clear error logs: `rm ~/.claude-flow/logs/*.log`

---

### Scenario 3.2: Missing Dependencies

**Objective**: Test behavior when required tools are missing

**Steps**:
1. Temporarily rename prettier (simulate missing dependency):
   ```bash
   # On Unix/Mac
   which prettier && mv $(which prettier) $(which prettier).bak

   # On Windows
   where prettier
   # Temporarily rename if found
   ```

2. Attempt to format file:
   ```bash
   echo 'const x={a:1};' > /tmp/format-test.js
   npx claude-flow@alpha hooks post-edit --file /tmp/format-test.js
   ```

3. Check error handling:
   ```bash
   # Check logs
   tail ~/.claude-flow/logs/post-edit-hook.log

   # Check if other actions still executed
   npx claude-flow@alpha memory retrieve --key "hooks/post-edit/format-test"
   ```

4. Restore prettier:
   ```bash
   # On Unix/Mac
   [ -f $(which prettier).bak ] && mv $(which prettier).bak $(which prettier)
   ```

**Expected Results**:
- Auto-format action fails (prettier not found)
- Error logged with descriptive message
- Subsequent actions (memory-update, quality-check) still execute
- Hook completes with partial success status

**Success Criteria**:
- ✅ Error logged: "prettier command not found"
- ✅ Hook continues to next action (continue_on_error: true)
- ✅ Memory update succeeds
- ✅ Exit code indicates partial failure

**Troubleshooting**:
- All actions fail: Check continue_on_error setting
- No error logged: Verify logging.enabled: true
- Prettier not restored: Run `npm install -g prettier`

---

### Scenario 3.3: File Permission Errors

**Objective**: Test handling of permission-denied errors

**Steps**:
1. Create read-only file:
   ```bash
   echo 'const test = 1;' > /tmp/readonly.js
   chmod 444 /tmp/readonly.js  # Read-only
   ```

2. Attempt to format (write operation):
   ```bash
   npx claude-flow@alpha hooks post-edit --file /tmp/readonly.js
   ```

3. Check error handling:
   ```bash
   # Verify error logged
   grep -i "permission" ~/.claude-flow/logs/post-edit-hook.log

   # Check hook metrics
   node resources/scripts/hook-manager.js metrics
   ```

4. Cleanup:
   ```bash
   chmod 644 /tmp/readonly.js
   rm /tmp/readonly.js
   ```

**Expected Results**:
- File write fails with permission error
- Error clearly identifies permission issue
- Hook logs error but doesn't crash
- File remains unchanged (read-only preserved)

**Success Criteria**:
- ✅ Permission error detected and logged
- ✅ Error message mentions file path
- ✅ Hook exits gracefully
- ✅ No partial/corrupted file writes

**Troubleshooting**:
- File modified: Check formatter isn't running with sudo
- No error: Verify file is actually read-only (ls -l)

---

### Scenario 3.4: Memory MCP Connection Failure

**Objective**: Test hook behavior when Memory MCP is unavailable

**Steps**:
1. Stop Memory MCP (simulate disconnection):
   ```bash
   # Note: This depends on your MCP setup
   # May need to restart Claude Desktop or stop MCP server
   ```

2. Execute hook that requires memory:
   ```bash
   npx claude-flow@alpha hooks session-start
   ```

3. Check fallback behavior:
   ```bash
   # Check logs for connection error
   grep -i "memory" ~/.claude-flow/logs/session-hooks.log

   # Verify session still started (degraded mode)
   ls ~/.claude-flow/sessions/
   ```

4. Restore Memory MCP:
   ```bash
   # Restart Claude Desktop or MCP server
   ```

**Expected Results**:
- Connection error detected and logged
- Hook falls back to local storage
- Session starts in degraded mode (no memory restore)
- Clear warning shown to user

**Success Criteria**:
- ✅ Error: "Failed to connect to Memory MCP"
- ✅ Fallback behavior activates
- ✅ Session still functional (basic features)
- ✅ Reconnection works after MCP restored

**Troubleshooting**:
- Hook crashes: Add try-catch around MCP calls
- No fallback: Verify fallback logic in config
- Can't restore MCP: Check Claude Desktop config

---

### Scenario 3.5: Timeout Handling

**Objective**: Test hook timeout and recovery

**Steps**:
1. Create slow-running test:
   ```bash
   cat > ~/.claude-flow/hooks/pre-task/slow-test.sh <<'EOF'
   #!/bin/bash
   sleep 30  # Exceeds default 5s timeout
   echo "Task complete"
   EOF
   chmod +x ~/.claude-flow/hooks/pre-task/slow-test.sh
   ```

2. Set low timeout in config:
   ```yaml
   # Edit pre-task-hook.yaml
   execution:
     timeout: 2000  # 2 seconds
   ```

3. Execute hook:
   ```bash
   npx claude-flow@alpha hooks pre-task --description "Slow task"
   ```

4. Check timeout handling:
   ```bash
   # Check logs
   grep -i "timeout" ~/.claude-flow/logs/pre-task-hook.log

   # Check metrics
   node resources/scripts/hook-manager.js metrics
   ```

**Expected Results**:
- Hook times out after 2 seconds
- Timeout error logged
- Process terminated gracefully
- Retry mechanism attempts if configured

**Success Criteria**:
- ✅ Timeout detected at configured threshold
- ✅ Error message: "Hook execution timed out"
- ✅ Process killed cleanly
- ✅ Retry attempted (if retry.enabled: true)

**Troubleshooting**:
- Timeout not working: Check execution.timeout value
- Process hangs: Verify timeout enforcement
- Restore config: `bash resources/scripts/hook-installer.sh`

---

### Scenario 3.6: Retry Mechanism

**Objective**: Test automatic retry on transient failures

**Steps**:
1. Create flaky test that succeeds on retry:
   ```bash
   cat > /tmp/flaky-test.sh <<'EOF'
   #!/bin/bash
   COUNTER_FILE="/tmp/retry-counter"
   COUNT=$(cat "$COUNTER_FILE" 2>/dev/null || echo 0)
   COUNT=$((COUNT + 1))
   echo $COUNT > "$COUNTER_FILE"

   if [ $COUNT -lt 3 ]; then
     echo "Attempt $COUNT: Failing" >&2
     exit 1
   else
     echo "Attempt $COUNT: Success"
     rm "$COUNTER_FILE"
     exit 0
   fi
   EOF
   chmod +x /tmp/flaky-test.sh
   ```

2. Configure retry in hook:
   ```yaml
   execution:
     retry:
       enabled: true
       max_attempts: 3
       backoff: exponential
   ```

3. Execute flaky hook:
   ```bash
   bash /tmp/flaky-test.sh
   ```

4. Verify retry behavior:
   ```bash
   # Check retry attempts in logs
   grep -i "retry\|attempt" ~/.claude-flow/logs/*.log
   ```

**Expected Results**:
- Attempt 1: Fails
- Attempt 2: Fails
- Attempt 3: Succeeds
- Hook completes successfully after retries

**Success Criteria**:
- ✅ 3 attempts logged
- ✅ Exponential backoff applied (increasing delays)
- ✅ Final attempt succeeds
- ✅ Overall hook status: success

**Troubleshooting**:
- No retries: Verify retry.enabled: true
- Too many retries: Check max_attempts setting
- Backoff not working: Verify backoff: exponential

---

### Scenario 3.7: Graceful Degradation

**Objective**: Test hook continues with reduced functionality

**Steps**:
1. Disable optional features:
   ```yaml
   # In post-edit-hook.json
   {
     "config": {
       "neural-training": { "enabled": false },
       "git-stage": { "enabled": false },
       "quality-check": { "enabled": false }
     }
   }
   ```

2. Execute hook:
   ```bash
   echo 'const x = 1;' > /tmp/degraded-test.js
   npx claude-flow@alpha hooks post-edit --file /tmp/degraded-test.js
   ```

3. Verify core functions still work:
   ```bash
   # Check if formatting happened
   cat /tmp/degraded-test.js

   # Check if memory update happened
   npx claude-flow@alpha memory retrieve --key "hooks/post-edit/degraded-test"
   ```

**Expected Results**:
- Core actions (auto-format, memory-update) execute
- Optional actions skipped gracefully
- No errors for disabled features
- Hook completes successfully

**Success Criteria**:
- ✅ File formatted (core functionality)
- ✅ Memory updated (core functionality)
- ✅ No errors about disabled features
- ✅ Performance improved (fewer actions)

**Troubleshooting**:
- Core actions fail: Check action dependencies
- Errors on disabled features: Add existence checks

---

## Error Recovery Testing

### Test 3.8: Automatic Error Correction

**Objective**: Test hooks can auto-fix common errors

**Steps**:
1. Create file with fixable errors:
   ```bash
   cat > /tmp/fixable.js <<'EOF'
   const x = 1
   const y = 2
   console.log(x, y)
   EOF
   ```

2. Run post-edit with auto-fix enabled:
   ```bash
   npx claude-flow@alpha hooks post-edit --file /tmp/fixable.js
   ```

3. Verify auto-fix applied:
   ```bash
   # Check semicolons added
   cat /tmp/fixable.js
   ```

**Expected Results**:
- Missing semicolons added automatically
- Code formatted to style guide
- No errors after auto-fix

**Success Criteria**:
- ✅ Semicolons present in output
- ✅ Code properly formatted
- ✅ No lint errors remain

---

## Test Execution Summary

### Error Handling Test Suite

```bash
# Run all error handling tests
bash resources/scripts/hook-tester.sh --profile

# Check error logs
tail -f ~/.claude-flow/logs/*.log

# Review metrics
node resources/scripts/hook-manager.js metrics
```

### Expected Error Rates

| Error Type | Detection Rate | Recovery Rate |
|------------|----------------|---------------|
| Config syntax | 100% | Manual fix |
| Missing deps | 100% | Graceful skip |
| Permissions | 100% | Error logged |
| Timeouts | 100% | Auto-retry |
| MCP connection | 100% | Fallback mode |

### Success Criteria Summary

- ✅ All errors detected and logged
- ✅ No crashes or hangs
- ✅ Graceful degradation works
- ✅ Retry mechanism succeeds
- ✅ Clear error messages
- ✅ Recovery possible

## Cleanup

```bash
# Remove test files
rm -f /tmp/*.js /tmp/*.sh /tmp/invalid-hook.yaml
rm -f /tmp/retry-counter

# Restore original configs
bash resources/scripts/hook-installer.sh

# Clear error logs (optional)
rm ~/.claude-flow/logs/*.log

# Reset metrics
node resources/scripts/hook-manager.js reset-metrics
```

## Next Steps

After error handling tests pass:
1. Review examples for real-world usage patterns
2. Integrate hooks into actual development workflow
3. Monitor production error rates
4. Tune timeout and retry settings

## References

- Error handling config: `resources/templates/*/execution.error_handling`
- Retry config: `resources/templates/*/execution.retry`
- Logging: `resources/templates/*/logging`
- Hook validator: `resources/scripts/hook-validator.py`
