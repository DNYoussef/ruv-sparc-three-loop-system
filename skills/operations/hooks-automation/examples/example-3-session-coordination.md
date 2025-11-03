# Example 3: Session Coordination - Context Restoration and State Management

## Scenario Description

**Problem**: When working across multiple sessions, developers lose:
- Context from previous work
- Active file positions
- Agent state and configurations
- Todo lists and task progress
- Performance metrics and insights

Manual restoration takes 5-10 minutes per session start.

**Solution**: Use session hooks to automatically:
1. Save complete session state on exit
2. Restore context on session start
3. Resume interrupted tasks
4. Generate summaries with insights
5. Export metrics for analysis

## Real-World Use Case

**Project**: Multi-week feature development across 30+ files
**Team**: Distributed team across 3 time zones
**Challenge**: Maintaining continuity across sessions

**Scenario**: Developer ends work at 6 PM, another developer starts at 9 AM next day. Session hooks ensure seamless handoff with full context restoration.

## Step-by-Step Walkthrough

### Step 1: Configure Session Hooks

```yaml
# ~/.claude-flow/hooks/session/config.yaml
hooks:
  session-start:
    enabled: true
    actions:
      - restore-context
      - load-memory
      - initialize-agents
      - load-preferences

    config:
      restore-context:
        enabled: true
        auto_restore: true
        restore_items:
          - working_directory
          - active_files
          - agent_state
          - todo_list
          - recent_commands

      load-memory:
        enabled: true
        layers:
          - short_term  # Last 24h
          - mid_term    # Last 7d
          - long_term   # 30d+
        max_items: 20

  session-end:
    enabled: true
    actions:
      - persist-state
      - generate-summary
      - export-metrics
      - cleanup-temp
      - backup-session

    config:
      persist-state:
        enabled: true
        output_file: "~/.claude-flow/sessions/{{session_id}}.json"
        state_items:
          - agent_metrics
          - memory_updates
          - completed_tasks
          - active_files
          - session_config
        compress: true

      generate-summary:
        enabled: true
        output_file: "~/.claude-flow/summaries/{{session_id}}.md"
        sections:
          - session_info
          - tasks_completed
          - agent_performance
          - code_changes
          - next_steps

      export-metrics:
        enabled: true
        metrics:
          - execution_time
          - token_usage
          - agent_usage
          - hook_performance
          - error_rate
```

### Step 2: Session Lifecycle Demonstration

#### Starting a New Session

```bash
# Start session with automatic restoration
SESSION_ID="feature-user-auth-$(date +%s)"

echo "Starting session: $SESSION_ID"
npx claude-flow@alpha hooks session-start

# Expected output:
# [INFO] Session started: feature-user-auth-1730563200
# [INFO] Restoring context from previous session
# [INFO] Loaded 15 memory items (short: 5, mid: 7, long: 3)
# [INFO] Restored working directory: /project/src
# [INFO] Restored 8 active files
# [INFO] Initialized 3 agents: coder, reviewer, tester
# [INFO] Loaded preferences from ~/.claude-flow/preferences.yaml
# [INFO] Session ready in 2.3s
```

#### Working During Session

```bash
# Perform development tasks
echo "Creating authentication module..."

# Pre-task automation
npx claude-flow@alpha hooks pre-task \
  --description "Create JWT authentication middleware"

# Create files
cat > src/middleware/auth.js <<'EOF'
const jwt = require('jsonwebtoken');

module.exports = async (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'No token' });

  try {
    req.user = jwt.verify(token, process.env.JWT_SECRET);
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
};
EOF

# Post-edit formatting
npx claude-flow@alpha hooks post-edit \
  --file src/middleware/auth.js \
  --memory-key "auth/middleware"

# Continue with more tasks...
```

#### Ending Session

```bash
# End session with full state persistence
echo "Ending session..."
npx claude-flow@alpha hooks session-end --export-metrics true

# Expected output:
# [INFO] Persisting session state...
# [INFO] Saved 23 memory updates
# [INFO] Recorded 12 completed tasks
# [INFO] Tracked 8 active files
# [INFO] Generating session summary...
# [INFO] Exporting metrics...
# [INFO] Session state saved: ~/.claude-flow/sessions/feature-user-auth-1730563200.json
# [INFO] Summary generated: ~/.claude-flow/summaries/feature-user-auth-1730563200.md
# [INFO] Metrics exported: ~/.claude-flow/metrics/feature-user-auth-1730563200.json
# [INFO] Session ended successfully
```

### Step 3: Reviewing Session Artifacts

#### Session State (JSON)

```bash
cat ~/.claude-flow/sessions/feature-user-auth-1730563200.json
```

```json
{
  "session_id": "feature-user-auth-1730563200",
  "started_at": "2025-11-02T10:00:00Z",
  "ended_at": "2025-11-02T14:30:00Z",
  "duration_seconds": 16200,
  "working_directory": "/project/src",
  "active_files": [
    "src/middleware/auth.js",
    "src/routes/auth.js",
    "tests/auth.test.js"
  ],
  "agent_metrics": {
    "coder": {
      "tasks_completed": 8,
      "files_created": 5,
      "lines_added": 234,
      "execution_time_ms": 12450
    },
    "reviewer": {
      "reviews_completed": 3,
      "issues_found": 7,
      "issues_fixed": 7
    },
    "tester": {
      "tests_written": 15,
      "test_coverage": "92%"
    }
  },
  "completed_tasks": [
    {
      "id": "AUTH-001",
      "description": "Create JWT middleware",
      "agent": "coder",
      "completed_at": "2025-11-02T10:45:00Z"
    }
  ]
}
```

#### Session Summary (Markdown)

```bash
cat ~/.claude-flow/summaries/feature-user-auth-1730563200.md
```

```markdown
# Session Summary: feature-user-auth-1730563200

**Started**: 2025-11-02 10:00:00 UTC
**Ended**: 2025-11-02 14:30:00 UTC
**Duration**: 4 hours 30 minutes

## Tasks Completed (12)

1. ✅ Create JWT authentication middleware
   - Agent: coder
   - Files: src/middleware/auth.js
   - Lines added: 45

2. ✅ Implement login and register routes
   - Agent: coder
   - Files: src/routes/auth.js
   - Lines added: 87

3. ✅ Write authentication tests
   - Agent: tester
   - Files: tests/auth.test.js
   - Test coverage: 92%

[... 9 more tasks ...]

## Agent Performance

### Coder
- Tasks: 8 completed
- Files created: 5
- Lines added: 234
- Avg execution time: 1.56s

### Reviewer
- Reviews: 3 completed
- Issues found: 7
- Issues fixed: 7
- Success rate: 100%

### Tester
- Tests written: 15
- Test coverage: 92%
- All tests passing: ✅

## Code Changes

### Files Created (5)
- src/middleware/auth.js
- src/routes/auth.js
- src/utils/jwt.js
- tests/auth.test.js
- docs/api/auth.md

### Files Modified (3)
- src/app.js (+12, -3)
- package.json (+2, -0)
- README.md (+15, -0)

## Next Steps

1. 🔲 Add refresh token rotation
2. 🔲 Implement rate limiting
3. 🔲 Add OAuth2 providers
4. 🔲 Deploy to staging environment
5. 🔲 Performance testing

## Recommendations

- **Security**: Consider adding 2FA support
- **Performance**: Cache JWT verification results
- **Testing**: Add E2E tests for auth flows
- **Monitoring**: Add auth metrics to dashboard
```

### Step 4: Restoring from Previous Session

```bash
# Next day: Restore from previous session
echo "Restoring session from yesterday..."

# Find latest session
LATEST_SESSION=$(ls -t ~/.claude-flow/sessions/*.json | head -1)
SESSION_ID=$(basename "$LATEST_SESSION" .json)

echo "Restoring session: $SESSION_ID"

# Restore state
npx claude-flow@alpha hooks session-restore \
  --session-id "$SESSION_ID"

# Expected output:
# [INFO] Validating session: feature-user-auth-1730563200
# [INFO] Session age: 16 hours (valid, < 24h threshold)
# [INFO] Restoring session state...
# [INFO] Restored working directory: /project/src
# [INFO] Restored 8 active files
# [INFO] Restored agent state (coder, reviewer, tester)
# [INFO] Restored todo list (5 pending tasks)
# [INFO] Loaded 23 memory items
# [INFO] Session restored successfully
# [INFO] Resuming from: Next Steps section

# Continue work
echo "Continuing with next task..."
npx claude-flow@alpha hooks pre-task \
  --description "Add refresh token rotation"
```

## Complete Workflow Example

**Scenario**: Multi-day feature development with session continuity

```bash
#!/bin/bash
# multi-session-development.sh

echo "=== Day 1: Start Feature Development ==="

# Session 1: Initial implementation
SESSION_1="feature-payment-$(date +%s)"
npx claude-flow@alpha hooks session-start

# Implement core features
npx claude-flow@alpha hooks pre-task --description "Create payment service"
# ... development work ...

# End day 1
npx claude-flow@alpha hooks session-end --export-metrics true

echo "=== Day 2: Continue Development ==="

# Session 2: Restore and continue
npx claude-flow@alpha hooks session-restore --session-id "$SESSION_1"

# Continue from where left off
npx claude-flow@alpha hooks pre-task --description "Add payment webhooks"
# ... more development ...

# End day 2
npx claude-flow@alpha hooks session-end --export-metrics true

echo "=== Day 3: Final Testing and Deployment ==="

# Session 3: Final push
npx claude-flow@alpha hooks session-restore --session-id "$SESSION_1"

# Complete feature
npx claude-flow@alpha hooks pre-task --description "Integration testing and deployment"
# ... testing and deployment ...

# End session
npx claude-flow@alpha hooks session-end --export-metrics true

echo "Feature complete! Review summaries:"
ls -lt ~/.claude-flow/summaries/ | head -3
```

## Expected Outcomes

### Time Savings

| Activity | Without Session Hooks | With Session Hooks | Savings |
|----------|---------------------|-------------------|---------|
| Context restoration | ~8 minutes | ~5 seconds | 96x faster |
| Finding active files | ~3 minutes | Automatic | 100% |
| Resuming work | ~5 minutes | ~10 seconds | 30x faster |
| Generating summary | ~15 minutes | ~2 seconds | 450x faster |
| Total per session | ~31 minutes | ~17 seconds | 109x faster |

### Quality Improvements

1. **100% context preservation** - Nothing lost between sessions
2. **Automated documentation** - Every session summarized
3. **Metrics tracking** - Performance insights automatically
4. **Seamless handoffs** - Team members can continue work instantly

## Tips and Best Practices

### Tip 1: Auto-Save During Session

```yaml
session:
  auto_save_interval: 300  # Save every 5 minutes
```

### Tip 2: Session Naming Convention

```yaml
session:
  naming:
    pattern: "{{project}}-{{feature}}-{{timestamp}}"
    include_git_branch: true
```

### Tip 3: Limit Session Count

```yaml
session:
  max_sessions: 50  # Keep only 50 most recent
```

### Tip 4: Backup Important Sessions

```yaml
backup-session:
  enabled: true
  backup_dir: "~/.claude-flow/backups"
  retention_days: 30
```

### Tip 5: Session Replay (Advanced)

```yaml
advanced:
  replay:
    enabled: true
    record_actions: true
```

### Tip 6: Monitor Session Health

```bash
# Check session storage
du -sh ~/.claude-flow/sessions/

# List recent sessions
ls -lht ~/.claude-flow/sessions/ | head -10

# Verify session integrity
python resources/scripts/hook-validator.py --check-sessions
```

### Tip 7: Cleanup Old Sessions

```bash
# Remove sessions older than 30 days
find ~/.claude-flow/sessions/ -name "*.json" -mtime +30 -delete
```

### Tip 8: Export for Reporting

```bash
# Generate weekly report from session metrics
for session in ~/.claude-flow/sessions/*.json; do
  cat "$session" | jq '{id, duration_seconds, tasks: .completed_tasks | length}'
done | jq -s 'add'
```

### Tip 9: Team Sharing

```bash
# Share session with team (remove sensitive data first)
cat ~/.claude-flow/sessions/session-123.json | \
  jq 'del(.sensitive_data)' > shared-session.json
```

### Tip 10: Session Templates

Create templates for common workflows:

```yaml
# ~/.claude-flow/templates/feature-development.yaml
session_template:
  agents: [coder, reviewer, tester]
  memory_layers: [short_term, mid_term]
  auto_save: true
  export_metrics: true
```

## Troubleshooting

### Issue: Session Not Restoring

**Solution**:
```bash
# Check session exists
ls ~/.claude-flow/sessions/

# Validate session file
cat ~/.claude-flow/sessions/session-123.json | jq .

# Check age
python -c "import json; print(json.load(open('session.json'))['ended_at'])"
```

### Issue: Large Session Files

**Solution**:
```yaml
# Enable compression
persist-state:
  compress: true
  format: json

# Limit memory items
load-memory:
  max_items: 10
```

### Issue: Missing Summaries

**Solution**:
```yaml
# Verify summary generation enabled
generate-summary:
  enabled: true
  output_file: "~/.claude-flow/summaries/{{session_id}}.md"
```

## Summary

Session coordination provides:
- **96x faster** context restoration
- **100% context** preservation
- **Automated documentation** for every session
- **Seamless handoffs** between developers
- **Performance insights** from metrics

By automating session management, developers maintain continuity across work sessions and gain valuable insights into productivity patterns.

## References

- Session hooks config: `resources/templates/session-hooks.yaml`
- Hook manager: `resources/scripts/hook-manager.js`
- Memory MCP integration: See CLAUDE.md
- Main skill: `skill.md`
