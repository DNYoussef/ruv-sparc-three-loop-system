# Hooks Automation Resources

This directory contains scripts, templates, and utilities for implementing automated hook workflows with Claude Flow.

## Directory Structure

### scripts/
Automation scripts for hook management:

- **hook-installer.sh** - Install and configure Claude Flow hooks
- **hook-validator.py** - Validate hook configuration and dependencies
- **hook-manager.js** - Manage hook lifecycle and execution
- **hook-tester.sh** - Test hooks in sandbox environments

### templates/
Ready-to-use hook configuration templates:

- **pre-task-hook.yaml** - Pre-task automation template
- **post-edit-hook.json** - Post-edit formatting template
- **session-hooks.yaml** - Session management template

## Usage

### 1. Installation

```bash
# Install hooks with default configuration
bash scripts/hook-installer.sh

# Install with custom config path
bash scripts/hook-installer.sh --config /path/to/config.yaml
```

### 2. Validation

```bash
# Validate hook setup
python scripts/hook-validator.py

# Validate specific hook
python scripts/hook-validator.py --hook pre-task

# Generate validation report
python scripts/hook-validator.py --report hooks-validation.md
```

### 3. Management

```bash
# Start hook manager
node scripts/hook-manager.js start

# List active hooks
node scripts/hook-manager.js list

# Enable specific hook
node scripts/hook-manager.js enable pre-task

# Disable hook
node scripts/hook-manager.js disable post-edit
```

### 4. Testing

```bash
# Test all hooks
bash scripts/hook-tester.sh

# Test specific hook type
bash scripts/hook-tester.sh --type pre-task

# Test with sandbox
bash scripts/hook-tester.sh --sandbox
```

## Template Customization

### Pre-Task Hook Template

```yaml
# pre-task-hook.yaml
hooks:
  pre-task:
    enabled: true
    actions:
      - agent-assignment
      - resource-preparation
      - validation
    config:
      agent_selection: auto
      resource_cache: enabled
      validation_level: strict
```

### Post-Edit Hook Template

```json
{
  "hooks": {
    "post-edit": {
      "enabled": true,
      "actions": [
        "auto-format",
        "memory-update",
        "neural-training"
      ],
      "config": {
        "formatter": "prettier",
        "memory_key_prefix": "hooks/post-edit",
        "training_enabled": true
      }
    }
  }
}
```

### Session Hooks Template

```yaml
# session-hooks.yaml
hooks:
  session-start:
    enabled: true
    actions:
      - restore-context
      - load-memory
      - initialize-agents

  session-end:
    enabled: true
    actions:
      - persist-state
      - generate-summary
      - export-metrics
```

## Integration Points

### Claude Flow MCP
All scripts integrate with Claude Flow MCP for:
- Hook registration
- Event triggering
- State management
- Metrics collection

### Memory MCP
Templates support Memory MCP for:
- Context persistence
- Pattern storage
- Knowledge sharing

### Git Integration
Scripts include Git hooks for:
- Auto-commit on task completion
- Change tracking
- Version control

## Best Practices

1. **Start with templates** - Use provided templates as starting points
2. **Validate before deploy** - Always run hook-validator.py before production
3. **Test in sandbox** - Use hook-tester.sh to verify behavior
4. **Monitor performance** - Track hook execution time and impact
5. **Iterate gradually** - Enable one hook at a time, measure results
6. **Version control** - Keep hook configs in Git for rollback capability

## Troubleshooting

### Common Issues

**Hook not triggering**:
```bash
# Check hook registration
node scripts/hook-manager.js list

# Validate configuration
python scripts/hook-validator.py --hook <hook-name>
```

**Performance degradation**:
```bash
# Profile hook execution
bash scripts/hook-tester.sh --profile

# Check metrics
node scripts/hook-manager.js metrics
```

**Configuration errors**:
```bash
# Validate syntax
python scripts/hook-validator.py --syntax-only

# Test with dry-run
bash scripts/hook-tester.sh --dry-run
```

## Advanced Usage

### Custom Hook Development

1. Copy template from `templates/`
2. Modify actions and config
3. Validate with hook-validator.py
4. Test with hook-tester.sh
5. Deploy with hook-installer.sh

### Multi-Hook Workflows

Chain multiple hooks together:
```yaml
workflow:
  - hook: pre-task
    next: execute-task
  - hook: post-task
    next: update-memory
  - hook: session-end
```

### Conditional Execution

Add conditions to hooks:
```yaml
hooks:
  auto-format:
    enabled: true
    conditions:
      - file_extension: ['.js', '.ts', '.jsx', '.tsx']
      - file_size_max: 10000
      - git_tracked: true
```

## Performance Metrics

Track hook performance:
- Execution time per hook
- Token usage impact
- Success/failure rates
- Memory consumption
- CPU utilization

Use `hook-manager.js metrics` to view detailed statistics.

## Support

For help with resources:
1. Check script headers for usage documentation
2. Review templates for configuration examples
3. Run validators to identify issues
4. Consult main skill.md for integration guides

---

**Maintained by**: Claude Code Plugin System
**Last Updated**: 2025-11-02
