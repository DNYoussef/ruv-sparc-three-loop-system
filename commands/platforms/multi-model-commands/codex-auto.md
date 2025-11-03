---
name: codex-auto
binding: multi-model:codex-auto
category: multi-model
version: 1.0.0
---

# /codex-auto

Unattended rapid prototyping in sandboxed environment with Full Auto mode.

## Usage
```bash
/codex-auto "<task>" [options]
```

## Parameters
- `task` - Implementation task description (required)
- `--context` - Context files or directory (optional)
- `--sandbox` - Enable isolated sandbox (default: true)
- `--max-iterations` - Max auto-iterations (default: 5)
- `--output` - Output directory for generated code (default: current directory)

## Examples
```bash
# Rapid prototype
/codex-auto "Implement user authentication with JWT"

# With context
/codex-auto "Add pagination to user list API" --context src/api/users.js

# Fix failing test
/codex-auto "Fix test failure in test_auth.py" --context tests/test_auth.py

# Sandbox iteration (used by functionality-audit)
/codex-auto "Fix timeout in API handler" --sandbox true --max-iterations 5
```

## When to Use
- ✅ Rapid prototyping
- ✅ Quick feature implementations
- ✅ Auto-fixing test failures
- ✅ Boilerplate generation
- ✅ Exploratory coding

## Safety Features
- **Sandbox**: Isolated execution environment
- **Network**: DISABLED in sandbox
- **Scope**: CWD only (no parent directory access)
- **Isolation**: Seatbelt (macOS) / Docker

## Capabilities
- **Model**: GPT-4-Codex (via ChatGPT Plus subscription)
- **Mode**: Full Auto (unattended execution)
- **Iteration**: Automatic test-fix-retest loop

## Implementation
```bash
# Executed via Codex CLI
codex --full-auto "<task>" --context <files> --sandbox
```

## Integration with Functionality Audit
```python
# Used in /functionality-audit Phase 2
for failing_test in test_results:
    # Spawn Codex to auto-fix
    /codex-auto "Fix test: {test_name}" \
      --context {test_file} \
      --sandbox true \
      --max-iterations 5
```

## Chains with
```bash
# Research → prototype → test → fix
/gemini-search "Best practices for feature X" | \
/codex-auto "Implement feature X with best practices" | \
/functionality-audit --model codex-auto

# Quick prototype → polish
/codex-auto "Rapid prototype for user feature" | \
/style-audit
```

## See also
- `/codex-reasoning` - Alternative reasoning approach
- `/functionality-audit` - Uses Codex for auto-fixing
- `/gemini-megacontext` - Large context analysis
- `/claude-reason` - Best reasoning (default)
