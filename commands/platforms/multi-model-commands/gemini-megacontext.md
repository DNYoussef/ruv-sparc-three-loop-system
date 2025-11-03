---
name: gemini-megacontext
binding: multi-model:gemini-megacontext
category: multi-model
version: 1.0.0
---

# /gemini-megacontext

Access Gemini's 1 million token context window for analyzing entire large codebases.

## Usage
```bash
/gemini-megacontext "<task>" [options]
```

## Parameters
- `task` - Analysis task description (required)
- `--context` - Directory or files to analyze (default: current directory)
- `--output` - Output file for analysis results (default: stdout)
- `--format` - Output format: markdown|json|text (default: markdown)

## Examples
```bash
# Analyze entire codebase
/gemini-megacontext "Analyze architecture patterns across entire codebase" --context src/

# Find all API endpoints
/gemini-megacontext "List all API endpoints with their auth requirements" --context src/

# Generate report
/gemini-megacontext "Security audit of entire application" --output security-report.md
```

## When to Use
- ✅ Analyzing large codebases (30K+ lines)
- ✅ Cross-file pattern detection
- ✅ Architecture documentation
- ✅ Comprehensive security audits
- ✅ Dependency mapping

## Capabilities
- **Context**: 1,000,000 tokens (~30,000 lines of code)
- **Model**: Gemini 2.0 Flash
- **Free tier**: 60 requests/min, 1000/day

## Implementation
```bash
# Executed via Gemini CLI
gemini "<task>" --files <context> --model gemini-2.0-flash
```

## Chains with
```bash
# Large codebase analysis → architecture design
/gemini-megacontext "Analyze codebase patterns" | \
/agent-architect "Design refactoring plan"
```

## See also
- `/gemini-search` - Real-time web information
- `/codex-auto` - Rapid prototyping
- `/claude-reason` - Best overall reasoning
