---
name: re:deep
binding: skill:reverse-engineering-deep
category: reverse-engineering
version: 1.0.0
---

# /re:deep

Deep reverse engineering with runtime execution and symbolic exploration (Levels 3-4: Dynamic Analysis + Symbolic Execution).

**Timebox**: 3-7 hours
**RE Levels**: 3 (Dynamic Analysis) + 4 (Symbolic Execution)

## Usage
```bash
/re:deep <binary-path> [options]
```

## Parameters
- `binary-path` - Path to binary/executable to analyze (required)
- `--level` - Specific level: 3 (dynamic only), 4 (symbolic only), or 3,4 (both, default)
- `--args` - Arguments to pass to binary during dynamic analysis (default: "")
- `--target-addr` - Target address for symbolic execution (hex, e.g., 0x401234)
- `--output` - Output directory (default: re-project/)
- `--sandbox` - Run in isolated sandbox for safety (default: true)
- `--store-findings` - Store in memory-mcp (default: true)

## Examples
```bash
# Full deep analysis with runtime + symbolic
/re:deep crackme.exe

# Dynamic analysis with arguments
/re:deep server.elf --level 3 --args "--port 8080"

# Symbolic execution to reach specific address
/re:deep binary.bin --level 4 --target-addr 0x401337

# Deep analysis with custom output
/re:deep challenge.exe --output ./re-deep-analysis/
```

## What It Does

### Level 3: Dynamic Analysis (≤1 hr)
1. Execute binary in sandbox (sandbox-validator)
2. Attach GDB with GEF/Pwndbg extensions
3. Set breakpoints at key functions (from Level 2 static analysis)
4. Trace execution flow and capture state
5. Dump memory regions and extract runtime secrets
6. Analyze crashes with stack traces

**Success Criteria**:
- ✅ Binary executed safely in sandbox
- ✅ Runtime state captured (registers, stack, heap)
- ✅ Secrets extracted (keys, passwords, tokens)
- ✅ Crash points identified (if applicable)

**Tools**: GDB+GEF, GDB+Pwndbg, ltrace, strace

### Level 4: Symbolic Execution (2-6 hrs)
1. Load binary into Angr symbolic execution engine
2. Define target state (address to reach)
3. Define avoid states (dead ends, infinite loops)
4. Explore paths symbolically with constraint solving
5. Synthesize inputs that reach target state
6. Generate Angr scripts for reproducibility

**Success Criteria**:
- ✅ Target state reached
- ✅ Input values synthesized
- ✅ Constraints solved (Z3 theorem proving)
- ✅ Angr script saved for future use

**Tools**: Angr, Z3, symbolic execution frameworks

## Decision Gate (Level 3 → Level 4)

After Level 3 completes, the skill asks:
- **Did dynamic analysis reveal sufficient runtime behavior?**
- **Do we need to explore ALL paths to reach a target state?**
- **Is manual analysis sufficient or need automated path exploration?**

If Level 3 answers your question → **EXIT EARLY** ✅
If need exhaustive path exploration → **PROCEED to Level 4** 🧠

## Agents Involved
- `RE-Runtime-Tracer` - GDB scripting and runtime tracing
- `RE-Symbolic-Solver` - Angr/Z3 symbolic execution
- `debugger` - General debugging assistance
- `sandbox-validator` - Safe binary execution
- `root-cause-detective` - Crash analysis
- `error-message-analyzer` - Stack trace parsing
- `memory-coordinator` - Store findings

## MCP Servers Used
- **memory-mcp**: Store runtime findings and symbolic solutions
- **filesystem**: Access binaries and sandbox files
- **sequential-thinking**: Complex decision trees for symbolic path selection
- **focused-changes**: Track which code paths were executed

## Output Structure
```
re-project/
├── dbg/
│   ├── breakpoints.txt         # GDB breakpoint script
│   ├── traces.log              # Execution traces
│   ├── memory-dumps/           # Memory snapshots
│   │   ├── 0x401000-stack.bin
│   │   └── 0x601000-heap.bin
│   ├── registers.txt           # Register state at breakpoints
│   └── runtime-secrets.txt     # Extracted keys/passwords
├── sym/
│   ├── angr-script.py          # Reproducible Angr script
│   ├── solutions.txt           # Synthesized inputs
│   ├── constraints.smt2        # Z3 constraint formulas
│   └── path-exploration.log    # Explored paths
├── notes/
│   ├── 003-dynamic-l3.md       # Dynamic analysis findings
│   └── 004-symbolic-l4.md      # Symbolic execution findings
└── artifacts/
    └── exploits/               # POC exploits if vulnerabilities found
```

## Security Warning ⚠️

**ALWAYS run in sandbox for untrusted binaries!**
- Use `--sandbox true` (default)
- Dynamic analysis executes potentially malicious code
- Sandbox isolates filesystem, network, and process access
- Review sandbox logs after execution

## Chains With
- `/re:quick` - Start with quick triage first
- `/re:firmware` - If binary is embedded firmware
- `/functionality-audit` - Validate reverse-engineered logic
- `/theater-detect` - Ensure findings are genuine

## Integration Notes

### Memory-MCP Tagging
```json
{
  "agent": "RE-Runtime-Tracer",
  "category": "reverse-engineering",
  "intent": "deep-analysis",
  "layer": "long_term",
  "project": "binary-analysis-2025-11-01",
  "keywords": ["dynamic", "symbolic", "runtime", "angr"],
  "re_level": "3-4",
  "binary_hash": "sha256:...",
  "target_address": "0x401337"
}
```

### Sandbox Integration
```python
# Safe execution wrapper
sandbox-validator.execute_safely(
    binary_path="suspicious.exe",
    args="--flag test",
    timeout=60,
    filesystem_isolation=True,
    network_disabled=True
)
```

### Sequential Thinking for Path Selection
```
QUESTION: "Should we explore this branch symbolically?"
REASONING:
- Branch contains comparison with user input ✓
- Static analysis shows interesting code after branch ✓
- No infinite loops detected ✓
DECISION: YES, add to symbolic exploration queue
```

## Performance Notes
- Level 3 is fast (≤1 hour) - runtime execution
- Level 4 is slow (2-6 hours) - explores MANY paths
- Use `--target-addr` to focus symbolic execution
- Define avoid states to skip dead code

## See Also
- `/re:quick` - Fast triage (Levels 1-2)
- `/re:firmware` - Firmware analysis (Level 5)
- `/re:dynamic` - Dynamic analysis only
- `/re:symbolic` - Symbolic execution only
